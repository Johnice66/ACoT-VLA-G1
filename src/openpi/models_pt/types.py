from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np
import torch


IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def _to_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(np.asarray(value).copy())
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device)


def _ensure_batch(tensor: torch.Tensor, *, min_rank: int) -> torch.Tensor:
    if tensor.ndim == min_rank - 1:
        return tensor.unsqueeze(0)
    return tensor


def _image_to_float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = _to_tensor(value, device=device)
    tensor = _ensure_batch(tensor, min_rank=4)

    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
    else:
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() > 0 and tensor.amin() >= 0 and tensor.amax() > 1.5:
            tensor = tensor / 255.0 * 2.0 - 1.0

    return tensor


@dataclasses.dataclass
class TorchObservation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        device: torch.device,
        image_keys: tuple[str, ...] = IMAGE_KEYS,
    ) -> "TorchObservation":
        if "image" not in data:
            raise ValueError('TorchObservation requires an "image" dictionary.')
        if "state" not in data:
            raise ValueError('TorchObservation requires a "state" array.')

        images = {
            key: _image_to_float_tensor(data["image"][key], device=device)
            for key in image_keys
            if key in data["image"]
        }
        missing_images = set(image_keys) - set(images)
        if missing_images:
            raise ValueError(f"images dict missing keys for Torch ACoT: {sorted(missing_images)}")

        batch_shape = next(iter(images.values())).shape[:1]
        raw_masks = data.get("image_mask", {})
        image_masks = {}
        for key in image_keys:
            mask = raw_masks.get(key, np.ones(batch_shape, dtype=np.bool_))
            image_masks[key] = _ensure_batch(_to_tensor(mask, device=device, dtype=torch.bool), min_rank=1)

        state = _ensure_batch(_to_tensor(data["state"], device=device, dtype=torch.float32), min_rank=2)

        tokenized_prompt = None
        tokenized_prompt_mask = None
        if "tokenized_prompt" in data or "tokenized_prompt_mask" in data:
            if "tokenized_prompt" not in data or "tokenized_prompt_mask" not in data:
                raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
            tokenized_prompt = _ensure_batch(
                _to_tensor(data["tokenized_prompt"], device=device, dtype=torch.long),
                min_rank=2,
            )
            tokenized_prompt_mask = _ensure_batch(
                _to_tensor(data["tokenized_prompt_mask"], device=device, dtype=torch.bool),
                min_rank=2,
            )

        return cls(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    def to(self, device: torch.device) -> "TorchObservation":
        return TorchObservation(
            images={key: value.to(device=device) for key, value in self.images.items()},
            image_masks={key: value.to(device=device) for key, value in self.image_masks.items()},
            state=self.state.to(device=device),
            tokenized_prompt=None if self.tokenized_prompt is None else self.tokenized_prompt.to(device=device),
            tokenized_prompt_mask=None
            if self.tokenized_prompt_mask is None
            else self.tokenized_prompt_mask.to(device=device),
        )
