"""Policy transforms for the AgiBot G01 robot."""

from collections.abc import Sequence
import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.transforms as transforms


G01_ACTION_DIM = 16
G01_FULL_STATE_DIM = 163
G01_FULL_ACTION_DIM = 36

G01_STATE_INDICES = tuple(range(28, 35)) + tuple(range(35, 42)) + (0, 1)
G01_ACTION_INDICES = tuple(range(16, 23)) + tuple(range(23, 30)) + (0, 1)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _slice_last_dim(value, *, full_dim: int, indices: Sequence[int], name: str) -> np.ndarray:
    array = _to_numpy(value)
    dim = array.shape[-1]
    if dim == G01_ACTION_DIM:
        return array.copy()
    if dim == full_dim:
        return array[..., list(indices)].copy()
    raise ValueError(f"G01 {name} must have last dimension {G01_ACTION_DIM} or {full_dim}; got {array.shape}.")


def slice_g01_state(state) -> np.ndarray:
    return _slice_last_dim(state, full_dim=G01_FULL_STATE_DIM, indices=G01_STATE_INDICES, name="state")


def slice_g01_actions(actions) -> np.ndarray:
    return _slice_last_dim(actions, full_dim=G01_FULL_ACTION_DIM, indices=G01_ACTION_INDICES, name="actions")


@dataclasses.dataclass(frozen=True)
class G01ACOTInputs(transforms.DataTransformFn):
    """Inputs for AgiBot G01 ACoT policies.

    Expected robot observation:
    - images: top_head, hand_left, hand_right
    - state: either raw G01 [163] or executable [16]
    - actions: optional raw G01 [horizon, 36] or executable [horizon, 16]
    """

    action_dim: int
    acot_action_generation: Sequence[Sequence[int]] | None = None
    action_mask: Sequence[bool] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(slice_g01_state(data["state"]), self.action_dim)

        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera not in data["images"]:
                raise ValueError(f"Camera {camera} not found in data")

            img = _to_numpy(data["images"][camera])
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            images[self.rename_map[camera]] = img

        inputs = {
            "image": images,
            "image_mask": {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS},
            "state": state,
        }

        if "actions" in data:
            actions = slice_g01_actions(data["actions"])
            if self.acot_action_generation is not None:
                action_horizons = self.acot_action_generation[0]
                joint_action_shifts = self.acot_action_generation[1]
                for idx, key in enumerate(("coarse_actions", "actions")):
                    action_horizon = action_horizons[idx]
                    joint_action_shift = joint_action_shifts[idx]
                    required_length = (action_horizon - 1) * joint_action_shift + 1
                    if len(actions) < required_length:
                        raise ValueError(
                            f"G01 {key} needs {required_length} action frames; got {len(actions)}."
                        )
                    inputs[key] = actions[:required_length:joint_action_shift].copy()
            else:
                inputs["actions"] = actions

        for key in ("coarse_actions", "actions"):
            if key not in inputs:
                continue
            if self.action_mask is not None:
                action_mask = np.asarray(self.action_mask)[: inputs[key].shape[-1]]
                inputs[key][..., action_mask] = 0
            inputs[key] = transforms.pad_to_dim(inputs[key], self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G01ACOTOutputs(transforms.DataTransformFn):
    """Outputs executable G01 action chunks."""

    def __call__(self, data: dict) -> dict:
        return {
            key: np.asarray(data[key][..., :G01_ACTION_DIM])
            for key in ("coarse_actions", "actions")
            if key in data
        }
