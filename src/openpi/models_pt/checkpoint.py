from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_STATE_DICT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.safetensors",
    "state_dict.safetensors",
    "model.pt",
    "pytorch_model.pt",
    "state_dict.pt",
)


def find_converted_checkpoint(checkpoint_dir: Path | str) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    for filename in _STATE_DICT_FILENAMES:
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate
    return None


def _load_safetensors(path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            f"Found {path.name}, but safetensors is not installed. Install safetensors or provide a .pt state_dict."
        ) from exc
    return load_file(str(path), device=str(device))


def load_state_dict(path: Path | str, *, device: torch.device) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return _load_safetensors(path, device=device)

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Converted Torch checkpoint must contain a state_dict, got {type(payload)!r}.")
    return payload


def load_converted_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    strict: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    logger.info("Loading converted Torch checkpoint from %s", checkpoint_path)
    state_dict = load_state_dict(checkpoint_path, device=device)
    if hasattr(model, "load_converted_state_dict"):
        model.load_converted_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)
    return checkpoint_path


def maybe_load_converted_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path | str,
    *,
    device: torch.device,
    strict: bool = False,
    allow_missing: bool = True,
) -> Path | None:
    checkpoint_path = find_converted_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        message = (
            f"No converted Torch checkpoint found under {checkpoint_dir}. Expected one of: "
            f"{', '.join(_STATE_DICT_FILENAMES)}."
        )
        if allow_missing:
            logger.warning("%s Running with initialized Torch skeleton weights.", message)
            return None
        raise FileNotFoundError(message)

    logger.info("Loading converted Torch checkpoint from %s", checkpoint_path)
    return load_converted_checkpoint(model, checkpoint_path, device=device, strict=strict)
