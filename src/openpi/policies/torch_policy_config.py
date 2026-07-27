from __future__ import annotations

import dataclasses
import logging
import pathlib
import urllib.parse
from typing import Any, Literal

import openpi.shared.download as download
import openpi.transforms as transforms
from openpi.models_pt import ACOTVLATorch
from openpi.models_pt import checkpoint as torch_checkpoint
from openpi.models_pt.config import BackboneMode, TorchACOTConfig
from openpi.policies import torch_policy as _torch_policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


TorchDType = Literal["float32", "bfloat16", "float16"]


def _torch_dtype(dtype: str):
    import torch

    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def _resolve_checkpoint_dir(checkpoint_dir: pathlib.Path | str, *, allow_missing: bool) -> pathlib.Path:
    checkpoint_str = str(checkpoint_dir)
    parsed = urllib.parse.urlparse(checkpoint_str)
    if parsed.scheme == "":
        path = pathlib.Path(checkpoint_str)
        if path.exists() or not allow_missing:
            return pathlib.Path(download.maybe_download(checkpoint_str))
        logging.warning("Torch checkpoint directory %s does not exist yet; running with skeleton weights.", path)
        return path
    return pathlib.Path(download.maybe_download(checkpoint_str))


def _load_norm_stats_if_available(
    checkpoint_dir: pathlib.Path,
    data_config: _config.DataConfig,
) -> dict[str, transforms.NormStats] | None:
    if data_config.asset_id is None:
        logging.warning("Torch policy has no asset_id; running without normalization stats.")
        return None

    try:
        return _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    except FileNotFoundError:
        logging.warning(
            "Torch policy could not find norm stats under %s; running without them.",
            checkpoint_dir / "assets",
        )
        return None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    device: str,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    strict_checkpoint: bool = False,
    allow_missing_checkpoint: bool = True,
    checkpoint_path: pathlib.Path | str | None = None,
    backbone: BackboneMode = "full",
    dtype: TorchDType | None = None,
) -> _torch_policy.TorchPolicy:
    """Create the Torch/NPU policy from an existing OpenPI training config.

    The config and transforms are reused from the JAX project, but model execution is delegated to PyTorch.
    Converted Torch weights are expected as state_dict/SafeTensors files inside the checkpoint directory.
    """

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, allow_missing=allow_missing_checkpoint)
    device_obj = _torch_policy.resolve_torch_device(device)

    torch_model_config = TorchACOTConfig.from_jax_config(train_config.model).with_backbone(backbone)
    if dtype is not None:
        torch_model_config = dataclasses.replace(torch_model_config, dtype=dtype)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        norm_stats = data_config.norm_stats
    if norm_stats is None:
        norm_stats = _load_norm_stats_if_available(checkpoint_dir, data_config)

    model = ACOTVLATorch(torch_model_config)
    model.to(device=device_obj, dtype=_torch_dtype(torch_model_config.dtype))
    if checkpoint_path is not None:
        loaded_checkpoint = torch_checkpoint.load_converted_checkpoint(
            model,
            checkpoint_path,
            device=device_obj,
            strict=strict_checkpoint,
        )
    else:
        loaded_checkpoint = torch_checkpoint.maybe_load_converted_checkpoint(
            model,
            checkpoint_dir,
            device=device_obj,
            strict=strict_checkpoint,
            allow_missing=allow_missing_checkpoint,
        )
    model.to(device=device_obj, dtype=_torch_dtype(torch_model_config.dtype))

    metadata = dict(train_config.policy_metadata or {})
    metadata.update(
        {
            "config": train_config.name,
            "converted_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
            "checkpoint_dir": str(checkpoint_dir),
            "torch_backbone": backbone,
            "torch_dtype": torch_model_config.dtype,
        }
    )

    return _torch_policy.TorchPolicy(
        model,
        device=device_obj,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
    )
