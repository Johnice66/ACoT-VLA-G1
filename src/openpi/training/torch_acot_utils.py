from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from openpi.models_pt.config import TorchACOTConfig
from openpi.shared import normalize
from openpi.training import config as train_config_lib


_RGB_VIDEO_KEYS = {
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
}


def torch_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def is_depth_feature(key: str, feature: dict[str, Any]) -> bool:
    video_info = feature.get("video_info", {})
    return key.endswith("_depth") or bool(video_info.get("video.is_depth_map", False))


def copy_json_without_depth(src: Path, dst: Path) -> None:
    info = json.loads(src.read_text(encoding="utf-8"))
    features = info.get("features", {})
    removed = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and is_depth_feature(key, feature)
    }
    info["features"] = {key: value for key, value in features.items() if key not in removed}
    info["total_videos"] = sum(
        1
        for feature in info["features"].values()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ) * int(info.get("total_episodes", 0))
    dst.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_jsonl_without_depth(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            item = json.loads(line)
            stats = item.get("stats")
            if isinstance(stats, dict):
                item["stats"] = {
                    key: value
                    for key, value in stats.items()
                    if not key.endswith("_depth") and key not in {
                        "observation.images.head_depth",
                        "observation.images.hand_left_depth",
                        "observation.images.hand_right_depth",
                    }
                }
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


def replace_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src.resolve(), dst)


def prepare_rgb_only_view(data_root: Path, view_dir: Path) -> Path:
    data_root = data_root.resolve()
    required = [data_root / "meta" / "info.json", data_root / "data", data_root / "videos"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset is not extracted or incomplete. Missing: {missing}")

    view_dir = view_dir.resolve()
    view_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = view_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    copy_json_without_depth(data_root / "meta" / "info.json", meta_dir / "info.json")

    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy2(data_root / "meta" / name, meta_dir / name)
    copy_jsonl_without_depth(data_root / "meta" / "episodes_stats.jsonl", meta_dir / "episodes_stats.jsonl")

    replace_symlink(data_root / "data", view_dir / "data")
    replace_symlink(data_root / "videos", view_dir / "videos")
    return view_dir


def torch_config(train_config: train_config_lib.TrainConfig, args: Any) -> TorchACOTConfig:
    config = TorchACOTConfig.from_jax_config(train_config.model)
    config = dataclasses.replace(config, dtype=args.torch_dtype).with_backbone(args.torch_backbone)
    return config


def data_config(train_config: train_config_lib.TrainConfig, repo_id: Path):
    assets = train_config.data.assets
    if assets.asset_id is None:
        original_repo_id = getattr(train_config.data, "repo_id", None)
        if isinstance(original_repo_id, str) and original_repo_id:
            assets = dataclasses.replace(assets, asset_id=Path(original_repo_id).name)
    data_factory = dataclasses.replace(train_config.data, repo_id=str(repo_id), assets=assets)
    return data_factory.create(train_config.assets_dirs, train_config.model)


def load_norm_stats(path: Path):
    norm_stats_dir = path.parent if path.name == "norm_stats.json" else path
    return normalize.load(norm_stats_dir)


def is_base_paligemma_llm_parameter(name: str) -> bool:
    if name.startswith("paligemma_llm.embedder."):
        return True
    if name.startswith("paligemma_llm.final_norms.0."):
        return True
    base_expert_parts = (
        ".qkv_einsums.0.",
        ".q_einsums.0.",
        ".kv_einsums.0.",
        ".attn_vec_einsums.0.",
        ".pre_attention_norms.0.",
        ".pre_ffw_norms.0.",
        ".mlps.0.",
    )
    return name.startswith("paligemma_llm.layers.") and any(part in name for part in base_expert_parts)


def set_trainable_scope(model: torch.nn.Module, scope: str) -> None:
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    if scope == "jax_frozen_llm":
        for name, parameter in model.named_parameters():
            frozen = is_base_paligemma_llm_parameter(name)
            parameter.requires_grad_(not frozen)
        return

    if scope != "action_heads":
        raise ValueError(f"Unknown trainable scope: {scope}")

    trainable_prefixes = (
        "coarse_action_in_proj.",
        "action_in_proj.",
        "coarse_time_mlp.",
        "time_mlp.",
        "coarse_action_out_proj.",
        "action_out_proj.",
        "explicit_action_reasoner.",
        "implicit_action_reasoner.",
        "implicit_action_reasoner_interact.",
        "explicit_action_reason_proj.",
        "implicit_action_reason_proj.",
        "action_reasoning_fusion.",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))


def grad_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = torch.sum(torch.square(parameter.grad.detach().float()))
        total = value if total is None else total + value
    if total is None:
        return torch.zeros(())
    return torch.sqrt(total)


def as_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(np.asarray(value).copy(), device=device)


def deterministic_training_randoms(
    *,
    actions: torch.Tensor,
    coarse_actions: torch.Tensor,
    seed: int,
    step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed + step)
    action_noise = rng.normal(size=tuple(actions.shape)).astype(np.float32)
    coarse_action_noise = rng.normal(size=tuple(coarse_actions.shape)).astype(np.float32)
    timestep = (rng.beta(1.5, 1.0, size=(actions.shape[0],)).astype(np.float32) * 0.999) + 0.001
    return {
        "action_noise": torch.from_numpy(action_noise).to(device=device),
        "coarse_action_noise": torch.from_numpy(coarse_action_noise).to(device=device),
        "timestep": torch.from_numpy(timestep).to(device=device),
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


_torch_dtype = torch_dtype
_prepare_rgb_only_view = prepare_rgb_only_view
_torch_config = torch_config
_data_config = data_config
_load_norm_stats = load_norm_stats
_set_trainable_scope = set_trainable_scope
_grad_norm = grad_norm
_as_tensor = as_tensor
_deterministic_training_randoms = deterministic_training_randoms
_scalar = scalar
