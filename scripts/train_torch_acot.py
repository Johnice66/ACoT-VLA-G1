from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import tyro

from openpi.models_pt import ACOTVLATorch
from openpi.models_pt import checkpoint as torch_checkpoint
from openpi.models_pt.device import resolve_torch_device
from openpi.models_pt.types import TorchObservation
from openpi.training import config as train_config_lib
from openpi.training import data_loader as openpi_data_loader
from openpi.training import torch_acot_utils as train_utils
from openpi.training import torch_distributed


@dataclasses.dataclass
class Args:
    """Train Torch ACoT-VLA on real LeRobot data.

    This is the PyTorch/NPU training entry. The original scripts/train.py is the
    JAX training entry and cannot run on Ascend NPU.
    """

    config_name: str = "acot_icra_simulation_challenge_reasoning_to_action"
    exp_name: str = tyro.MISSING
    checkpoint_base_dir: Path = Path("checkpoints_torch")
    data_root: Path = Path("vla-dataset/open-door")
    dataset_mode: Literal["go2_rgb", "libero"] = "go2_rgb"
    rgb_only: bool = True
    rgb_view_dir: Path = Path("tmp/acot_stage10_rgb_open_door_view")
    device: Literal["cpu", "cuda", "npu"] = "npu"
    torch_backbone: Literal["skeleton", "siglip", "gemma", "full"] = "full"
    torch_dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    trainable_dtype: Literal["model", "float32"] = "model"
    init_checkpoint_path: Path | None = Path("tmp/acot_stage6_gemma_siglip_state_dict.safetensors")
    strict: bool = False
    norm_stats_path: Path | None = None
    skip_norm_stats: bool = True
    trainable_scope: Literal["action_heads", "jax_frozen_llm", "all"] = "action_heads"
    task_indexes: tuple[int, ...] = ()
    batch_size: int = 1
    num_workers: int = 0
    train_steps: int = 1
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    seed: int = 0
    deterministic: bool = False
    shuffle: bool = True
    log_interval: int = 1
    eval_interval: int = 0
    eval_batches: int = 4
    fixed_loss_fixture: Path | None = None
    fixed_loss_interval: int = 0
    update_debug_interval: int = 0
    save_interval: int = 0
    save_final_checkpoint: bool = True
    overwrite: bool = False
    output_dir: Path | None = None
    summary_path: Path | None = None
    distributed: torch_distributed.DistributedMode = "auto"
    dist_backend: torch_distributed.DistributedBackend = "auto"
    local_rank: int | None = None
    ddp_find_unused_parameters: bool = True


class _TrainingLossModule(torch.nn.Module):
    def __init__(self, model: ACOTVLATorch):
        super().__init__()
        self.model = model

    def forward(
        self,
        observation: TorchObservation,
        actions: torch.Tensor,
        coarse_actions: torch.Tensor,
        *,
        timestep: torch.Tensor,
        action_noise: torch.Tensor,
        coarse_action_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model.compute_training_loss(
            observation,
            actions=actions,
            coarse_actions=coarse_actions,
            timestep=timestep,
            action_noise=action_noise,
            coarse_action_noise=coarse_action_noise,
        )


def _checkpoint_dir(args: Args) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.checkpoint_base_dir / args.config_name / args.exp_name


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to save a .safetensors checkpoint.") from exc
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(path))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _configure_reproducibility(args: Args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested, but torch_npu is not installed or not visible in this environment.") from exc
        manual_seed_all = getattr(torch.npu, "manual_seed_all", None)
        if manual_seed_all is not None:
            manual_seed_all(args.seed)
    elif args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)


def _prepare_repo_id(args: Args, dist_context: torch_distributed.DistributedContext) -> Path:
    if args.dataset_mode == "libero":
        return args.data_root.resolve()
    if args.dataset_mode != "go2_rgb":
        raise ValueError(f"Unsupported dataset_mode: {args.dataset_mode!r}.")
    if not args.rgb_only:
        return args.data_root.resolve()

    if dist_context.is_main_process:
        repo_id = train_utils._prepare_rgb_only_view(args.data_root, args.rgb_view_dir)  # pylint: disable=protected-access
    else:
        repo_id = args.rgb_view_dir.resolve()
    torch_distributed.barrier(dist_context)
    return repo_id


def _load_training_objects(args: Args, dist_context: torch_distributed.DistributedContext):
    repo_id = _prepare_repo_id(args, dist_context)
    device = resolve_torch_device(args.device, local_rank=dist_context.local_rank if dist_context.enabled else args.local_rank)
    train_config = train_config_lib.get_config(args.config_name)
    torch_config = train_utils._torch_config(train_config, args)  # pylint: disable=protected-access
    data_config = train_utils._data_config(train_config, repo_id)  # pylint: disable=protected-access
    if args.task_indexes:
        object.__setattr__(data_config, "task_indexes", args.task_indexes)
    skip_norm_stats = args.skip_norm_stats
    if args.norm_stats_path is not None:
        object.__setattr__(
            data_config,
            "norm_stats",
            train_utils._load_norm_stats(args.norm_stats_path),  # pylint: disable=protected-access
        )
        skip_norm_stats = False

    dataset = openpi_data_loader.create_torch_dataset(data_config, train_config.model)
    dataset = openpi_data_loader.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    dataset = openpi_data_loader.SafeDataset(dataset)
    sampler = None
    if dist_context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_context.world_size,
            rank=dist_context.rank,
            shuffle=args.shuffle,
            seed=args.seed,
            drop_last=True,
        )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle if sampler is None else False,
        num_workers=args.num_workers,
        collate_fn=openpi_data_loader._collate_fn,  # pylint: disable=protected-access
        drop_last=True,
        generator=generator if args.shuffle and sampler is None else None,
        sampler=sampler,
    )

    model = ACOTVLATorch(torch_config)
    model.to(device=device, dtype=train_utils._torch_dtype(args.torch_dtype))  # pylint: disable=protected-access
    loaded_checkpoint = None
    if args.init_checkpoint_path is not None:
        loaded_checkpoint = torch_checkpoint.load_converted_checkpoint(
            model,
            args.init_checkpoint_path,
            device=device,
            strict=args.strict,
        )
        model.to(device=device, dtype=train_utils._torch_dtype(args.torch_dtype))  # pylint: disable=protected-access

    train_utils._set_trainable_scope(model, args.trainable_scope)  # pylint: disable=protected-access
    if args.trainable_dtype == "float32":
        for parameter in model.parameters():
            if parameter.requires_grad and torch.is_floating_point(parameter):
                parameter.data = parameter.data.float()
    elif args.trainable_dtype != "model":
        raise ValueError(f"Unsupported trainable_dtype: {args.trainable_dtype!r}.")

    named_trainable_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_parameters = [parameter for _, parameter in named_trainable_parameters]
    if not trainable_parameters:
        raise RuntimeError(f"No trainable parameters selected by trainable_scope={args.trainable_scope!r}.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    return repo_id, device, loader, sampler, model, loaded_checkpoint, named_trainable_parameters, optimizer


def _wrap_for_training(
    model: ACOTVLATorch,
    args: Args,
    dist_context: torch_distributed.DistributedContext,
) -> torch.nn.Module:
    training_module = _TrainingLossModule(model)
    if not dist_context.enabled:
        return training_module

    if args.device in {"cuda", "npu"}:
        return DistributedDataParallel(
            training_module,
            device_ids=[dist_context.local_rank],
            output_device=dist_context.local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
    return DistributedDataParallel(
        training_module,
        find_unused_parameters=args.ddp_find_unused_parameters,
        broadcast_buffers=False,
    )


def _reduced_loss_payload(
    losses: dict[str, torch.Tensor],
    dist_context: torch_distributed.DistributedContext,
) -> dict[str, float]:
    return {
        key: train_utils._scalar(torch_distributed.mean_reduce(losses[key], dist_context))  # pylint: disable=protected-access
        for key in _LOSS_LOG_KEYS
    }


_LOSS_LOG_KEYS = (
    "total_loss",
    "coarse_loss",
    "action_loss",
    "timestep_mean",
    "predicted_action_velocity_mean",
    "predicted_action_velocity_std",
)


def _loss_payload_from_tensors(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: train_utils._scalar(losses[key])  # pylint: disable=protected-access
        for key in _LOSS_LOG_KEYS
    }


def _new_loss_accumulator(device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), device=device) for key in _LOSS_LOG_KEYS}


def _add_losses_to_accumulator(
    accumulator: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
) -> None:
    for key in _LOSS_LOG_KEYS:
        accumulator[key] = accumulator[key] + losses[key].detach()


def _average_loss_accumulator(
    accumulator: dict[str, torch.Tensor],
    count: int,
) -> dict[str, torch.Tensor]:
    divisor = max(count, 1)
    return {key: value / divisor for key, value in accumulator.items()}


def _losses_finite(losses: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(loss).all().item()) for loss in losses.values())


_FIXTURE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _load_fixed_loss_fixture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    batch: dict[str, Any] = {
        "image": {key: data[f"image__{key}"] for key in _FIXTURE_IMAGE_KEYS},
        "image_mask": {key: data[f"image_mask__{key}"] for key in _FIXTURE_IMAGE_KEYS},
        "state": data["state"],
        "actions": data["actions"],
        "coarse_actions": data["coarse_actions"],
    }
    if "tokenized_prompt" in data:
        batch["tokenized_prompt"] = data["tokenized_prompt"]
    if "tokenized_prompt_mask" in data:
        batch["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
    randoms = {
        "timestep": data["timestep"],
        "action_noise": data["action_noise"],
        "coarse_action_noise": data["coarse_action_noise"],
    }
    return batch, randoms


def _fixed_loss_fixture_to_tensors(
    fixture: tuple[dict[str, Any], dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> tuple[TorchObservation, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    batch, randoms_np = fixture
    observation = TorchObservation.from_dict(batch, device=device)
    actions = train_utils._as_tensor(batch["actions"], device=device).float()  # pylint: disable=protected-access
    coarse_actions = train_utils._as_tensor(batch["coarse_actions"], device=device).float()  # pylint: disable=protected-access
    randoms = {
        "timestep": torch.from_numpy(randoms_np["timestep"]).to(device=device),
        "action_noise": torch.from_numpy(randoms_np["action_noise"]).to(device=device),
        "coarse_action_noise": torch.from_numpy(randoms_np["coarse_action_noise"]).to(device=device),
    }
    return observation, actions, coarse_actions, randoms


@torch.no_grad()
def _evaluate_fixed_loss_fixture(
    model: torch.nn.Module,
    fixture: tuple[dict[str, Any], dict[str, np.ndarray]] | None,
    *,
    device: torch.device,
    dist_context: torch_distributed.DistributedContext,
) -> dict[str, float] | None:
    if fixture is None:
        return None

    was_training = model.training
    model.eval()
    observation, actions, coarse_actions, randoms = _fixed_loss_fixture_to_tensors(fixture, device=device)
    losses = model.compute_training_loss(
        observation,
        actions=actions,
        coarse_actions=coarse_actions,
        **randoms,
    )
    if was_training:
        model.train()
    return {
        key: train_utils._scalar(torch_distributed.mean_reduce(losses[key], dist_context))  # pylint: disable=protected-access
        for key in ("total_loss", "coarse_loss", "action_loss", "timestep_mean")
    }


def _run_fixed_train_probe(
    training_module: torch.nn.Module,
    fixture: tuple[dict[str, Any], dict[str, np.ndarray]] | None,
    *,
    trainable_parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dist_context: torch_distributed.DistributedContext,
) -> dict[str, float] | None:
    if fixture is None:
        return None

    was_training = training_module.training
    training_module.train()
    optimizer.zero_grad(set_to_none=True)

    observation, actions, coarse_actions, randoms = _fixed_loss_fixture_to_tensors(fixture, device=device)
    losses = training_module(
        observation,
        actions,
        coarse_actions,
        **randoms,
    )
    losses["total_loss"].backward()
    grad_norm = train_utils._grad_norm(trainable_parameters).to(device=device)  # pylint: disable=protected-access
    payload = {
        key: train_utils._scalar(torch_distributed.mean_reduce(losses[key], dist_context))  # pylint: disable=protected-access
        for key in ("total_loss", "coarse_loss", "action_loss", "timestep_mean")
    }
    payload["grad_norm"] = _reduced_scalar(grad_norm, dist_context)
    optimizer.zero_grad(set_to_none=True)
    if not was_training:
        training_module.eval()
    return payload


def _reduced_scalar(value: torch.Tensor, dist_context: torch_distributed.DistributedContext) -> float:
    return train_utils._scalar(torch_distributed.mean_reduce(value, dist_context))  # pylint: disable=protected-access


def _set_sampler_epoch(sampler: DistributedSampler | None, epoch: int) -> None:
    if sampler is not None:
        sampler.set_epoch(epoch)


def _parameter_count(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _parameter_dtype_counts(parameters: list[torch.nn.Parameter]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in parameters:
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
    return counts


def _trainable_param_norm(parameters: list[torch.nn.Parameter], *, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for parameter in parameters:
        if torch.is_floating_point(parameter):
            value = parameter.detach().float()
            total = total + torch.sum(value * value)
    return torch.sqrt(total)


def _group_name(parameter_name: str) -> str:
    return parameter_name.split(".", maxsplit=1)[0]


def _empty_update_group(device: torch.device) -> dict[str, Any]:
    return {
        "parameter_count": 0,
        "parameter_numel": 0,
        "grad_sq": torch.zeros((), device=device),
        "expected_delta_sq": torch.zeros((), device=device),
        "effective_delta_sq": torch.zeros((), device=device),
        "effective_delta_abs_max": torch.zeros((), device=device),
        "effective_delta_zero_numel": 0,
    }


def _finalize_update_group(group: dict[str, Any]) -> dict[str, float | int]:
    effective_delta_numel = max(int(group["parameter_numel"]), 1)
    return {
        "parameter_count": int(group["parameter_count"]),
        "parameter_numel": int(group["parameter_numel"]),
        "grad_norm_after_clip": train_utils._scalar(torch.sqrt(group["grad_sq"])),  # pylint: disable=protected-access
        "expected_delta_norm": train_utils._scalar(torch.sqrt(group["expected_delta_sq"])),  # pylint: disable=protected-access
        "effective_delta_norm": train_utils._scalar(torch.sqrt(group["effective_delta_sq"])),  # pylint: disable=protected-access
        "effective_delta_abs_max": train_utils._scalar(group["effective_delta_abs_max"]),  # pylint: disable=protected-access
        "effective_delta_zero_fraction": float(group["effective_delta_zero_numel"]) / effective_delta_numel,
    }


def _adamw_update_debug_payload(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_norm_before_clip: torch.Tensor,
    grad_norm_after_clip: torch.Tensor,
    param_norm_before: torch.Tensor,
) -> dict[str, Any]:
    name_by_parameter_id = {id(parameter): name for name, parameter in named_parameters}
    groups: dict[str, dict[str, Any]] = {}
    global_group = _empty_update_group(device)
    param_group_payloads = []

    for group_index, param_group in enumerate(optimizer.param_groups):
        lr = float(param_group["lr"])
        beta1, beta2 = param_group["betas"]
        eps = float(param_group["eps"])
        weight_decay = float(param_group["weight_decay"])
        group_parameter_numel = 0
        group_expected_delta_sq = torch.zeros((), device=device)
        group_effective_delta_sq = torch.zeros((), device=device)

        for parameter in param_group["params"]:
            if parameter.grad is None or not torch.is_floating_point(parameter):
                continue

            parameter_name = name_by_parameter_id.get(id(parameter), "<unnamed>")
            module_group_name = _group_name(parameter_name)
            module_group = groups.setdefault(module_group_name, _empty_update_group(device))

            param_float = parameter.detach().float()
            grad_float = parameter.grad.detach().float()
            optimizer_state = optimizer.state.get(parameter, {})
            step = optimizer_state.get("step", 0)
            if isinstance(step, torch.Tensor):
                step_value = int(step.detach().cpu().item()) + 1
            else:
                step_value = int(step) + 1

            exp_avg = optimizer_state.get("exp_avg")
            if exp_avg is None:
                exp_avg_new = grad_float * (1.0 - beta1)
            else:
                exp_avg_new = exp_avg.detach().float().mul(beta1).add(grad_float, alpha=1.0 - beta1)

            exp_avg_sq = optimizer_state.get("exp_avg_sq")
            if exp_avg_sq is None:
                exp_avg_sq_new = grad_float.square() * (1.0 - beta2)
            else:
                exp_avg_sq_new = exp_avg_sq.detach().float().mul(beta2).addcmul(
                    grad_float,
                    grad_float,
                    value=1.0 - beta2,
                )

            bias_correction1 = 1.0 - beta1**step_value
            bias_correction2 = 1.0 - beta2**step_value
            denom = exp_avg_sq_new.sqrt().div(bias_correction2**0.5).add(eps)
            adam_delta = exp_avg_new.div(denom).mul(-lr / bias_correction1)
            if weight_decay:
                expected_delta = adam_delta.add(param_float, alpha=-lr * weight_decay)
            else:
                expected_delta = adam_delta

            effective_delta = (param_float + expected_delta).to(dtype=parameter.dtype).float() - param_float
            grad_sq = torch.sum(grad_float.square())
            expected_delta_sq = torch.sum(expected_delta.square())
            effective_delta_sq = torch.sum(effective_delta.square())
            effective_delta_abs_max = effective_delta.abs().amax()
            effective_delta_zero_numel = int((effective_delta == 0).sum().detach().cpu().item())
            parameter_numel = parameter.numel()

            for target_group in (global_group, module_group):
                target_group["parameter_count"] += 1
                target_group["parameter_numel"] += parameter_numel
                target_group["grad_sq"] = target_group["grad_sq"] + grad_sq
                target_group["expected_delta_sq"] = target_group["expected_delta_sq"] + expected_delta_sq
                target_group["effective_delta_sq"] = target_group["effective_delta_sq"] + effective_delta_sq
                target_group["effective_delta_abs_max"] = torch.maximum(
                    target_group["effective_delta_abs_max"],
                    effective_delta_abs_max,
                )
                target_group["effective_delta_zero_numel"] += effective_delta_zero_numel

            group_parameter_numel += parameter_numel
            group_expected_delta_sq = group_expected_delta_sq + expected_delta_sq
            group_effective_delta_sq = group_effective_delta_sq + effective_delta_sq

        param_group_payloads.append(
            {
                "group_index": group_index,
                "lr": lr,
                "beta1": float(beta1),
                "beta2": float(beta2),
                "eps": eps,
                "weight_decay": weight_decay,
                "parameter_numel": group_parameter_numel,
                "expected_delta_norm": train_utils._scalar(torch.sqrt(group_expected_delta_sq)),  # pylint: disable=protected-access
                "effective_delta_norm": train_utils._scalar(torch.sqrt(group_effective_delta_sq)),  # pylint: disable=protected-access
            }
        )

    sorted_groups = {
        name: _finalize_update_group(group)
        for name, group in sorted(groups.items(), key=lambda item: item[0])
    }
    payload = _finalize_update_group(global_group)
    payload.update(
        {
            "grad_norm_before_clip": train_utils._scalar(grad_norm_before_clip),  # pylint: disable=protected-access
            "grad_norm_after_clip": train_utils._scalar(grad_norm_after_clip),  # pylint: disable=protected-access
            "param_norm_before": train_utils._scalar(param_norm_before),  # pylint: disable=protected-access
            "param_groups": param_group_payloads,
            "groups": sorted_groups,
        }
    )
    return payload


def _make_fixed_eval_batches(
    dataset: torch.utils.data.Dataset,
    *,
    args: Args,
    dist_context: torch_distributed.DistributedContext,
) -> list[Any]:
    if args.eval_interval <= 0 or args.eval_batches <= 0:
        return []

    print(f"Preparing fixed eval batches: eval_batches={args.eval_batches} batch_size={args.batch_size}", flush=True)
    batches = []
    # Build fixed eval batches by direct indexing instead of a DataLoader. This avoids
    # forking worker processes after JAX has initialized, which can deadlock in some
    # NPU environments.
    stride = dist_context.world_size if dist_context.enabled else 1
    rank = dist_context.rank if dist_context.enabled else 0
    for batch_idx in range(args.eval_batches):
        global_start = batch_idx * args.batch_size * stride
        indices = [global_start + rank + item_idx * stride for item_idx in range(args.batch_size)]
        if not indices or max(indices) >= len(dataset):
            break
        print(f"Preparing fixed eval batch {batch_idx}: indices={indices}", flush=True)
        samples = [dataset[index] for index in indices]
        batches.append(openpi_data_loader._collate_fn(samples))  # pylint: disable=protected-access
    print(f"Prepared fixed eval batches: count={len(batches)}", flush=True)
    return batches


def _deterministic_training_randoms_for_context(
    *,
    actions: torch.Tensor,
    coarse_actions: torch.Tensor,
    seed: int,
    step: int,
    device: torch.device,
    dist_context: torch_distributed.DistributedContext,
) -> dict[str, torch.Tensor]:
    if not dist_context.enabled:
        return train_utils._deterministic_training_randoms(  # pylint: disable=protected-access
            actions=actions,
            coarse_actions=coarse_actions,
            seed=seed,
            step=step,
            device=device,
        )

    local_batch_size = actions.shape[0]
    global_batch_size = local_batch_size * dist_context.world_size
    global_actions = torch.empty(
        (global_batch_size, *actions.shape[1:]),
        dtype=actions.dtype,
        device=device,
    )
    global_coarse_actions = torch.empty(
        (global_batch_size, *coarse_actions.shape[1:]),
        dtype=coarse_actions.dtype,
        device=device,
    )
    global_randoms = train_utils._deterministic_training_randoms(  # pylint: disable=protected-access
        actions=global_actions,
        coarse_actions=global_coarse_actions,
        seed=seed,
        step=step,
        device=device,
    )
    # DistributedSampler with shuffle=False yields rank-strided samples:
    # rank0 -> 0, world_size, ...; rank1 -> 1, world_size + 1, ...
    # Select the matching entries from the single-process global batch random stream.
    global_indices = torch.arange(local_batch_size, device=device, dtype=torch.long) * dist_context.world_size
    global_indices = global_indices + dist_context.rank
    return {key: value.index_select(0, global_indices) for key, value in global_randoms.items()}


@torch.no_grad()
def _evaluate_fixed_batches(
    model: torch.nn.Module,
    batches: list[Any],
    *,
    device: torch.device,
    seed: int,
    dist_context: torch_distributed.DistributedContext,
) -> dict[str, float] | None:
    if not batches:
        return None

    was_training = model.training
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    for batch_idx, batch in enumerate(batches):
        observation = TorchObservation.from_dict(batch, device=device)
        actions = train_utils._as_tensor(batch["actions"], device=device).float()  # pylint: disable=protected-access
        coarse_actions = train_utils._as_tensor(batch["coarse_actions"], device=device).float()  # pylint: disable=protected-access
        randoms = _deterministic_training_randoms_for_context(
            actions=actions,
            coarse_actions=coarse_actions,
            seed=seed,
            step=batch_idx,
            device=device,
            dist_context=dist_context,
        )
        losses = model.compute_training_loss(
            observation,
            actions=actions,
            coarse_actions=coarse_actions,
            **randoms,
        )
        for key in ("total_loss", "coarse_loss", "action_loss"):
            totals[key] = totals.get(key, torch.zeros((), device=device)) + losses[key].detach()

    if was_training:
        model.train()

    count = torch.tensor(float(len(batches)), device=device)
    return {
        key: train_utils._scalar(torch_distributed.mean_reduce(value / count, dist_context))  # pylint: disable=protected-access
        for key, value in totals.items()
    }


def main(args: Args) -> None:
    dist_context = torch_distributed.init_distributed(
        mode=args.distributed,
        device=args.device,
        backend=args.dist_backend,
        local_rank=args.local_rank,
    )
    try:
        _configure_reproducibility(args)
        if args.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
        if args.train_steps <= 0:
            raise ValueError(f"train_steps must be positive, got {args.train_steps}.")
        if args.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {args.log_interval}.")
        if args.eval_interval < 0:
            raise ValueError(f"eval_interval must be non-negative, got {args.eval_interval}.")
        if args.eval_batches < 0:
            raise ValueError(f"eval_batches must be non-negative, got {args.eval_batches}.")
        if args.fixed_loss_interval < 0:
            raise ValueError(f"fixed_loss_interval must be non-negative, got {args.fixed_loss_interval}.")
        if args.update_debug_interval < 0:
            raise ValueError(f"update_debug_interval must be non-negative, got {args.update_debug_interval}.")
        if args.fixed_loss_fixture is not None:
            if args.fixed_loss_interval <= 0:
                raise ValueError("--fixed-loss-interval must be positive when --fixed-loss-fixture is set.")
            if not args.fixed_loss_fixture.exists():
                raise FileNotFoundError(args.fixed_loss_fixture)
        if args.save_interval < 0:
            raise ValueError(f"save_interval must be non-negative, got {args.save_interval}.")

        out_dir = _checkpoint_dir(args)
        if dist_context.is_main_process:
            if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
                raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to reuse it.")
            out_dir.mkdir(parents=True, exist_ok=True)

            args_payload = dataclasses.asdict(args)
            args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in args_payload.items()}
            args_payload["distributed_context"] = dataclasses.asdict(dist_context)
            _write_json(out_dir / "args.json", args_payload)
        torch_distributed.barrier(dist_context)

        repo_id, device, loader, sampler, model, loaded_checkpoint, named_trainable_parameters, optimizer = (
            _load_training_objects(args, dist_context)
        )
        trainable_parameters = [parameter for _, parameter in named_trainable_parameters]
        dataset_size = len(loader.dataset)
        trainable_parameter_count = _parameter_count(trainable_parameters)
        trainable_dtype_counts = _parameter_dtype_counts(trainable_parameters)
        if dist_context.is_main_process:
            total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"trainable_parameters={trainable_parameter_count} "
                f"total_parameters={total_parameter_count} trainable_scope={args.trainable_scope} "
                f"trainable_dtype={args.trainable_dtype} trainable_dtype_counts={trainable_dtype_counts}"
            )
            print(
                f"seed={args.seed} deterministic={args.deterministic} "
                f"deterministic_algorithms_enabled={torch.are_deterministic_algorithms_enabled()}"
            )
            if args.task_indexes:
                print(f"task_indexes={args.task_indexes} filtered_dataset_rows={dataset_size}")
        eval_batches = _make_fixed_eval_batches(loader.dataset, args=args, dist_context=dist_context)
        fixed_loss_fixture = (
            _load_fixed_loss_fixture(args.fixed_loss_fixture) if args.fixed_loss_fixture is not None else None
        )
        if fixed_loss_fixture is not None and dist_context.is_main_process:
            print(
                f"Loaded fixed loss fixture: path={args.fixed_loss_fixture} "
                f"interval={args.fixed_loss_interval}",
                flush=True,
            )
        model.train()
        training_module = _wrap_for_training(model, args, dist_context)
        training_module.train()
        initial_fixed_loss_payload = None
        if fixed_loss_fixture is not None:
            initial_fixed_loss_payload = _evaluate_fixed_loss_fixture(
                model,
                fixed_loss_fixture,
                device=device,
                dist_context=dist_context,
            )
        initial_fixed_train_probe_payload = None
        if fixed_loss_fixture is not None:
            initial_fixed_train_probe_payload = _run_fixed_train_probe(
                training_module,
                fixed_loss_fixture,
                trainable_parameters=trainable_parameters,
                optimizer=optimizer,
                device=device,
                dist_context=dist_context,
            )
        sampler_epoch = 0
        _set_sampler_epoch(sampler, sampler_epoch)
        iterator = iter(loader)
        history: list[dict[str, Any]] = []
        last_losses: dict[str, torch.Tensor] | None = None
        last_global_loss_payload: dict[str, float] | None = None
        last_grad_norm = torch.zeros((), device=device)
        last_global_grad_norm = 0.0
        log_loss_totals = _new_loss_accumulator(device)
        log_grad_norm_total = torch.zeros((), device=device)
        log_count = 0
        start_time = time.monotonic()

        for step in range(args.train_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                _set_sampler_epoch(sampler, sampler_epoch)
                iterator = iter(loader)
                batch = next(iterator)

            observation = TorchObservation.from_dict(batch, device=device)
            actions = train_utils._as_tensor(batch["actions"], device=device).float()  # pylint: disable=protected-access
            coarse_actions = train_utils._as_tensor(batch["coarse_actions"], device=device).float()  # pylint: disable=protected-access
            randoms = _deterministic_training_randoms_for_context(
                actions=actions,
                coarse_actions=coarse_actions,
                seed=args.seed,
                step=step,
                device=device,
                dist_context=dist_context,
            )

            optimizer.zero_grad(set_to_none=True)
            losses = training_module(
                observation,
                actions,
                coarse_actions,
                **randoms,
            )
            losses["total_loss"].backward()
            grad_norm = train_utils._grad_norm(trainable_parameters).to(device=device)  # pylint: disable=protected-access
            update_debug_payload = None
            update_debug_enabled = args.update_debug_interval > 0 and (
                step % args.update_debug_interval == 0 or step == args.train_steps - 1
            )
            param_norm_before_update = None
            if update_debug_enabled:
                param_norm_before_update = _trainable_param_norm(trainable_parameters, device=device)
            if args.clip_gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.clip_gradient_norm)
            grad_norm_after_clip = train_utils._grad_norm(trainable_parameters).to(device=device)  # pylint: disable=protected-access
            if update_debug_enabled:
                update_debug_payload = _adamw_update_debug_payload(
                    named_trainable_parameters,
                    optimizer,
                    device=device,
                    grad_norm_before_clip=grad_norm,
                    grad_norm_after_clip=grad_norm_after_clip,
                    param_norm_before=param_norm_before_update,
                )
            optimizer.step()
            if update_debug_payload is not None:
                param_norm_after_update = _trainable_param_norm(trainable_parameters, device=device)
                update_debug_payload["param_norm_after"] = train_utils._scalar(param_norm_after_update)  # pylint: disable=protected-access
                update_debug_payload["param_norm_delta"] = train_utils._scalar(  # pylint: disable=protected-access
                    param_norm_after_update - param_norm_before_update
                )
                if dist_context.is_main_process:
                    print(
                        "update_debug_step={step} ".format(step=step)
                        + json.dumps(update_debug_payload, sort_keys=True, separators=(",", ":")),
                        flush=True,
                    )

            last_losses = losses
            last_grad_norm = grad_norm
            _add_losses_to_accumulator(log_loss_totals, losses)
            log_grad_norm_total = log_grad_norm_total + grad_norm.detach()
            log_count += 1
            if step % args.log_interval == 0 or step == args.train_steps - 1:
                averaged_losses = _average_loss_accumulator(log_loss_totals, log_count)
                local_loss_payload = _loss_payload_from_tensors(averaged_losses)
                global_loss_payload = _reduced_loss_payload(averaged_losses, dist_context)
                eval_loss_payload = None
                if args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.train_steps - 1):
                    eval_loss_payload = _evaluate_fixed_batches(
                        model,
                        eval_batches,
                        device=device,
                        seed=args.seed,
                        dist_context=dist_context,
                    )
                fixed_loss_payload = None
                if fixed_loss_fixture is not None and (
                    step % args.fixed_loss_interval == 0 or step == args.train_steps - 1
                ):
                    fixed_loss_payload = _evaluate_fixed_loss_fixture(
                        model,
                        fixed_loss_fixture,
                        device=device,
                        dist_context=dist_context,
                    )
                averaged_grad_norm = log_grad_norm_total / max(log_count, 1)
                local_grad_norm = train_utils._scalar(averaged_grad_norm)  # pylint: disable=protected-access
                global_grad_norm = _reduced_scalar(averaged_grad_norm, dist_context)
                last_global_loss_payload = global_loss_payload
                last_global_grad_norm = global_grad_norm
                row = {
                    "step": step,
                    "loss": global_loss_payload,
                    "eval_loss": eval_loss_payload,
                    "fixed_loss": fixed_loss_payload,
                    "rank0_loss": local_loss_payload if dist_context.is_main_process else None,
                    "loss_finite": _losses_finite(averaged_losses),
                    "grad_norm": global_grad_norm,
                    "rank0_grad_norm": local_grad_norm if dist_context.is_main_process else None,
                    "grad_norm_finite": bool(torch.isfinite(averaged_grad_norm).all().item()),
                    "log_count": log_count,
                    "update_debug": update_debug_payload,
                }
                if dist_context.is_main_process:
                    history.append(row)
                    print(
                        "step={step} total_loss={total_loss:.6f} coarse_loss={coarse_loss:.6f} "
                        "action_loss={action_loss:.6f} grad_norm={grad_norm:.6f} "
                        "rank0_total_loss={rank0_total_loss:.6f}".format(
                            step=step,
                            grad_norm=global_grad_norm,
                            rank0_total_loss=local_loss_payload["total_loss"],
                            **global_loss_payload,
                        )
                        )
                    if eval_loss_payload is not None:
                        print(
                            "eval_step={step} eval_total_loss={total_loss:.6f} "
                            "eval_coarse_loss={coarse_loss:.6f} eval_action_loss={action_loss:.6f}".format(
                                step=step,
                                **eval_loss_payload,
                            )
                        )
                    if fixed_loss_payload is not None:
                        print(
                            "fixed_eval_step={step} fixed_total_loss={total_loss:.6f} "
                            "fixed_coarse_loss={coarse_loss:.6f} fixed_action_loss={action_loss:.6f} "
                            "fixed_timestep_mean={timestep_mean:.6f}".format(
                                step=step,
                                **fixed_loss_payload,
                            )
                        )
                log_loss_totals = _new_loss_accumulator(device)
                log_grad_norm_total = torch.zeros((), device=device)
                log_count = 0

            if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                if dist_context.is_main_process:
                    _save_checkpoint(model, out_dir / f"step_{step + 1}.safetensors")
                torch_distributed.barrier(dist_context)

        final_checkpoint = out_dir / "final.safetensors" if args.save_final_checkpoint else None
        if final_checkpoint is not None and dist_context.is_main_process:
            _save_checkpoint(model, final_checkpoint)
        torch_distributed.barrier(dist_context)

        elapsed_s = time.monotonic() - start_time
        if dist_context.is_main_process:
            final_loss = last_global_loss_payload
            if final_loss is None and last_losses is not None:
                final_loss = _reduced_loss_payload(last_losses, dist_context)
            summary = {
                "device": str(device),
                "config_name": args.config_name,
                "exp_name": args.exp_name,
                "data_root": str(args.data_root),
                "repo_id": str(repo_id),
                "rgb_only": args.rgb_only,
                "torch_backbone": args.torch_backbone,
                "torch_dtype": args.torch_dtype,
                "trainable_dtype": args.trainable_dtype,
                "model_status": model.model_status,
                "loaded_checkpoint": None if loaded_checkpoint is None else str(loaded_checkpoint),
                "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
                "seed": args.seed,
                "deterministic": args.deterministic,
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "batch_size": args.batch_size,
                "global_batch_size": args.batch_size * dist_context.world_size,
                "train_steps": args.train_steps,
                "trainable_scope": args.trainable_scope,
                "task_indexes": list(args.task_indexes),
                "filtered_dataset_rows": dataset_size,
                "trainable_parameters": trainable_parameter_count,
                "trainable_dtype_counts": trainable_dtype_counts,
                "learning_rate": args.learning_rate,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "clip_gradient_norm": args.clip_gradient_norm,
                "fixed_loss_fixture": None if args.fixed_loss_fixture is None else str(args.fixed_loss_fixture),
                "fixed_loss_interval": args.fixed_loss_interval,
                "update_debug_interval": args.update_debug_interval,
                "initial_fixed_loss": initial_fixed_loss_payload,
                "initial_fixed_train_probe": initial_fixed_train_probe_payload,
                "distributed": dataclasses.asdict(dist_context),
                "elapsed_s": elapsed_s,
                "steps_per_second": args.train_steps / elapsed_s if elapsed_s > 0 else None,
                "samples_per_second": (args.train_steps * args.batch_size * dist_context.world_size) / elapsed_s
                if elapsed_s > 0
                else None,
                "final_loss": final_loss,
                "final_loss_finite": None
                if last_losses is None
                else bool(torch.isfinite(last_losses["total_loss"]).all().item()),
                "final_grad_norm": last_global_grad_norm,
                "final_grad_norm_finite": bool(torch.isfinite(last_grad_norm).all().item()),
                "history": history,
            }
            summary_path = args.summary_path if args.summary_path is not None else out_dir / "summary.json"
            _write_json(summary_path, summary)
            print(
                json.dumps(
                    {
                        "summary_path": str(summary_path),
                        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
                    },
                    indent=2,
                )
            )
    finally:
        torch_distributed.destroy_process_group(dist_context)


if __name__ == "__main__":
    main(tyro.cli(Args))
