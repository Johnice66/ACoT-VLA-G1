from __future__ import annotations

import dataclasses
import os
from typing import Literal

import torch


DistributedMode = Literal["auto", "true", "false"]
DistributedBackend = Literal["auto", "hccl", "nccl", "gloo"]


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str | None
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_world_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def should_init_distributed(mode: DistributedMode) -> bool:
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode != "auto":
        raise ValueError(f"Unsupported distributed mode: {mode!r}.")
    return _env_world_size() > 1


def default_backend(device: str) -> str:
    if device == "npu":
        return "hccl"
    if device == "cuda":
        return "nccl"
    if device == "cpu":
        return "gloo"
    raise ValueError(f"Unsupported distributed device: {device!r}.")


def _set_process_device(device: str, local_rank: int) -> None:
    if device == "npu":
        import torch_npu  # noqa: F401

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu is not available after importing torch_npu.")
        torch.npu.set_device(local_rank)
        return

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda is not available.")
        torch.cuda.set_device(local_rank)


def init_distributed(
    *,
    mode: DistributedMode,
    device: str,
    backend: DistributedBackend = "auto",
    local_rank: int | None = None,
) -> DistributedContext:
    enabled = should_init_distributed(mode)
    rank = _env_int("RANK", 0)
    env_local_rank = _env_int("LOCAL_RANK", 0)
    resolved_local_rank = env_local_rank if local_rank is None else local_rank
    world_size = _env_world_size()
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
    resolved_backend = default_backend(device) if backend == "auto" else backend

    if not enabled:
        return DistributedContext(
            enabled=False,
            backend=None,
            rank=rank,
            local_rank=resolved_local_rank,
            world_size=1,
            local_world_size=1,
        )

    if world_size <= 1:
        raise RuntimeError("Distributed mode was requested, but WORLD_SIZE is not greater than 1.")
    _set_process_device(device, resolved_local_rank)

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=resolved_backend, init_method="env://")

    return DistributedContext(
        enabled=True,
        backend=resolved_backend,
        rank=rank,
        local_rank=resolved_local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
    )


def mean_reduce(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    reduced = value.detach().float().clone()
    if context.enabled:
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        reduced /= context.world_size
    return reduced


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        if context.backend in {"hccl", "nccl"}:
            torch.distributed.barrier(device_ids=[context.local_rank])
        else:
            torch.distributed.barrier()


def destroy_process_group(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
