from __future__ import annotations

import torch


def resolve_torch_device(device: str, *, local_rank: int | None = None) -> torch.device:
    if device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested, but torch_npu is not installed or not visible in this environment.") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu is not available after importing torch_npu.")
        index = 0 if local_rank is None else local_rank
        torch.npu.set_device(index)
        return torch.device(f"npu:{index}")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda is not available.")
        index = 0 if local_rank is None else local_rank
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported torch device: {device}")
