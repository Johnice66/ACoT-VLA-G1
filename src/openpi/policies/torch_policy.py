from __future__ import annotations

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models_pt.device import resolve_torch_device
from openpi.models_pt.types import TorchObservation


def _compose(transforms: Sequence[_transforms.DataTransformFn]) -> _transforms.DataTransformFn:
    return _transforms.compose(transforms)


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(child) for child in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(child) for key, child in value.items()}
    return value


def _unbatch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unbatch(child) for key, child in value.items()}
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


class TorchPolicy(_base_policy.BasePolicy):
    """Policy wrapper that mirrors the JAX Policy API while executing the model with PyTorch."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._device = resolve_torch_device(device) if isinstance(device, str) else device
        self._model = model.to(self._device)
        self._model.eval()
        self._input_transform = _compose(transforms)
        self._output_transform = _compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})
        self._metadata.update(
            {
                "backend": "torch",
                "device": str(self._device),
                "policy_kind": getattr(model, "model_status", model.__class__.__name__),
            }
        )
        self._generator = None
        if self._device.type != "npu":
            self._generator = torch.Generator(device=self._device)
            self._generator.manual_seed(0)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        inputs = _copy_tree(obs)
        inputs = self._input_transform(inputs)
        observation = TorchObservation.from_dict(inputs, device=self._device)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = self._model.sample_actions(
                observation,
                generator=self._generator,
                **self._sample_kwargs,
            )

        outputs = _to_numpy(outputs)
        outputs["state"] = observation.state.detach().cpu().numpy()
        outputs = _unbatch(outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return self.post_process(obs, outputs)

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = obs.get("task_name")
        if task_name is None:
            return outputs

        logging.info(
            "Torch policy infering for task: %s, with inference time: %.3f ms",
            task_name,
            outputs["policy_timing"]["infer_ms"],
        )
        if task_name not in task_name_requiring_waist:
            outputs["actions"] = outputs["actions"][:, :16]
            return outputs

        raw_state = obs.get("state")
        if raw_state is None:
            raise ValueError("State is required for post-processing waist actions.")
        outputs["actions"][:, 16:20] = np.asarray(raw_state)[16:20]
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class TorchSmokePolicy(_base_policy.BasePolicy):
    """Minimal Torch policy used to validate serving and device plumbing before the real model is ported."""

    def __init__(
        self,
        *,
        device: str,
        action_horizon: int,
        action_dim: int,
        coarse_action_horizon: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._device = resolve_torch_device(device)
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._coarse_action_horizon = coarse_action_horizon
        self._metadata = metadata or {}
        self._metadata.update(
            {
                "backend": "torch",
                "policy_kind": "smoke",
                "device": str(self._device),
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "coarse_action_horizon": coarse_action_horizon,
            }
        )

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        start_time = time.monotonic()

        state = obs.get("state")
        if state is None:
            state_tensor = torch.zeros(self._action_dim, dtype=torch.float32, device=self._device)
        else:
            state_tensor = torch.as_tensor(np.asarray(state).copy(), dtype=torch.float32, device=self._device).flatten()

        if state_tensor.numel() < self._action_dim:
            padded = torch.zeros(self._action_dim, dtype=torch.float32, device=self._device)
            padded[: state_tensor.numel()] = state_tensor
            state_tensor = padded
        else:
            state_tensor = state_tensor[: self._action_dim]

        actions = state_tensor.unsqueeze(0).repeat(self._action_horizon, 1)
        outputs = {
            "actions": actions.cpu().numpy().astype(np.float32),
            "policy_timing": {
                "infer_ms": (time.monotonic() - start_time) * 1000,
            },
        }

        if self._coarse_action_horizon is not None:
            coarse_actions = state_tensor.unsqueeze(0).repeat(self._coarse_action_horizon, 1)
            outputs["coarse_actions"] = coarse_actions.cpu().numpy().astype(np.float32)

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class TorchPolicyRecorder(_base_policy.BasePolicy):
    """JAX-free recorder for Torch policies."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0
        logging.info("Dumping Torch policy records to: %s", self._record_dir)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        outputs = self._policy.infer(obs)
        output_path = self._record_dir / f"step_{self._record_step}.npy"
        self._record_step += 1
        np.save(output_path, {"inputs": obs, "outputs": outputs}, allow_pickle=True)
        return outputs


def create_acot_icra_smoke_policy(*, device: str) -> TorchSmokePolicy:
    return TorchSmokePolicy(
        device=device,
        action_horizon=30,
        action_dim=32,
        coarse_action_horizon=30,
        metadata={
            "config": "acot_icra_simulation_challenge_reasoning_to_action",
            "note": "Torch/NPU smoke policy only; real ACoT-VLA weights are not loaded.",
        },
    )
