from __future__ import annotations

import dataclasses
import math
import re

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    """Torch copy of the LoRA settings used by the JAX Gemma modules."""

    rank: int
    alpha: float = 1.0
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum weight with optional LoRA adapters."""

    def __init__(self, shape: tuple[int, ...], *, lora_config: LoRAConfig | None = None):
        super().__init__()
        self.shape = shape
        self.lora_config = lora_config
        self.w = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.w.reshape(shape[0], -1) if len(shape) > 2 else self.w)

        if lora_config is not None:
            shape_a, shape_b = list(shape), list(shape)
            shape_a[lora_config.axes[1]] = lora_config.rank
            shape_b[lora_config.axes[0]] = lora_config.rank
            self.lora_a = nn.Parameter(torch.empty(tuple(shape_a)))
            self.lora_b = nn.Parameter(torch.empty(tuple(shape_b)))
            nn.init.normal_(self.lora_a, std=0.01)
            nn.init.normal_(self.lora_b, std=0.01)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, eqn: str, x: torch.Tensor) -> torch.Tensor:
        weight = self.w.to(dtype=x.dtype)
        result = torch.einsum(eqn, x, weight)

        if self.lora_config is not None:
            if self.lora_a is None or self.lora_b is None:
                raise RuntimeError("LoRA parameters are not initialized.")
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = torch.einsum(eqn_a, x, self.lora_a.to(dtype=x.dtype))
            lora = torch.einsum(eqn_b, lora, self.lora_b.to(dtype=x.dtype))
            result = result + lora * self.lora_config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if self.lora_config is None:
            raise RuntimeError("LoRA equation requested without LoRA config.")
        if self.lora_config.label in eqn:
            raise ValueError(f"{self.lora_config.label} already in eqn: {eqn}")
        match = re.match(r"(.*),(.*)->(.*)", eqn)
        if match is None:
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = match.groups()

        a_label, b_label = (rhs[index] for index in self.lora_config.axes)
        label = self.lora_config.label
        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"
        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Gemma gated feed-forward block with optional LoRA adapters."""

    def __init__(self, *, features: int, hidden_dim: int, lora_config: LoRAConfig | None = None):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config
        self.gating_einsum = nn.Parameter(torch.empty(2, features, hidden_dim))
        self.linear = nn.Parameter(torch.empty(hidden_dim, features))
        nn.init.xavier_uniform_(self.gating_einsum.reshape(2 * features, hidden_dim))
        nn.init.xavier_uniform_(self.linear)

        if lora_config is not None:
            self.gating_einsum_lora_a = nn.Parameter(torch.empty(2, features, lora_config.rank))
            self.gating_einsum_lora_b = nn.Parameter(torch.empty(2, lora_config.rank, hidden_dim))
            self.linear_lora_a = nn.Parameter(torch.empty(hidden_dim, lora_config.rank))
            self.linear_lora_b = nn.Parameter(torch.empty(lora_config.rank, features))
            for param in (
                self.gating_einsum_lora_a,
                self.gating_einsum_lora_b,
                self.linear_lora_a,
                self.linear_lora_b,
            ):
                nn.init.normal_(param, std=0.01)
        else:
            self.gating_einsum_lora_a = None
            self.gating_einsum_lora_b = None
            self.linear_lora_a = None
            self.linear_lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._dot(
            x,
            self.gating_einsum[0],
            None
            if self.gating_einsum_lora_a is None or self.gating_einsum_lora_b is None
            else (self.gating_einsum_lora_a[0], self.gating_einsum_lora_b[0]),
        )
        gate_value = F.gelu(gate, approximate="tanh")
        value = self._dot(
            x,
            self.gating_einsum[1],
            None
            if self.gating_einsum_lora_a is None or self.gating_einsum_lora_b is None
            else (self.gating_einsum_lora_a[1], self.gating_einsum_lora_b[1]),
        )
        activations = gate_value * value
        return self._dot(
            activations,
            self.linear,
            None
            if self.linear_lora_a is None or self.linear_lora_b is None
            else (self.linear_lora_a, self.linear_lora_b),
        )

    def _dot(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        lora_weights: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        base = torch.matmul(x, weight.to(dtype=x.dtype))
        if lora_weights is None:
            return base
        lora_a, lora_b = lora_weights
        lora = torch.matmul(torch.matmul(x, lora_a.to(dtype=x.dtype)), lora_b.to(dtype=x.dtype))
        if self.lora_config is None:
            return base + lora
        return base + lora * self.lora_config.scaling_value
