from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pt import lora
from openpi.models_pt.config import TorchGemmaConfig

PALIGEMMA_VOCAB_SIZE = 257_152


def _torch_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported Gemma dtype: {dtype}")


def _lora_config(config: TorchGemmaConfig) -> lora.LoRAConfig | None:
    if config.lora_rank is None:
        return None
    return lora.LoRAConfig(rank=config.lora_rank, alpha=config.lora_alpha or float(config.lora_rank))


def _apply_rope(x: torch.Tensor, *, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply Gemma RoPE to tensors shaped [B, L, heads, head_dim]."""

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        x.shape[-1] // 2,
        dtype=torch.float32,
        device=x.device,
    )
    timescale = max_wavelength**freq_exponents
    radians = positions.to(dtype=torch.float32)[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.chunk(x.to(dtype=torch.float32), 2, dim=-1)
    result = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return result.to(dtype=x.dtype)


def _name(name: str, expert_index: int) -> str:
    return name if expert_index == 0 else f"{name}_{expert_index}"


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if (x is None) != (y is None):
        raise ValueError("Residual inputs must either both exist or both be None.")
    if x is None or y is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate


class RMSNorm(nn.Module):
    """Gemma RMSNorm matching either regular RMSNorm or AdaRMS parameter layout."""

    def __init__(self, width: int, *, eps: float = 1e-6, use_adarms: bool = False):
        super().__init__()
        self.width = width
        self.eps = eps
        self.use_adarms = use_adarms
        if use_adarms:
            self.scale = None
            self.adarms_weight = nn.Parameter(torch.zeros(3 * width, width))
            self.adarms_bias = nn.Parameter(torch.zeros(3 * width))
        else:
            self.scale = nn.Parameter(torch.zeros(width))
            self.adarms_weight = None
            self.adarms_bias = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        variance = torch.mean(torch.square(x.to(dtype=torch.float32)), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)

        if not self.use_adarms:
            if self.scale is None:
                raise RuntimeError("Regular RMSNorm is missing scale.")
            return (normed * (1 + self.scale.to(dtype=dtype))).to(dtype=dtype), None

        if cond is None:
            return normed.to(dtype=dtype), None
        if self.adarms_weight is None or self.adarms_bias is None:
            raise RuntimeError("AdaRMS parameters are not initialized.")

        modulation = F.linear(
            cond.to(dtype=dtype),
            self.adarms_weight.to(dtype=dtype),
            self.adarms_bias.to(dtype=dtype),
        )
        scale, shift, gate = torch.chunk(modulation[:, None, :], 3, dim=-1)
        return (normed * (1 + scale) + shift).to(dtype=dtype), gate


class Embedder(nn.Module):
    """Token embedding table used by the first Gemma expert."""

    def __init__(self, *, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.input_embedding = nn.Parameter(torch.empty(vocab_size, embed_dim))
        nn.init.normal_(self.input_embedding)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = F.embedding(tokens, self.input_embedding)
        return embedded * (self.embed_dim**0.5)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.input_embedding.to(dtype=x.dtype).T)


class Attention(nn.Module):
    """Multi-expert Gemma attention.

    The first expert corresponds to PaliGemma tokens. Additional experts carry action/reasoner tokens and use separate
    projection weights while sharing the same attention operation.
    """

    def __init__(self, configs: Sequence[TorchGemmaConfig]):
        super().__init__()
        self.configs = tuple(configs)
        if not self.configs:
            raise ValueError("Gemma Attention requires at least one config.")
        if not all(config.head_dim == self.configs[0].head_dim for config in self.configs):
            raise ValueError("All Gemma experts must use the same head_dim.")
        if not all(config.num_heads == self.configs[0].num_heads for config in self.configs):
            raise ValueError("All Gemma experts must use the same num_heads.")
        if not all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs):
            raise ValueError("All Gemma experts must use the same num_kv_heads.")

        self.qkv_einsums = nn.ModuleList()
        self.q_einsums = nn.ModuleList()
        self.kv_einsums = nn.ModuleList()
        self.attn_vec_einsums = nn.ModuleList()
        for config in self.configs:
            lora_config = _lora_config(config)
            if config.num_kv_heads == config.num_heads:
                self.qkv_einsums.append(
                    lora.Einsum((3, config.num_heads, config.width, config.head_dim), lora_config=lora_config)
                )
                self.q_einsums.append(nn.Identity())
                self.kv_einsums.append(nn.Identity())
            else:
                self.qkv_einsums.append(nn.Identity())
                self.q_einsums.append(
                    lora.Einsum((config.num_heads, config.width, config.head_dim), lora_config=lora_config)
                )
                self.kv_einsums.append(
                    lora.Einsum((2, config.num_kv_heads, config.width, config.head_dim), lora_config=lora_config)
                )
            self.attn_vec_einsums.append(
                lora.Einsum((config.num_heads, config.head_dim, config.width), lora_config=lora_config)
            )

    def forward(
        self,
        xs: Sequence[torch.Tensor | None],
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        dtype = next((x.dtype for x in xs if x is not None), torch.float32)
        qkvs = []
        token_lengths = []
        active_expert_indices = []
        for index, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                token_lengths.append(0)
                continue
            token_lengths.append(x.shape[1])
            active_expert_indices.append(index)
            if config.num_kv_heads == config.num_heads:
                qkv = self.qkv_einsums[index]("bsd,akdh->abskh", x)
                qkvs.append((qkv[0], qkv[1], qkv[2]))
            else:
                q = self.q_einsums[index]("btd,ndh->btnh", x)
                k, v = self.kv_einsums[index]("bsd,akdh->abskh", x)
                qkvs.append((q, k, v))

        if not qkvs:
            return [None for _ in xs], kv_cache  # type: ignore[return-value]

        q = torch.cat([item[0] for item in qkvs], dim=1)
        k = torch.cat([item[1] for item in qkvs], dim=1)
        v = torch.cat([item[2] for item in qkvs], dim=1)

        q = _apply_rope(q, positions=positions)
        q = q * (self.configs[0].head_dim**-0.5)
        k = _apply_rope(k, positions=positions)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)

        batch_size, query_len, _, head_dim = q.shape
        num_kv_heads = self.configs[0].num_kv_heads
        groups = self.configs[0].num_heads // num_kv_heads
        q = q.reshape(batch_size, query_len, num_kv_heads, groups, head_dim)
        logits = torch.einsum("btkgh,bskh->bkgts", q, k)

        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        expected_mask_shape = (batch_size, 1, query_len, k.shape[1])
        if attn_mask.shape != expected_mask_shape:
            raise ValueError(
                f"Attention mask with shape {tuple(attn_mask.shape)} but expected {expected_mask_shape}."
            )
        logits = logits.masked_fill(~attn_mask[:, :, None, :, :], torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1).to(dtype=dtype)

        encoded = torch.einsum("bkgts,bskh->btkgh", probs, v)
        encoded = encoded.reshape(batch_size, query_len, self.configs[0].num_heads, head_dim)

        outputs: list[torch.Tensor | None] = [None for _ in xs]
        start = 0
        for index, length in zip(active_expert_indices, (token_lengths[i] for i in active_expert_indices), strict=True):
            end = start + length
            outputs[index] = self.attn_vec_einsums[index]("btnh,nhd->btd", encoded[:, start:end])
            start = end

        return outputs, (k, v)


class Block(nn.Module):
    """Single multi-expert Gemma transformer block."""

    def __init__(self, configs: Sequence[TorchGemmaConfig], *, dropout: float = 0.0):
        super().__init__()
        self.configs = tuple(configs)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.attn = Attention(self.configs)
        self.pre_attention_norms = nn.ModuleList(
            [RMSNorm(config.width, use_adarms=index > 0) for index, config in enumerate(self.configs)]
        )
        self.pre_ffw_norms = nn.ModuleList(
            [RMSNorm(config.width, use_adarms=index > 0) for index, config in enumerate(self.configs)]
        )
        self.mlps = nn.ModuleList(
            [
                lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    lora_config=_lora_config(config),
                )
                for config in self.configs
            ]
        )

    def forward(
        self,
        xs: Sequence[torch.Tensor | None],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        adarms_cond: Sequence[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
        pre_attn = []
        gates = []
        for index, x in enumerate(xs):
            if x is None:
                pre_attn.append(None)
                gates.append(None)
                continue
            normed, gate = self.pre_attention_norms[index](x, adarms_cond[index])
            pre_attn.append(normed)
            gates.append(gate)

        post_attn, kv_cache = self.attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = [self.dropout(x) if x is not None else None for x in post_attn]
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]

        post_ffn = []
        gates = []
        for index, x in enumerate(xs):
            if x is None:
                post_ffn.append(None)
                gates.append(None)
                continue
            normed, gate = self.pre_ffw_norms[index](x, adarms_cond[index])
            post_ffn.append(self.mlps[index](normed))
            gates.append(gate)

        post_ffn = [self.dropout(x) if x is not None else None for x in post_ffn]
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_ffn, gates, strict=True)]
        return xs, kv_cache


class GemmaModule(nn.Module):
    """Torch Gemma module for the ACoT migration path."""

    def __init__(
        self,
        configs: Sequence[TorchGemmaConfig],
        *,
        embed_dtype: str = "bfloat16",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.configs = tuple(configs)
        if not self.configs:
            raise ValueError("GemmaModule requires at least one config.")
        if not all(config.depth == self.configs[0].depth for config in self.configs):
            raise ValueError("All Gemma experts must use the same depth.")
        self.embed_dtype = _torch_dtype(embed_dtype)
        self.embedder = Embedder(vocab_size=PALIGEMMA_VOCAB_SIZE, embed_dim=self.configs[0].width)
        self.layers = nn.ModuleList([Block(self.configs, dropout=dropout) for _ in range(self.configs[0].depth)])
        self.final_norms = nn.ModuleList(
            [RMSNorm(config.width, use_adarms=index > 0) for index, config in enumerate(self.configs)]
        )

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedder.encode(tokens).to(dtype=self.embed_dtype)

    def forward(
        self,
        embedded: Sequence[torch.Tensor | None],
        positions: torch.Tensor,
        mask: torch.Tensor,
        adarms_cond: Sequence[torch.Tensor | None] | None = None,
        *,
        kv_cache: Sequence[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[tuple[torch.Tensor, torch.Tensor]]]:
        if len(embedded) != len(self.configs):
            raise ValueError(f"Expected {len(self.configs)} expert inputs, got {len(embedded)}.")
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        if len(adarms_cond) != len(self.configs):
            raise ValueError(f"Expected {len(self.configs)} AdaRMS conditions, got {len(adarms_cond)}.")
        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} cache entries, got {len(kv_cache)}.")

        xs = [x.to(dtype=self.embed_dtype) if x is not None else None for x in embedded]
        new_cache = []
        for layer, cache in zip(self.layers, kv_cache, strict=True):
            xs, cache = layer(xs, cache, positions, mask, adarms_cond)
            new_cache.append(cache)

        outputs = []
        for norm, x, cond in zip(self.final_norms, xs, adarms_cond, strict=True):
            outputs.append(norm(x, cond)[0] if x is not None else None)
        return outputs, new_cache


Module = GemmaModule
