from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def posemb_sincos(
    pos: torch.Tensor,
    width: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if width % 2 != 0:
        raise ValueError(f"posemb_sincos requires an even width, got {width}.")

    dtype = pos.dtype if torch.is_floating_point(pos) else torch.float32
    pos = pos.to(dtype=dtype)
    scales = torch.linspace(0.0, 1.0, width // 2, dtype=dtype, device=pos.device)
    periods = min_period * (max_period / min_period) ** scales
    angles = pos[..., None] / periods[None, :] * (2.0 * math.pi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def make_attn_mask(input_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
    """Match the JAX ACoT cumulative attention-mask convention."""

    if input_mask.ndim != 2:
        raise ValueError(f"input_mask must be [batch, seq], got {tuple(input_mask.shape)}")
    if ar_mask.ndim == 1:
        ar_mask = ar_mask.unsqueeze(0).expand(input_mask.shape[0], -1)
    elif ar_mask.ndim != 2:
        raise ValueError(f"ar_mask must be [seq] or [batch, seq], got {tuple(ar_mask.shape)}")
    if ar_mask.shape != input_mask.shape:
        raise ValueError(f"ar_mask shape {tuple(ar_mask.shape)} must broadcast to input_mask {tuple(input_mask.shape)}")

    cumulative = torch.cumsum(ar_mask.to(dtype=torch.long), dim=1)
    attention_mask = cumulative[:, None, :] <= cumulative[:, :, None]
    valid_mask = input_mask[:, None, :] & input_mask[:, :, None]
    return attention_mask & valid_mask


class RMSNorm(nn.Module):
    def __init__(self, width: int, *, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.scale)).to(dtype=dtype)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, activate: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.fc1(x)
        if self.activate:
            x = swish(x)
        return self.fc2(x)


class UnifiedAttentionModule(nn.Module):
    """Cross-attention block used by explicit/implicit ACoT reasoning fusion."""

    def __init__(
        self,
        *,
        in_dim_1: int,
        in_dim_2: int,
        out_dim: int,
        hidden_dim: int,
        num_heads: int,
        apply_sigmoid: bool = False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(in_dim_1, hidden_dim)
        self.k_proj = nn.Linear(in_dim_2, hidden_dim)
        self.v_proj = nn.Linear(in_dim_2, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
        feat_1 = feat_1.to(dtype=self.q_proj.weight.dtype)
        feat_2 = feat_2.to(dtype=self.k_proj.weight.dtype)
        query = self.q_proj(feat_1)
        key = self.k_proj(feat_2)
        value = self.v_proj(feat_2)
        q_weight, k_weight, v_weight = self.attn.in_proj_weight.chunk(3, dim=0)
        if self.attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attn.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        batch_size, query_len, embed_dim = q.shape
        key_len = k.shape[1]
        head_dim = embed_dim // self.attn.num_heads
        q = q.reshape(batch_size, query_len, self.attn.num_heads, head_dim)
        k = k.reshape(batch_size, key_len, self.attn.num_heads, head_dim)
        v = v.reshape(batch_size, key_len, self.attn.num_heads, head_dim)

        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
        weights = torch.softmax(logits, dim=-1).to(dtype=q.dtype)
        output = torch.einsum("bhqk,bkhd->bqhd", weights, v)
        output = output.reshape(batch_size, query_len, embed_dim)
        output = self.attn.out_proj(output)
        output = self.fc_out(output)
        if self.apply_sigmoid:
            return torch.sigmoid(output)
        return output


class DownsampleExtractor(nn.Module):
    """Torch equivalent of ACoT's downsample-based implicit extractor."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        downsample_dim: int,
        depth: int = 1,
        group_size: int = 3,
        num_queries: int = 1,
        num_heads: int = 8,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")
        if downsample_dim % num_heads != 0:
            raise ValueError(f"downsample_dim={downsample_dim} must be divisible by num_heads={num_heads}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.downsample_dim = downsample_dim
        self.depth = depth
        self.group_size = group_size
        self.num_groups = (depth + group_size - 1) // group_size
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = downsample_dim // num_heads

        self.query_params = nn.ParameterList([nn.Parameter(torch.empty(num_queries, input_dim)) for _ in range(depth)])
        self.q_proj = nn.ModuleList([nn.Linear(input_dim, downsample_dim) for _ in range(self.num_groups)])
        self.k_proj = nn.ModuleList([nn.Linear(input_dim, downsample_dim) for _ in range(self.num_groups)])
        self.v_proj = nn.ModuleList([nn.Linear(input_dim, downsample_dim) for _ in range(self.num_groups)])
        self.out_proj = nn.ModuleList([nn.Linear(downsample_dim, output_dim) for _ in range(self.num_groups)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for query in self.query_params:
            nn.init.normal_(query)

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("Implicit extractor expects [batch, layers, tokens, dim] key/value tensors.")
        if keys.shape != values.shape:
            raise ValueError(f"Key/value shapes must match, got {tuple(keys.shape)} and {tuple(values.shape)}.")
        batch_size, depth, token_count, dim = keys.shape
        if depth != self.depth:
            raise ValueError(f"Expected {self.depth} layers, got {depth}.")
        if dim != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {dim}.")

        keys = keys.to(dtype=self.q_proj[0].weight.dtype)
        values = values.to(dtype=self.v_proj[0].weight.dtype)
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for layer_index in range(depth):
            group_index = min(layer_index // self.group_size, self.num_groups - 1)
            query = self.query_params[layer_index].unsqueeze(0)
            query = self.q_proj[group_index](query)
            query = query.reshape(1, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
            query = query.expand(batch_size, -1, -1, -1)

            key = self.k_proj[group_index](keys[:, layer_index])
            key = key.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj[group_index](values[:, layer_index])
            value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(query, key.transpose(-1, -2)) * scale
            pooled = torch.matmul(torch.softmax(attn, dim=-1), value)
            if self.num_queries > 1:
                pooled = pooled.mean(dim=2)
            else:
                pooled = pooled.squeeze(dim=2)
            pooled = pooled.transpose(1, 2).reshape(batch_size, self.downsample_dim)
            outputs.append(self.out_proj[group_index](pooled))

        return torch.stack(outputs, dim=1)


def normal_like(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if device.type == "npu":
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def scale_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.matmul(F.softmax(scores, dim=-1), value)
