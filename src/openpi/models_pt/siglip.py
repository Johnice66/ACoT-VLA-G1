from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pt.config import TorchSigLIPConfig


def posemb_sincos_2d(
    height: int,
    width: int,
    dim: int,
    *,
    temperature: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """2D sin/cos position embedding matching the JAX SigLIP helper."""

    if dim % 4 != 0:
        raise ValueError(f"2D sin/cos posemb requires width divisible by 4, got {dim}.")
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = torch.einsum("m,d->md", y.flatten(), omega)
    x = torch.einsum("m,d->md", x.flatten(), omega)
    pos = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pos.to(dtype=dtype).unsqueeze(0)


def _torch_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported SigLIP dtype: {dtype}")


class MLPBlock(nn.Module):
    def __init__(self, *, width: int, mlp_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = mlp_dim or 4 * width
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        return self.fc2(x)


class Encoder1DBlock(nn.Module):
    def __init__(self, *, width: int, mlp_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLPBlock(width=width, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dropout(y)
        y = self.norm2(x)
        return x + self.dropout(self.mlp(y))


class Encoder(nn.Module):
    def __init__(self, *, width: int, depth: int, mlp_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Encoder1DBlock(width=width, mlp_dim=mlp_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(width, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.encoder_norm(x)


class SigLIPModule(nn.Module):
    """Torch SigLIP ViT encoder used by PaliGemma image prefix tokens."""

    def __init__(
        self,
        *,
        num_classes: int | None,
        config: TorchSigLIPConfig | None = None,
        dtype: str = "bfloat16",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = config or TorchSigLIPConfig()
        self.num_classes = num_classes
        self.dtype = _torch_dtype(dtype)
        self.embedding = nn.Conv2d(
            3,
            self.config.width,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            padding=0,
        )
        if self.config.posemb == "learn":
            num_patches = self.config.num_patches or 1
            self.pos_embedding: nn.Parameter | None = nn.Parameter(torch.empty(1, num_patches, self.config.width))
            nn.init.normal_(self.pos_embedding, std=1 / math.sqrt(self.config.width))
        elif self.config.posemb == "sincos2d":
            self.pos_embedding = None
        else:
            raise ValueError(f"Unknown SigLIP position embedding: {self.config.posemb}")
        self.transformer = Encoder(
            width=self.config.width,
            depth=self.config.depth,
            mlp_dim=self.config.mlp_dim,
            num_heads=self.config.num_heads,
            dropout=dropout,
        )
        self.head = None if num_classes is None else nn.Linear(self.config.width, num_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 4:
            raise ValueError(f"SigLIP image input must be [B,H,W,C] or [B,C,H,W], got {tuple(image.shape)}.")
        image = image.to(dtype=self.embedding.weight.dtype)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] != 3:
            raise ValueError(f"SigLIP image input must have 3 channels, got {tuple(image.shape)}.")

        x = self.embedding(image)
        batch_size, width, patch_h, patch_w = x.shape
        x = x.flatten(2).transpose(1, 2)

        if self.pos_embedding is None:
            pos_embedding = posemb_sincos_2d(patch_h, patch_w, width, dtype=torch.float32, device=x.device)
        else:
            if self.pos_embedding.shape[1] != x.shape[1]:
                raise ValueError(
                    f"Learned SigLIP pos_embedding has {self.pos_embedding.shape[1]} patches, "
                    f"but image produced {x.shape[1]} patches."
                )
            pos_embedding = self.pos_embedding
        x = x + pos_embedding.to(dtype=x.dtype)
        transformer_dtype = self.transformer.encoder_norm.weight.dtype
        encoded = self.transformer(x.to(dtype=transformer_dtype)).to(dtype=self.dtype)

        if self.config.pool_type == "none":
            output = encoded
        elif self.config.pool_type == "gap":
            output = encoded.mean(dim=1)
        elif self.config.pool_type == "0":
            output = encoded[:, 0]
        else:
            raise ValueError(f"Unsupported Torch SigLIP pool_type: {self.config.pool_type}")

        if self.head is not None:
            output = self.head(output.to(dtype=self.head.weight.dtype)).to(dtype=self.dtype)

        out = {
            "encoded": encoded,
            "pre_logits": output,
            "pre_logits_2d": encoded.reshape(batch_size, patch_h, patch_w, -1),
        }
        return output, out


Module = SigLIPModule
