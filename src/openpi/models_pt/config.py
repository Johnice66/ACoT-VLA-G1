from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Literal


GemmaVariant = Literal[
    "dummy",
    "gemma_50m",
    "gemma_150m",
    "gemma_250m",
    "gemma_300m",
    "gemma_300m_lora",
    "gemma_500m",
    "gemma_600m",
    "gemma_2b",
    "gemma_2b_lora",
]
BackboneMode = Literal["skeleton", "siglip", "gemma", "full"]


@dataclasses.dataclass(frozen=True)
class TorchGemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_rank: int | None = None
    lora_alpha: float | None = None


@dataclasses.dataclass(frozen=True)
class TorchSigLIPConfig:
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    patch_size: tuple[int, int] = (14, 14)
    pool_type: str = "none"
    posemb: str = "sincos2d"
    num_patches: int | None = None


_GEMMA_CONFIGS: Mapping[str, TorchGemmaConfig] = {
    "dummy": TorchGemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "gemma_50m": TorchGemmaConfig(width=128, depth=18, mlp_dim=1024, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_150m": TorchGemmaConfig(width=768, depth=18, mlp_dim=3072, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_250m": TorchGemmaConfig(width=896, depth=18, mlp_dim=3584, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_300m": TorchGemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_300m_lora": TorchGemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        lora_rank=32,
        lora_alpha=32.0,
    ),
    "gemma_500m": TorchGemmaConfig(width=1408, depth=18, mlp_dim=5632, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_600m": TorchGemmaConfig(width=1536, depth=18, mlp_dim=6144, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b": TorchGemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b_lora": TorchGemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        lora_rank=16,
        lora_alpha=16.0,
    ),
}


def get_gemma_config(variant: str) -> TorchGemmaConfig:
    try:
        return _GEMMA_CONFIGS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown Gemma variant for Torch ACoT: {variant}") from exc


@dataclasses.dataclass(frozen=True)
class TorchACOTConfig:
    """JAX-free copy of the ACoT model settings needed by the Torch inference path."""

    dtype: str = "bfloat16"
    paligemma_variant: GemmaVariant = "gemma_2b"
    coarse_action_expert_variant: GemmaVariant = "gemma_300m"
    action_expert_variant: GemmaVariant = "gemma_300m"
    action_dim: int = 32
    coarse_action_horizon: int = 50
    action_horizon: int = 30
    max_token_len: int | None = None
    pi05: bool = True
    discrete_state_input: bool | None = None
    adopt_explicit_action_reasoner: bool = False
    adopt_implicit_action_reasoner: bool = False
    query_based_implicit_extractor: bool = False
    attention_pooling_implicit_extractor: bool = False
    downsample_based_implicit_extractor: bool = False
    use_real_gemma_backbone: bool = False
    use_real_siglip_backbone: bool = False
    siglip: TorchSigLIPConfig = dataclasses.field(default_factory=TorchSigLIPConfig)

    def __post_init__(self) -> None:
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    def paligemma(self) -> TorchGemmaConfig:
        return get_gemma_config(self.paligemma_variant)

    @property
    def coarse_action_expert(self) -> TorchGemmaConfig:
        return get_gemma_config(self.coarse_action_expert_variant)

    @property
    def action_expert(self) -> TorchGemmaConfig:
        return get_gemma_config(self.action_expert_variant)

    def with_backbone(self, backbone: BackboneMode) -> "TorchACOTConfig":
        include_gemma = backbone in ("gemma", "full")
        include_siglip = backbone in ("siglip", "full")
        siglip_config = self.siglip
        if include_siglip and (siglip_config.posemb != "learn" or siglip_config.num_patches is None):
            siglip_config = dataclasses.replace(
                siglip_config,
                posemb="learn",
                num_patches=siglip_config.num_patches or 256,
            )
        return dataclasses.replace(
            self,
            use_real_gemma_backbone=include_gemma,
            use_real_siglip_backbone=include_siglip,
            siglip=siglip_config,
        )

    @classmethod
    def from_jax_config(cls, config: Any) -> "TorchACOTConfig":
        """Build from the existing Flax ACOTConfig without depending on its concrete type."""

        fields = {field.name for field in dataclasses.fields(cls)}
        values = {name: getattr(config, name) for name in fields if hasattr(config, name)}
        return cls(**values)


def acot_icra_simulation_challenge_config() -> TorchACOTConfig:
    return TorchACOTConfig(
        coarse_action_horizon=30,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        adopt_explicit_action_reasoner=True,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
        siglip=TorchSigLIPConfig(posemb="learn", num_patches=256),
    )
