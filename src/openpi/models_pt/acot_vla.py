from __future__ import annotations

from typing import Any

import torch
from torch import nn

from openpi.models_pt import gemma
from openpi.models_pt import layers
from openpi.models_pt import siglip
from openpi.models_pt.config import TorchACOTConfig
from openpi.models_pt.types import TorchObservation


def _torch_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported Torch ACoT dtype: {dtype}")


class ACOTVLATorch(nn.Module):
    """Torch ACoT-VLA inference skeleton.

    This module preserves the ACoT inference contract and sampling flow while Gemma/SigLIP are being ported. It is the
    place where the full PyTorch vision-language backbone and converted weights will be attached in later phases.
    """

    def __init__(self, config: TorchACOTConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.coarse_action_horizon = config.coarse_action_horizon
        self.dtype = _torch_dtype(config.dtype)
        self.use_real_gemma_backbone = config.use_real_gemma_backbone
        self.use_real_siglip_backbone = config.use_real_siglip_backbone

        action_width = config.action_expert.width
        hidden_width = action_width if self.use_real_gemma_backbone else min(action_width, 1024)

        if config.use_real_siglip_backbone:
            self.paligemma_img = siglip.SigLIPModule(
                num_classes=config.paligemma.width,
                config=config.siglip,
                dtype=config.dtype,
            )

        if self.use_real_gemma_backbone:
            self.paligemma_llm = gemma.GemmaModule(
                (config.paligemma, config.coarse_action_expert, config.action_expert),
                embed_dtype=config.dtype,
            )
            if not config.use_real_siglip_backbone:
                self.image_summary_proj = nn.Linear(3, config.paligemma.width)

        if not self.use_real_gemma_backbone:
            self.state_proj = nn.Linear(config.action_dim, hidden_width)
        self.coarse_action_in_proj = nn.Linear(config.action_dim, hidden_width)
        self.action_in_proj = nn.Linear(config.action_dim, hidden_width)
        self.coarse_time_mlp = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.coarse_action_out_proj = nn.Linear(hidden_width, config.action_dim)
        self.action_out_proj = nn.Linear(hidden_width, config.action_dim)

        self.adopt_explicit_action_reasoner = config.adopt_explicit_action_reasoner
        self.adopt_implicit_action_reasoner = config.adopt_implicit_action_reasoner

        if self.adopt_explicit_action_reasoner:
            self.explicit_action_reasoner = layers.UnifiedAttentionModule(
                in_dim_1=hidden_width,
                in_dim_2=hidden_width,
                out_dim=hidden_width,
                hidden_dim=hidden_width,
                num_heads=4,
            )

        if self.adopt_implicit_action_reasoner:
            self.implicit_action_reasoner = layers.DownsampleExtractor(
                input_dim=config.paligemma.head_dim,
                output_dim=hidden_width,
                depth=config.paligemma.depth,
                downsample_dim=config.paligemma.head_dim // 2,
                num_heads=config.paligemma.num_heads,
                group_size=3,
            )
            self.implicit_action_reasoner_interact = layers.UnifiedAttentionModule(
                in_dim_1=hidden_width,
                in_dim_2=hidden_width,
                out_dim=hidden_width,
                hidden_dim=hidden_width,
                num_heads=4,
            )

        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            self.explicit_action_reason_proj = nn.Linear(2 * hidden_width, hidden_width)
            self.implicit_action_reason_proj = nn.Linear(2 * hidden_width, hidden_width)
            self.action_reasoning_fusion = layers.UnifiedAttentionModule(
                in_dim_1=2 * hidden_width,
                in_dim_2=2 * hidden_width,
                out_dim=hidden_width,
                hidden_dim=hidden_width,
                num_heads=4,
            )
        elif self.adopt_explicit_action_reasoner or self.adopt_implicit_action_reasoner:
            self.action_reasoning_fusion = layers.MLP(2 * hidden_width, hidden_width, hidden_width, activate=False)

        if self.use_real_gemma_backbone and config.use_real_siglip_backbone:
            self.model_status = "torch_acot_gemma_siglip_full_inference"
            self.expected_missing_backbone = ()
        elif self.use_real_gemma_backbone:
            self.model_status = "torch_acot_gemma_prefix"
            self.expected_missing_backbone = ("PaliGemma.img",)
        elif config.use_real_siglip_backbone:
            self.model_status = "torch_acot_siglip_prefix"
            self.expected_missing_backbone = ("PaliGemma.llm",)
        else:
            self.model_status = "torch_acot_skeleton"
            self.expected_missing_backbone = ("PaliGemma.llm", "PaliGemma.img")

    def embed_prefix(self, observation: TorchObservation) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prefix tokens and masks.

        Full parity requires the PyTorch SigLIP/Gemma port. For phase 1, this creates a deterministic prefix summary
        from state, image masks and prompt masks so the Torch policy has the same data-flow shape.
        """

        if self.use_real_gemma_backbone:
            return self._embed_prefix_with_gemma(observation)
        if self.use_real_siglip_backbone:
            return self._embed_prefix_with_siglip_skeleton(observation)

        state_tokens = self.state_proj(observation.state.to(dtype=self.state_proj.weight.dtype)).unsqueeze(1)
        image_mask_value = torch.stack(
            [observation.image_masks[key].to(dtype=state_tokens.dtype) for key in observation.image_masks],
            dim=-1,
        ).mean(dim=-1, keepdim=True)
        state_tokens = state_tokens * image_mask_value.unsqueeze(-1)

        if observation.tokenized_prompt_mask is None:
            return state_tokens, torch.ones(state_tokens.shape[:2], dtype=torch.bool, device=state_tokens.device)

        prompt_scale = observation.tokenized_prompt_mask.to(dtype=state_tokens.dtype).mean(dim=-1, keepdim=True)
        prompt_tokens = state_tokens * (1.0 + prompt_scale.unsqueeze(-1))
        tokens = torch.cat([state_tokens, prompt_tokens], dim=1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask

    def _embed_prefix_with_gemma(self, observation: TorchObservation) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self._prefix_tokens_with_gemma(observation)
        attn_mask = layers.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.to(dtype=torch.long), dim=1) - 1
        encoded, _ = self.paligemma_llm([prefix_tokens, None, None], positions, attn_mask)
        if encoded[0] is None:
            raise RuntimeError("Gemma prefix encoder returned no PaliGemma tokens.")
        return encoded[0], prefix_mask

    def _prefix_tokens_with_gemma(
        self,
        observation: TorchObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = []
        input_masks = []

        for name, image in observation.images.items():
            if self.config.use_real_siglip_backbone:
                image_tokens, _ = self.paligemma_img(image)
                tokens.append(image_tokens)
                input_masks.append(
                    observation.image_masks[name].reshape(observation.batch_size, 1).expand(-1, image_tokens.shape[1])
                )
            else:
                image_summary = image.to(dtype=self.image_summary_proj.weight.dtype).mean(dim=(1, 2))
                image_token = self.image_summary_proj(image_summary).unsqueeze(1)
                tokens.append(image_token)
                input_masks.append(observation.image_masks[name].reshape(observation.batch_size, 1))

        if observation.tokenized_prompt is not None:
            prompt_tokens = self.paligemma_llm.embed(observation.tokenized_prompt)
            tokens.append(prompt_tokens)
            input_masks.append(observation.tokenized_prompt_mask)

        prefix_tokens = torch.cat(tokens, dim=1)
        prefix_mask = torch.cat(input_masks, dim=1).to(dtype=torch.bool)
        prefix_ar_mask = torch.zeros(prefix_tokens.shape[1], dtype=torch.bool, device=prefix_tokens.device)
        return prefix_tokens, prefix_mask, prefix_ar_mask

    def _prefix_context_with_gemma(
        self,
        observation: TorchObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self._prefix_tokens_with_gemma(observation)
        prefix_attn_mask = layers.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.to(dtype=torch.long), dim=1) - 1
        _, kv_cache = self.paligemma_llm([prefix_tokens, None, None], positions, prefix_attn_mask)
        return prefix_tokens, prefix_mask, prefix_ar_mask, kv_cache

    def _embed_prefix_with_siglip_skeleton(self, observation: TorchObservation) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.state_proj(observation.state.to(dtype=self.state_proj.weight.dtype)).unsqueeze(1)]
        masks = [torch.ones((observation.batch_size, 1), dtype=torch.bool, device=observation.state.device)]

        for name, image in observation.images.items():
            image_tokens, _ = self.paligemma_img(image)
            image_tokens = self._project_prefix_width(image_tokens, tokens[0].shape[-1])
            mask = observation.image_masks[name].reshape(observation.batch_size, 1).expand(-1, image_tokens.shape[1])
            tokens.append(image_tokens * mask.unsqueeze(-1).to(dtype=image_tokens.dtype))
            masks.append(mask)

        if observation.tokenized_prompt_mask is not None:
            prompt_scale = observation.tokenized_prompt_mask.to(dtype=tokens[0].dtype).mean(dim=-1, keepdim=True)
            tokens[0] = tokens[0] * (1.0 + prompt_scale.unsqueeze(-1))

        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)

    def _implicit_action_reason(self, prefix_tokens: torch.Tensor) -> torch.Tensor | None:
        if not self.adopt_implicit_action_reasoner:
            return None
        prefix_features = self._prefix_tokens_to_implicit_features(prefix_tokens)
        keys = prefix_features.unsqueeze(1).expand(-1, self.config.paligemma.depth, -1, -1)
        return self.implicit_action_reasoner(keys, keys)

    def _implicit_action_reason_from_kv_cache(
        self,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor | None:
        if not self.adopt_implicit_action_reasoner:
            return None
        keys = []
        values = []
        for key, value in kv_cache:
            if key.shape[2] != 1 or value.shape[2] != 1:
                raise ValueError(
                    "ACoT implicit extractor expects Gemma KV cache with one kv head, "
                    f"got key/value shapes {tuple(key.shape)} and {tuple(value.shape)}."
                )
            keys.append(key[:, :, 0, :])
            values.append(value[:, :, 0, :])
        return self.implicit_action_reasoner(torch.stack(keys, dim=1), torch.stack(values, dim=1))

    def _prefix_tokens_to_implicit_features(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        target_dim = self.config.paligemma.head_dim
        features = prefix_tokens
        if features.shape[-1] == target_dim:
            return features
        if features.shape[-1] > target_dim:
            if features.shape[-1] % target_dim == 0:
                return features.reshape(*features.shape[:-1], -1, target_dim).mean(dim=-2)
            return features[..., :target_dim]
        pad_width = target_dim - features.shape[-1]
        return torch.nn.functional.pad(features, (0, pad_width))

    def _project_prefix_width(self, prefix_tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
        features = prefix_tokens
        if features.shape[-1] == target_dim:
            return features
        if features.shape[-1] > target_dim:
            if features.shape[-1] % target_dim == 0:
                return features.reshape(*features.shape[:-1], -1, target_dim).mean(dim=-2)
            return features[..., :target_dim]
        return torch.nn.functional.pad(features, (0, target_dim - features.shape[-1]))

    def _embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        explicit_action_reason: torch.Tensor | None,
        implicit_action_reason: torch.Tensor | None,
        suffix_type: str,
    ) -> torch.Tensor:
        if suffix_type == "reasoner":
            tokens = self.coarse_action_in_proj(noisy_actions.to(dtype=self.coarse_action_in_proj.weight.dtype))
            time_emb = layers.posemb_sincos(timestep, tokens.shape[-1]).to(dtype=tokens.dtype)
            tokens = tokens + self.coarse_time_mlp(time_emb).unsqueeze(1)
            return tokens

        if suffix_type != "expert":
            raise ValueError(f"Unknown suffix_type: {suffix_type}")

        tokens = self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))
        time_emb = layers.posemb_sincos(timestep, tokens.shape[-1]).to(dtype=tokens.dtype)
        tokens = tokens + self.time_mlp(time_emb).unsqueeze(1)

        if self.adopt_explicit_action_reasoner and explicit_action_reason is not None:
            explicit_tokens = self.coarse_action_in_proj(
                explicit_action_reason.to(dtype=self.coarse_action_in_proj.weight.dtype)
            )
            aligned_explicit = self.explicit_action_reasoner(tokens, explicit_tokens)
        else:
            aligned_explicit = None

        if self.adopt_implicit_action_reasoner and implicit_action_reason is not None:
            aligned_implicit = self.implicit_action_reasoner_interact(tokens, implicit_action_reason)
        else:
            aligned_implicit = None

        if aligned_explicit is not None and aligned_implicit is not None:
            explicit = self.explicit_action_reason_proj(torch.cat([tokens, aligned_explicit], dim=-1))
            implicit = self.implicit_action_reason_proj(torch.cat([tokens, aligned_implicit], dim=-1))
            fused = torch.cat([explicit, implicit], dim=-1)
            return self.action_reasoning_fusion(fused, fused)
        if aligned_explicit is not None:
            return self.action_reasoning_fusion(torch.cat([tokens, aligned_explicit], dim=-1))
        if aligned_implicit is not None:
            return self.action_reasoning_fusion(torch.cat([tokens, aligned_implicit], dim=-1))
        return tokens

    def _time_condition(self, timestep: torch.Tensor, width: int, *, suffix_type: str) -> torch.Tensor:
        time_emb = layers.posemb_sincos(timestep, width)
        if suffix_type == "reasoner":
            time_emb = self.coarse_time_mlp(time_emb.to(dtype=self.coarse_time_mlp[0].weight.dtype))
        elif suffix_type == "expert":
            time_emb = self.time_mlp(time_emb.to(dtype=self.time_mlp[0].weight.dtype))
        else:
            raise ValueError(f"Unknown suffix_type: {suffix_type}")
        return layers.swish(time_emb)

    def _embed_suffix_for_gemma(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        explicit_action_reason: torch.Tensor | None,
        implicit_action_reason: torch.Tensor | None,
        suffix_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = noisy_actions.device
        batch_size = noisy_actions.shape[0]
        if suffix_type == "reasoner":
            tokens = self.coarse_action_in_proj(noisy_actions.to(dtype=self.coarse_action_in_proj.weight.dtype))
            adarms_cond = self._time_condition(timestep, tokens.shape[-1], suffix_type=suffix_type)
            horizon = self.coarse_action_horizon
        elif suffix_type == "expert":
            tokens = self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))
            adarms_cond = self._time_condition(timestep, tokens.shape[-1], suffix_type=suffix_type)
            horizon = self.action_horizon

            if self.adopt_explicit_action_reasoner and explicit_action_reason is not None:
                explicit_tokens = self.coarse_action_in_proj(
                    explicit_action_reason.to(dtype=self.coarse_action_in_proj.weight.dtype)
                )
                aligned_explicit = self.explicit_action_reasoner(tokens, explicit_tokens)
            else:
                aligned_explicit = None

            if self.adopt_implicit_action_reasoner and implicit_action_reason is not None:
                aligned_implicit = self.implicit_action_reasoner_interact(tokens, implicit_action_reason)
            else:
                aligned_implicit = None

            if aligned_explicit is not None and aligned_implicit is not None:
                explicit = self.explicit_action_reason_proj(torch.cat([tokens, aligned_explicit], dim=-1))
                implicit = self.implicit_action_reason_proj(torch.cat([tokens, aligned_implicit], dim=-1))
                fused = torch.cat([explicit, implicit], dim=-1)
                tokens = self.action_reasoning_fusion(fused, fused)
            elif aligned_explicit is not None:
                tokens = self.action_reasoning_fusion(torch.cat([tokens, aligned_explicit], dim=-1))
            elif aligned_implicit is not None:
                tokens = self.action_reasoning_fusion(torch.cat([tokens, aligned_implicit], dim=-1))
        else:
            raise ValueError(f"Unknown suffix_type: {suffix_type}")

        suffix_mask = torch.ones((batch_size, horizon), dtype=torch.bool, device=device)
        suffix_ar_mask = torch.zeros(horizon, dtype=torch.bool, device=device)
        suffix_ar_mask[0] = True
        return tokens, suffix_mask, suffix_ar_mask, adarms_cond

    def _velocity(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        explicit_action_reason: torch.Tensor | None,
        implicit_action_reason: torch.Tensor | None,
        suffix_type: str,
    ) -> torch.Tensor:
        tokens = self._embed_suffix(
            x_t,
            timestep,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suffix_type=suffix_type,
        )
        if suffix_type == "reasoner":
            return self.coarse_action_out_proj(tokens).to(dtype=x_t.dtype)
        return self.action_out_proj(tokens).to(dtype=x_t.dtype)

    def _velocity_with_gemma(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        prefix_mask: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        explicit_action_reason: torch.Tensor | None,
        implicit_action_reason: torch.Tensor | None,
        suffix_type: str,
    ) -> torch.Tensor:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_suffix_for_gemma(
            x_t,
            timestep,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suffix_type=suffix_type,
        )
        suffix_attn_mask = layers.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = prefix_mask[:, None, :].expand(-1, suffix_tokens.shape[1], -1)
        full_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
        positions = prefix_mask.to(dtype=torch.long).sum(dim=-1, keepdim=True)
        positions = positions + torch.cumsum(suffix_mask.to(dtype=torch.long), dim=-1) - 1

        if suffix_type == "reasoner":
            outputs, _ = self.paligemma_llm(
                [None, suffix_tokens, None],
                positions,
                full_attn_mask,
                [None, adarms_cond, None],
                kv_cache=kv_cache,
            )
            suffix_out = outputs[1]
            if suffix_out is None:
                raise RuntimeError("Gemma reasoner suffix forward returned no suffix output.")
            suffix_out = suffix_out[:, -self.coarse_action_horizon :].to(dtype=self.coarse_action_out_proj.weight.dtype)
            return self.coarse_action_out_proj(suffix_out).to(dtype=x_t.dtype)

        if suffix_type == "expert":
            outputs, _ = self.paligemma_llm(
                [None, None, suffix_tokens],
                positions,
                full_attn_mask,
                [None, None, adarms_cond],
                kv_cache=kv_cache,
            )
            suffix_out = outputs[2]
            if suffix_out is None:
                raise RuntimeError("Gemma expert suffix forward returned no suffix output.")
            suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out).to(dtype=x_t.dtype)

        raise ValueError(f"Unknown suffix_type: {suffix_type}")

    def compute_training_loss(
        self,
        observation: TorchObservation,
        actions: torch.Tensor,
        coarse_actions: torch.Tensor | None = None,
        *,
        timestep: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        coarse_action_noise: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the Torch ACoT flow-matching training loss.

        The random tensors are injectable so GPU/NPU validation can run with the exact same batch, noise, and timestep.
        """

        device = observation.state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if actions.shape != (observation.batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                "actions must have shape "
                f"{(observation.batch_size, self.action_horizon, self.action_dim)}, got {tuple(actions.shape)}."
            )

        if self.adopt_explicit_action_reasoner:
            if coarse_actions is None:
                raise ValueError("coarse_actions is required when adopt_explicit_action_reasoner=True.")
            coarse_actions = coarse_actions.to(device=device, dtype=torch.float32)
            expected_coarse_shape = (observation.batch_size, self.coarse_action_horizon, self.action_dim)
            if coarse_actions.shape != expected_coarse_shape:
                raise ValueError(
                    f"coarse_actions must have shape {expected_coarse_shape}, got {tuple(coarse_actions.shape)}."
                )
        elif coarse_actions is not None:
            coarse_actions = coarse_actions.to(device=device, dtype=torch.float32)

        if timestep is None:
            timestep = torch.distributions.Beta(1.5, 1.0).sample((observation.batch_size,)).to(device=device)
            timestep = timestep * 0.999 + 0.001
        else:
            timestep = timestep.to(device=device, dtype=torch.float32)
        if timestep.shape != (observation.batch_size,):
            raise ValueError(f"timestep must have shape {(observation.batch_size,)}, got {tuple(timestep.shape)}.")

        if action_noise is None:
            action_noise = torch.randn_like(actions)
        else:
            action_noise = action_noise.to(device=device, dtype=torch.float32)
        if action_noise.shape != actions.shape:
            raise ValueError(f"action_noise must have shape {tuple(actions.shape)}, got {tuple(action_noise.shape)}.")

        time_expanded = timestep[:, None, None]
        x_expert_t = time_expanded * action_noise + (1.0 - time_expanded) * actions
        target_expert_velocity = action_noise - actions

        if self.use_real_gemma_backbone:
            _, prefix_mask, _, kv_cache = self._prefix_context_with_gemma(observation)
            implicit_action_reason = self._implicit_action_reason_from_kv_cache(kv_cache)
        else:
            prefix_tokens, _ = self.embed_prefix(observation)
            prefix_mask = None
            kv_cache = None
            implicit_action_reason = self._implicit_action_reason(prefix_tokens)

        losses: dict[str, torch.Tensor] = {}
        explicit_action_reason = None
        if self.adopt_explicit_action_reasoner:
            assert coarse_actions is not None
            if coarse_action_noise is None:
                coarse_action_noise = torch.randn_like(coarse_actions)
            else:
                coarse_action_noise = coarse_action_noise.to(device=device, dtype=torch.float32)
            if coarse_action_noise.shape != coarse_actions.shape:
                raise ValueError(
                    f"coarse_action_noise must have shape {tuple(coarse_actions.shape)}, "
                    f"got {tuple(coarse_action_noise.shape)}."
                )

            x_ref_t = time_expanded * coarse_action_noise + (1.0 - time_expanded) * coarse_actions
            target_ref_velocity = coarse_action_noise - coarse_actions
            if self.use_real_gemma_backbone:
                assert prefix_mask is not None and kv_cache is not None
                predicted_ref_velocity = self._velocity_with_gemma(
                    x_ref_t,
                    timestep,
                    prefix_mask=prefix_mask,
                    kv_cache=kv_cache,
                    explicit_action_reason=None,
                    implicit_action_reason=None,
                    suffix_type="reasoner",
                )
            else:
                predicted_ref_velocity = self._velocity(
                    x_ref_t,
                    timestep,
                    explicit_action_reason=None,
                    implicit_action_reason=None,
                    suffix_type="reasoner",
                )
            losses["coarse_loss"] = torch.mean(torch.square(target_ref_velocity - predicted_ref_velocity))
            explicit_action_reason = coarse_actions
        else:
            losses["coarse_loss"] = torch.zeros((), dtype=torch.float32, device=device)
            target_ref_velocity = None
            predicted_ref_velocity = None

        if self.use_real_gemma_backbone:
            assert prefix_mask is not None and kv_cache is not None
            predicted_expert_velocity = self._velocity_with_gemma(
                x_expert_t,
                timestep,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
                suffix_type="expert",
            )
        else:
            predicted_expert_velocity = self._velocity(
                x_expert_t,
                timestep,
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
                suffix_type="expert",
            )
        losses["action_loss"] = torch.mean(torch.square(target_expert_velocity - predicted_expert_velocity))
        losses["total_loss"] = losses["coarse_loss"] + losses["action_loss"]
        losses["timestep_mean"] = timestep.mean()
        losses["predicted_action_velocity_mean"] = predicted_expert_velocity.float().mean()
        losses["predicted_action_velocity_std"] = predicted_expert_velocity.float().std(unbiased=False)
        if return_debug:
            losses["target_expert_velocity"] = target_expert_velocity
            losses["predicted_expert_velocity"] = predicted_expert_velocity
            if target_ref_velocity is not None and predicted_ref_velocity is not None:
                losses["target_ref_velocity"] = target_ref_velocity
                losses["predicted_ref_velocity"] = predicted_ref_velocity
        return losses

    @torch.no_grad()
    def sample_actions(
        self,
        observation: TorchObservation,
        *,
        generator: torch.Generator | None = None,
        num_steps: int = 10,
        initial_coarse_action_noise: torch.Tensor | None = None,
        initial_action_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")

        device = observation.state.device
        dtype = torch.float32
        batch_size = observation.batch_size
        dt = -1.0 / num_steps

        if self.use_real_gemma_backbone:
            prefix_tokens, prefix_mask, _, kv_cache = self._prefix_context_with_gemma(observation)
            implicit_action_reason = self._implicit_action_reason_from_kv_cache(kv_cache)
        else:
            prefix_tokens, _ = self.embed_prefix(observation)
            prefix_mask = None
            kv_cache = None
            implicit_action_reason = self._implicit_action_reason(prefix_tokens)

        coarse_actions = None
        if self.adopt_explicit_action_reasoner:
            coarse_shape = (batch_size, self.coarse_action_horizon, self.action_dim)
            if initial_coarse_action_noise is None:
                coarse_actions = layers.normal_like(
                    coarse_shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            else:
                coarse_actions = initial_coarse_action_noise.to(device=device, dtype=dtype)
                if tuple(coarse_actions.shape) != coarse_shape:
                    raise ValueError(
                        f"initial_coarse_action_noise must have shape {coarse_shape}, got {tuple(coarse_actions.shape)}."
                    )
            time = torch.ones(batch_size, dtype=dtype, device=device)
            for _ in range(num_steps):
                if kv_cache is None or prefix_mask is None:
                    velocity = self._velocity(
                        coarse_actions,
                        time,
                        explicit_action_reason=None,
                        implicit_action_reason=None,
                        suffix_type="reasoner",
                    )
                else:
                    velocity = self._velocity_with_gemma(
                        coarse_actions,
                        time,
                        prefix_mask=prefix_mask,
                        kv_cache=kv_cache,
                        explicit_action_reason=None,
                        implicit_action_reason=None,
                        suffix_type="reasoner",
                    )
                coarse_actions = coarse_actions + dt * velocity
                time = time + dt

        action_shape = (batch_size, self.action_horizon, self.action_dim)
        if initial_action_noise is None:
            actions = layers.normal_like(
                action_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            actions = initial_action_noise.to(device=device, dtype=dtype)
            if tuple(actions.shape) != action_shape:
                raise ValueError(f"initial_action_noise must have shape {action_shape}, got {tuple(actions.shape)}.")
        time = torch.ones(batch_size, dtype=dtype, device=device)
        for _ in range(num_steps):
            if kv_cache is None or prefix_mask is None:
                velocity = self._velocity(
                    actions,
                    time,
                    explicit_action_reason=coarse_actions,
                    implicit_action_reason=implicit_action_reason,
                    suffix_type="expert",
                )
            else:
                velocity = self._velocity_with_gemma(
                    actions,
                    time,
                    prefix_mask=prefix_mask,
                    kv_cache=kv_cache,
                    explicit_action_reason=coarse_actions,
                    implicit_action_reason=implicit_action_reason,
                    suffix_type="expert",
                )
            actions = actions + dt * velocity
            time = time + dt

        outputs: dict[str, torch.Tensor] = {"actions": actions}
        if coarse_actions is not None:
            outputs["coarse_actions"] = coarse_actions
        return outputs

    def load_converted_state_dict(self, state_dict: dict[str, Any], *, strict: bool = False) -> None:
        incompatible = self.load_state_dict(state_dict, strict=strict)
        if strict:
            return
        self._last_load_incompatible_keys = incompatible
