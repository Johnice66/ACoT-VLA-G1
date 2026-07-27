import numpy as np
import torch

from openpi.models_pt import ACOTVLATorch
from openpi.models_pt.config import TorchACOTConfig, TorchSigLIPConfig
from openpi.models_pt.types import TorchObservation


def _dummy_observation(action_dim: int) -> TorchObservation:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return TorchObservation.from_dict(
        {
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.zeros(action_dim, dtype=np.float32),
            "tokenized_prompt": np.ones(4, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(4, dtype=bool),
        },
        device=torch.device("cpu"),
    )


def test_acot_torch_can_route_prefix_through_gemma_backbone():
    config = TorchACOTConfig(
        dtype="float32",
        paligemma_variant="dummy",
        coarse_action_expert_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        coarse_action_horizon=2,
        action_horizon=2,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
        use_real_gemma_backbone=True,
    )
    model = ACOTVLATorch(config)
    observation = _dummy_observation(config.action_dim)

    prefix_tokens, prefix_mask = model.embed_prefix(observation)
    outputs = model.sample_actions(observation, num_steps=1)

    assert model.model_status == "torch_acot_gemma_prefix"
    assert prefix_tokens.shape == (1, 7, config.paligemma.width)
    assert prefix_mask.shape == (1, 7)
    assert outputs["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert any(name.startswith("paligemma_llm.") for name in model.state_dict())


def test_acot_torch_can_route_images_through_siglip_and_gemma_backbones():
    config = TorchACOTConfig(
        dtype="float32",
        paligemma_variant="dummy",
        coarse_action_expert_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        coarse_action_horizon=2,
        action_horizon=2,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
        use_real_gemma_backbone=True,
        use_real_siglip_backbone=True,
        siglip=TorchSigLIPConfig(
            width=32,
            depth=1,
            mlp_dim=64,
            num_heads=4,
            patch_size=(4, 4),
            pool_type="none",
            posemb="sincos2d",
        ),
    )
    model = ACOTVLATorch(config)
    observation = _dummy_observation(config.action_dim)

    prefix_tokens, prefix_mask = model.embed_prefix(observation)
    outputs = model.sample_actions(observation, num_steps=1)

    assert model.model_status == "torch_acot_gemma_siglip_prefix"
    assert prefix_tokens.shape == (1, 16, config.paligemma.width)
    assert prefix_mask.shape == (1, 16)
    assert outputs["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert any(name.startswith("paligemma_img.") for name in model.state_dict())


def test_acot_torch_can_route_images_through_siglip_without_gemma_backbone():
    config = TorchACOTConfig(
        dtype="float32",
        paligemma_variant="dummy",
        coarse_action_expert_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        coarse_action_horizon=2,
        action_horizon=2,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
        use_real_siglip_backbone=True,
        siglip=TorchSigLIPConfig(
            width=32,
            depth=1,
            mlp_dim=64,
            num_heads=4,
            patch_size=(4, 4),
            pool_type="none",
            posemb="learn",
            num_patches=4,
        ),
    )
    model = ACOTVLATorch(config)
    observation = _dummy_observation(config.action_dim)

    prefix_tokens, prefix_mask = model.embed_prefix(observation)
    outputs = model.sample_actions(observation, num_steps=1)

    assert model.model_status == "torch_acot_siglip_prefix"
    assert prefix_tokens.shape == (1, 13, config.action_expert.width)
    assert prefix_mask.shape == (1, 13)
    assert outputs["actions"].shape == (1, config.action_horizon, config.action_dim)
    assert any(name.startswith("paligemma_img.") for name in model.state_dict())
    assert not any(name.startswith("paligemma_llm.") for name in model.state_dict())


def test_torch_acot_config_full_backbone_uses_checkpoint_siglip_layout():
    config = TorchACOTConfig().with_backbone("full")

    assert config.use_real_gemma_backbone
    assert config.use_real_siglip_backbone
    assert config.siglip.posemb == "learn"
    assert config.siglip.num_patches == 256


def test_acot_torch_training_loss_is_finite_and_backward_ready():
    config = TorchACOTConfig(
        dtype="float32",
        action_dim=8,
        coarse_action_horizon=2,
        action_horizon=2,
        adopt_explicit_action_reasoner=True,
    )
    model = ACOTVLATorch(config)
    observation = _dummy_observation(config.action_dim)
    actions = torch.linspace(
        -0.2,
        0.2,
        steps=config.action_horizon * config.action_dim,
        dtype=torch.float32,
    ).reshape(1, config.action_horizon, config.action_dim)
    coarse_actions = torch.linspace(
        0.1,
        -0.1,
        steps=config.coarse_action_horizon * config.action_dim,
        dtype=torch.float32,
    ).reshape(1, config.coarse_action_horizon, config.action_dim)
    action_noise = torch.full_like(actions, 0.25)
    coarse_action_noise = torch.full_like(coarse_actions, -0.25)
    timestep = torch.tensor([0.4], dtype=torch.float32)

    losses = model.compute_training_loss(
        observation,
        actions,
        coarse_actions,
        timestep=timestep,
        action_noise=action_noise,
        coarse_action_noise=coarse_action_noise,
    )
    losses["total_loss"].backward()

    assert losses["total_loss"].shape == ()
    assert losses["coarse_loss"].shape == ()
    assert losses["action_loss"].shape == ()
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["coarse_loss"])
    assert torch.isfinite(losses["action_loss"])
    assert model.action_out_proj.weight.grad is not None
    assert torch.isfinite(model.action_out_proj.weight.grad).all()
