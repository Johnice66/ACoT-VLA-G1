import torch

from openpi.models_pt.config import TorchSigLIPConfig
from openpi.models_pt.siglip import SigLIPModule


def test_torch_siglip_pool_none_returns_patch_tokens():
    config = TorchSigLIPConfig(
        width=32,
        depth=1,
        mlp_dim=64,
        num_heads=4,
        patch_size=(4, 4),
        pool_type="none",
        posemb="sincos2d",
    )
    model = SigLIPModule(num_classes=64, config=config, dtype="float32")
    image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)

    tokens, aux = model(image)

    assert tokens.shape == (2, 4, 64)
    assert aux["encoded"].shape == (2, 4, config.width)
    assert aux["pre_logits_2d"].shape == (2, 2, 2, config.width)


def test_torch_siglip_learned_posemb_uses_configured_patch_count():
    config = TorchSigLIPConfig(
        width=32,
        depth=1,
        mlp_dim=64,
        num_heads=4,
        patch_size=(4, 4),
        pool_type="none",
        posemb="learn",
        num_patches=4,
    )
    model = SigLIPModule(num_classes=64, config=config, dtype="float32")
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    tokens, _ = model(image)

    assert model.pos_embedding is not None
    assert model.pos_embedding.shape == (1, 4, config.width)
    assert tokens.shape == (1, 4, 64)
