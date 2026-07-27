import torch

from openpi.models_pt import gemma
from openpi.models_pt.config import TorchGemmaConfig, get_gemma_config


def test_torch_gemma_dummy_multi_expert_shapes():
    config = get_gemma_config("dummy")
    model = gemma.GemmaModule((config, config), embed_dtype="float32")

    text_tokens = torch.ones((2, 5), dtype=torch.long)
    text_emb = model.embed(text_tokens)
    action_emb = torch.zeros((2, 3, config.width), dtype=torch.float32)
    positions = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    mask = torch.ones((2, 8, 8), dtype=torch.bool)
    action_cond = torch.zeros((2, config.width), dtype=torch.float32)

    outputs, kv_cache = model([text_emb, action_emb], positions, mask, adarms_cond=[None, action_cond])

    assert outputs[0].shape == (2, 5, config.width)
    assert outputs[1].shape == (2, 3, config.width)
    assert len(kv_cache) == config.depth
    assert kv_cache[0][0].shape == (2, 8, config.num_kv_heads, config.head_dim)
    assert kv_cache[0][1].shape == (2, 8, config.num_kv_heads, config.head_dim)


def test_torch_gemma_lora_parameters_are_registered():
    config = TorchGemmaConfig(
        width=64,
        depth=1,
        mlp_dim=128,
        num_heads=8,
        num_kv_heads=1,
        head_dim=16,
        lora_rank=4,
        lora_alpha=8.0,
    )
    model = gemma.GemmaModule((config,), embed_dtype="float32")

    tokens = torch.ones((1, 4), dtype=torch.long)
    embedded = model.embed(tokens)
    positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
    mask = torch.ones((1, 4, 4), dtype=torch.bool)

    outputs, _ = model([embedded], positions, mask)

    assert outputs[0].shape == (1, 4, config.width)
    assert any("lora" in name for name in model.state_dict())


def test_torch_gemma_rmsnorm_parameters_match_adarms_expert_layout():
    config = get_gemma_config("dummy")
    model = gemma.GemmaModule((config, config, config), embed_dtype="float32")
    keys = set(model.state_dict())

    assert "layers.0.pre_attention_norms.0.scale" in keys
    assert "layers.0.pre_attention_norms.0.adarms_weight" not in keys
    assert "layers.0.pre_attention_norms.1.scale" not in keys
    assert "layers.0.pre_attention_norms.1.adarms_weight" in keys
    assert "layers.0.pre_attention_norms.2.scale" not in keys
    assert "layers.0.pre_attention_norms.2.adarms_weight" in keys
    assert "final_norms.0.scale" in keys
    assert "final_norms.1.adarms_weight" in keys
    assert "final_norms.2.adarms_weight" in keys
