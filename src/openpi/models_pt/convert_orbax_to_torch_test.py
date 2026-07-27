import numpy as np
import pytest
import torch

from scripts import convert_orbax_to_torch


def test_multihead_in_proj_conversion_concatenates_qkv_weights():
    query = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    key = query + 100
    value = query + 200
    target = torch.zeros((18, 2), dtype=torch.float32)

    converted = convert_orbax_to_torch._adapt_multihead_in_proj((query, key, value), target, "attn.in_proj_weight")

    expected = np.concatenate(
        [
            query.reshape(2, 6).T,
            key.reshape(2, 6).T,
            value.reshape(2, 6).T,
        ],
        axis=0,
    )
    assert converted is not None
    assert torch.equal(converted, torch.from_numpy(expected))


def test_multihead_in_proj_conversion_concatenates_qkv_biases():
    query = np.arange(6, dtype=np.float32).reshape(2, 3)
    key = query + 100
    value = query + 200
    target = torch.zeros((18,), dtype=torch.float32)

    converted = convert_orbax_to_torch._adapt_multihead_in_proj((query, key, value), target, "attn.in_proj_bias")

    expected = np.concatenate([query.reshape(-1), key.reshape(-1), value.reshape(-1)], axis=0)
    assert converted is not None
    assert torch.equal(converted, torch.from_numpy(expected))


def test_torch_from_numpy_accepts_orbax_bfloat16_arrays():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    array = np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16)

    tensor = convert_orbax_to_torch._torch_from_numpy(array)

    assert tensor.dtype == torch.bfloat16
    assert torch.equal(tensor.float(), torch.tensor([1.0, 2.0]))


def test_siglip_scanned_block_candidates_slice_layer_weights():
    candidates = convert_orbax_to_torch._candidates_for("paligemma_img.transformer.blocks.2.mlp.fc1.weight")

    assert any(
        candidate.source == "PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"
        and candidate.transform == "linear_kernel_to_weight"
        and candidate.layer_index == 2
        for candidate in candidates
    )


def test_siglip_scanned_attention_candidates_slice_qkv_sources():
    candidates = convert_orbax_to_torch._multihead_in_proj_candidates(
        "paligemma_img.transformer.blocks.3.attn.in_proj_weight"
    )

    assert candidates is not None
    assert [candidate.source.rsplit(".", 2)[-2] for candidate in candidates] == ["query", "key", "value"]
    assert all(candidate.layer_index == 3 for candidate in candidates)


def test_gemma_final_adarms_candidates_map_dense_parameters():
    candidates = convert_orbax_to_torch._candidates_for("paligemma_llm.final_norms.1.adarms_weight")

    assert any(
        candidate.source == "PaliGemma.llm.final_norm_1.Dense_0.kernel"
        and candidate.transform == "linear_kernel_to_weight"
        for candidate in candidates
    )


def test_gemma_scanned_layer_candidates_slice_expert_weights():
    candidates = convert_orbax_to_torch._candidates_for("paligemma_llm.layers.7.attn.q_einsums.2.lora_a")

    assert any(
        candidate.source == "PaliGemma.llm.layers.attn.q_einsum_2.lora_a"
        and candidate.layer_index == 7
        for candidate in candidates
    )


def test_gemma_scanned_adarms_candidates_map_dense_parameters():
    candidates = convert_orbax_to_torch._candidates_for("paligemma_llm.layers.4.pre_ffw_norms.2.adarms_weight")

    assert any(
        candidate.source == "PaliGemma.llm.layers.pre_ffw_norm_2.Dense_0.kernel"
        and candidate.transform == "linear_kernel_to_weight"
        and candidate.layer_index == 4
        for candidate in candidates
    )
