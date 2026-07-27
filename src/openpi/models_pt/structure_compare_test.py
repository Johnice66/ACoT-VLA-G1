from scripts import compare_model_structures


def test_compare_model_structures_classifies_basic_conversion_cases():
    jax_parameters = [
        {"name": "PaliGemma.llm.embedder.input_embedding", "shape": [8, 4], "dtype": "float32", "numel": 32},
        {"name": "PaliGemma.img.embedding.weight", "shape": [4, 4, 3, 16], "dtype": "float32", "numel": 768},
        {"name": "PaliGemma.llm.proj.weight", "shape": [32, 16], "dtype": "float32", "numel": 512},
        {"name": "only_jax.bias", "shape": [3], "dtype": "float32", "numel": 3},
    ]
    torch_parameters = [
        {"name": "paligemma_llm.embedder.input_embedding", "shape": [8, 4], "dtype": "float32", "numel": 32},
        {"name": "paligemma_img.embedding.weight", "shape": [16, 3, 4, 4], "dtype": "float32", "numel": 768},
        {"name": "paligemma_llm.proj.weight", "shape": [16, 32], "dtype": "float32", "numel": 512},
        {"name": "only_torch.bias", "shape": [5], "dtype": "float32", "numel": 5},
    ]

    result = compare_model_structures.compare_structures(jax_parameters, torch_parameters)

    assert result["summary"]["exact_matches"] == 1
    assert result["summary"]["transpose_candidates"] == 1
    assert result["summary"]["conv_candidates"] == 1
    assert result["summary"]["unmatched_jax"] >= 1
    assert result["summary"]["unmatched_torch"] >= 1
    assert result["exact_matches"][0]["jax"] == "PaliGemma.llm.embedder.input_embedding"
