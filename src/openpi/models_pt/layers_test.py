import torch

from openpi.models_pt import layers


def test_downsample_extractor_matches_acot_parameter_layout():
    extractor = layers.DownsampleExtractor(
        input_dim=8,
        output_dim=16,
        downsample_dim=4,
        depth=4,
        group_size=2,
        num_heads=2,
    )

    keys = torch.zeros((2, 4, 3, 8))
    values = torch.zeros((2, 4, 3, 8))
    output = extractor(keys, values)

    assert output.shape == (2, 4, 16)
    state_names = set(extractor.state_dict())
    assert "query_params.0" in state_names
    assert "q_proj.0.weight" in state_names
    assert "k_proj.0.weight" in state_names
    assert "v_proj.0.weight" in state_names
    assert "out_proj.0.weight" in state_names
