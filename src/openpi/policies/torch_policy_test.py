import numpy as np

from openpi.models_pt import ACOTVLATorch
from openpi.models_pt.config import acot_icra_simulation_challenge_config
from openpi.policies import torch_policy


def _fake_model_obs() -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
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
        "state": np.zeros(32, dtype=np.float32),
        "tokenized_prompt": np.ones(200, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(200, dtype=bool),
    }


def test_torch_policy_returns_acot_shapes_on_cpu():
    config = acot_icra_simulation_challenge_config()
    model = ACOTVLATorch(config)
    policy = torch_policy.TorchPolicy(model, device="cpu", sample_kwargs={"num_steps": 1})

    outputs = policy.infer(_fake_model_obs())

    assert outputs["actions"].shape == (config.action_horizon, config.action_dim)
    assert outputs["coarse_actions"].shape == (config.coarse_action_horizon, config.action_dim)
    assert outputs["policy_timing"]["infer_ms"] >= 0
    assert policy.metadata["backend"] == "torch"


def test_torch_policy_applies_output_transform():
    config = acot_icra_simulation_challenge_config()
    model = ACOTVLATorch(config)

    def truncate_actions(data: dict) -> dict:
        data["actions"] = data["actions"][:, :21]
        return data

    policy = torch_policy.TorchPolicy(
        model,
        device="cpu",
        output_transforms=[truncate_actions],
        sample_kwargs={"num_steps": 1},
    )

    outputs = policy.infer(_fake_model_obs())

    assert outputs["actions"].shape == (config.action_horizon, 21)
