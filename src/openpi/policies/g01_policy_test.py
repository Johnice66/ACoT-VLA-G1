import numpy as np

import openpi.transforms as _transforms
from openpi.policies import g01_policy


def _example_images():
    return {
        "top_head": np.zeros((3, 6, 8), dtype=np.uint8),
        "hand_left": np.ones((3, 4, 5), dtype=np.uint8),
        "hand_right": np.full((3, 4, 5), 2, dtype=np.uint8),
    }


def test_g01_acot_inputs_slice_raw_state_and_actions():
    state = np.arange(163, dtype=np.float32)
    actions = np.tile(np.arange(36, dtype=np.float32), (16, 1))

    transform = g01_policy.G01ACOTInputs(
        action_dim=32,
        acot_action_generation=((16, 16), (1, 1)),
    )

    data = transform({"images": _example_images(), "state": state, "actions": actions, "prompt": "task"})

    expected_state = np.concatenate([state[28:35], state[35:42], state[0:1], state[1:2]])
    expected_action = np.concatenate([actions[0, 16:23], actions[0, 23:30], actions[0, 0:1], actions[0, 1:2]])

    assert data["state"].shape == (32,)
    assert np.allclose(data["state"][:16], expected_state)
    assert np.allclose(data["state"][16:], 0)
    assert data["actions"].shape == (16, 32)
    assert data["coarse_actions"].shape == (16, 32)
    assert np.allclose(data["actions"][0, :16], expected_action)
    assert np.allclose(data["actions"][0, 16:], 0)
    assert data["image"]["base_0_rgb"].shape == (6, 8, 3)
    assert data["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert data["prompt"] == "task"


def test_g01_acot_inputs_accept_executable_state():
    state = np.arange(16, dtype=np.float32)

    transform = g01_policy.G01ACOTInputs(action_dim=32)
    data = transform({"images": _example_images(), "state": state})

    assert data["state"].shape == (32,)
    assert np.allclose(data["state"][:16], state)
    assert np.allclose(data["state"][16:], 0)


def test_g01_delta_actions_leave_grippers_absolute():
    state = np.arange(163, dtype=np.float32)
    actions = np.tile(np.arange(36, dtype=np.float32), (16, 1))

    input_transform = g01_policy.G01ACOTInputs(
        action_dim=32,
        acot_action_generation=((16, 16), (1, 1)),
    )
    data = input_transform({"images": _example_images(), "state": state, "actions": actions})
    original_actions = data["actions"].copy()

    delta_transform = _transforms.ACOTDeltaActions(
        _transforms.make_bool_mask(14, -18),
        use_delta_joint_actions=(True, True),
    )
    data = delta_transform(data)

    assert np.allclose(data["actions"][..., :14], original_actions[..., :14] - data["state"][:14])
    assert np.allclose(data["actions"][..., 14:16], original_actions[..., 14:16])

    absolute_transform = _transforms.ACOTAbsoluteActions(
        _transforms.make_bool_mask(14, -18),
        use_delta_joint_actions=(True, True),
    )
    data = absolute_transform(data)

    assert np.allclose(data["actions"], original_actions)


def test_g01_acot_outputs_clip_to_executable_action_dim():
    outputs = {
        "actions": np.ones((16, 32), dtype=np.float32),
        "coarse_actions": np.ones((16, 32), dtype=np.float32) * 2,
        "state": np.zeros((32,), dtype=np.float32),
    }

    data = g01_policy.G01ACOTOutputs()(outputs)

    assert data["actions"].shape == (16, 16)
    assert data["coarse_actions"].shape == (16, 16)
    assert np.all(data["actions"] == 1)
    assert np.all(data["coarse_actions"] == 2)
