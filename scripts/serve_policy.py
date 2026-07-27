import dataclasses
import logging
import pathlib
import socket
from typing import Any, Literal

import tyro

from openpi.serving import websocket_policy_server


EnvMode = Literal["aloha", "aloha_sim", "droid", "libero", "vlabench", "liberoplus", "g2sim", "g2sim_smoke"]


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = "aloha_sim"

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Execution backend. The JAX backend is the original implementation. The Torch backend is being added for Ascend.
    backend: Literal["jax", "torch"] = "jax"

    # Torch device used when backend=torch.
    device: Literal["cpu", "cuda", "npu"] = "npu"
    # Torch ACoT backbone to instantiate. "full" enables real SigLIP + Gemma/LoRA.
    torch_backbone: Literal["skeleton", "siglip", "gemma", "full"] = "full"
    # Optional converted Torch checkpoint file. If omitted, the checkpoint directory is searched.
    torch_checkpoint_path: pathlib.Path | None = None
    # Torch model dtype used for policy inference.
    torch_dtype: Literal["float32", "bfloat16", "float16"] | None = None
    # Number of flow-matching integration steps for Torch sampling.
    torch_num_steps: int = 10
    # If true, require an exact state_dict match when loading converted Torch weights.
    torch_strict_checkpoint: bool = False
    # If true, allow backend=torch to start without converted Torch weights.
    torch_allow_missing_checkpoint: bool = False
    # If true, skip normalization stats loading for Torch smoke/server validation.
    torch_skip_norm_stats: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[str, Checkpoint] = {
    "aloha": Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1",
    ),
    "aloha_sim": Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    "droid": Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    "libero": Checkpoint(
        config="acot_libero_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/exp_name/40000",
    ),
    "liberoplus": Checkpoint(
        config="acot_libero_plus_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_plus_action_cot_explicit_implicit_co_fusion/exp_name/100000",
    ),
    "vlabench": Checkpoint(
        config="acot_vlabench_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_vlabench_action_cot_explicit_implicit_co_fusion/exp_name/60000",
    ),
    "g2sim": Checkpoint(
        config="acot_icra_simulation_challenge_reasoning_to_action",
        dir="./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/exp_name/30000",
    ),
    "g2sim_smoke": Checkpoint(
        config="acot_icra_simulation_challenge_reasoning_to_action",
        dir="./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/a800_smoke_open_door/0",
    )
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> Any:
    """Create a default policy for the given environment."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_jax_policy(args: Args) -> Any:
    """Create a policy from the given arguments."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def create_torch_policy(args: Args) -> Any:
    """Create a Torch policy for the Ascend NPU migration path."""
    from openpi.policies import torch_policy_config as _torch_policy_config
    from openpi.training import config as _config

    match args.policy:
        case Checkpoint():
            return _torch_policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                device=args.device,
                sample_kwargs={"num_steps": args.torch_num_steps},
                strict_checkpoint=args.torch_strict_checkpoint,
                allow_missing_checkpoint=args.torch_allow_missing_checkpoint,
                checkpoint_path=args.torch_checkpoint_path,
                backbone=args.torch_backbone,
                dtype=args.torch_dtype,
                norm_stats={} if args.torch_skip_norm_stats else None,
            )
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                return _torch_policy_config.create_trained_policy(
                    _config.get_config(checkpoint.config),
                    checkpoint.dir,
                    default_prompt=args.default_prompt,
                    device=args.device,
                    sample_kwargs={"num_steps": args.torch_num_steps},
                    strict_checkpoint=args.torch_strict_checkpoint,
                    allow_missing_checkpoint=args.torch_allow_missing_checkpoint,
                    checkpoint_path=args.torch_checkpoint_path,
                    backbone=args.torch_backbone,
                    dtype=args.torch_dtype,
                    norm_stats={} if args.torch_skip_norm_stats else None,
                )
            raise ValueError(f"Unsupported environment mode: {args.env}")


def create_policy(args: Args) -> Any:
    match args.backend:
        case "jax":
            return create_jax_policy(args)
        case "torch":
            return create_torch_policy(args)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        match args.backend:
            case "jax":
                from openpi.policies import policy as _policy

                policy = _policy.PolicyRecorder(policy, "policy_records")
            case "torch":
                from openpi.policies import torch_policy as _torch_policy

                policy = _torch_policy.TorchPolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
