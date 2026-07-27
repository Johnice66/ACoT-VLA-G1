from __future__ import annotations

import dataclasses
import functools
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tyro

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


DEFAULT_PI05_PARAMS_PATH = "gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params"


@dataclasses.dataclass(frozen=True)
class Args:
    """Export a step-0 ACoT Orbax params checkpoint initialized from pi0.5."""

    config_name: str
    output_dir: Path
    pi05_params_path: str | None = None
    seed: int | None = None
    fsdp_devices: int | None = None
    overwrite: bool = False


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _init_params(config: _config.TrainConfig, rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> nnx.State:
    model = config.model.create(rng)
    if partial_params is not None:
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(partial_params)
        model = nnx.merge(graphdef, state)

    params = nnx.state(model)
    return nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))


def _prepare_config(args: Args) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    params_path = args.pi05_params_path or os.getenv("ACOT_PI05_PARAMS_PATH") or DEFAULT_PI05_PARAMS_PATH
    updates: dict[str, Any] = {
        "weight_loader": _weight_loaders.ACOTCheckpointWeightLoader(params_path),
        "wandb_enabled": False,
    }
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.fsdp_devices is not None:
        updates["fsdp_devices"] = args.fsdp_devices
    return dataclasses.replace(config, **updates)


def _save_params(params: nnx.State, output_dir: Path, *, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    params_dir = output_dir / "params"
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(epath.Path(params_dir), {"params": params})


def main(args: Args) -> None:
    _init_logging()
    config = _prepare_config(args)

    if config.fsdp_devices > jax.device_count():
        raise ValueError(f"fsdp_devices={config.fsdp_devices} exceeds jax.device_count()={jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    init_fn = functools.partial(_init_params, config)
    params_shape = jax.eval_shape(init_fn, init_rng)
    params_sharding = sharding.fsdp_sharding(params_shape, mesh, log=True)
    partial_params = _load_weights_and_validate(config.weight_loader, params_shape.to_pure_dict())

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params = jax.jit(
        init_fn,
        donate_argnums=(1,),
        in_shardings=(replicated_sharding, replicated_sharding),
        out_shardings=params_sharding,
    )(init_rng, partial_params)
    jax.block_until_ready(params)

    logging.info("Initialized params:\n%s", training_utils.array_tree_to_info(params))
    logging.info("Total parameters: %s", f"{training_utils.count_parameters(params):,}")
    _save_params(params, args.output_dir, overwrite=args.overwrite)
    logging.info("Wrote ACoT init checkpoint to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
