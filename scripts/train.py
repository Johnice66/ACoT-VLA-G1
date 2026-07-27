import dataclasses
import functools
import json
import logging
import platform
from typing import Any
import os
import pathlib
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_FIXTURE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _load_fixed_loss_fixture(path: str | os.PathLike[str]):
    fixture_path = pathlib.Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(fixture_path)
    data = np.load(fixture_path, allow_pickle=False)
    observation = _model.Observation.from_dict(
        {
            "image": {key: jnp.asarray(data[f"image__{key}"]) for key in _FIXTURE_IMAGE_KEYS},
            "image_mask": {key: jnp.asarray(data[f"image_mask__{key}"]) for key in _FIXTURE_IMAGE_KEYS},
            "state": jnp.asarray(data["state"].astype(np.float32)),
            **(
                {"tokenized_prompt": jnp.asarray(data["tokenized_prompt"].astype(np.int32))}
                if "tokenized_prompt" in data
                else {}
            ),
            **(
                {"tokenized_prompt_mask": jnp.asarray(data["tokenized_prompt_mask"].astype(np.bool_))}
                if "tokenized_prompt_mask" in data
                else {}
            ),
        }
    )
    return (
        observation,
        jnp.asarray(data["actions"].astype(np.float32)),
        jnp.asarray(data["coarse_actions"].astype(np.float32)),
        jnp.asarray(data["timestep"].astype(np.float32)),
        jnp.asarray(data["action_noise"].astype(np.float32)),
        jnp.asarray(data["coarse_action_noise"].astype(np.float32)),
    )


def _fixed_loss_line(step: int | str, fixed_loss_info: dict[str, Any]) -> str:
    return (
        f"fixed_eval_step={step} "
        f"fixed_total_loss={float(fixed_loss_info['total_loss']):.6f} "
        f"fixed_coarse_loss={float(fixed_loss_info['coarse_loss']):.6f} "
        f"fixed_action_loss={float(fixed_loss_info['action_loss']):.6f} "
        f"fixed_timestep_mean={float(fixed_loss_info['timestep_mean']):.6f}"
    )


def _fixed_train_probe_line(step: int | str, fixed_loss_info: dict[str, Any]) -> str:
    return (
        f"fixed_train_probe_step={step} "
        f"fixed_total_loss={float(fixed_loss_info['total_loss']):.6f} "
        f"fixed_coarse_loss={float(fixed_loss_info['coarse_loss']):.6f} "
        f"fixed_action_loss={float(fixed_loss_info['action_loss']):.6f} "
        f"fixed_timestep_mean={float(fixed_loss_info['timestep_mean']):.6f} "
        f"fixed_grad_norm={float(fixed_loss_info['grad_norm']):.6f}"
    )


def _debug_array_stats(value: Any) -> dict[str, Any]:
    array = np.asarray(jax.device_get(value))
    stats: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return stats

    numeric = array.astype(np.float32)
    flat = numeric.reshape(-1)
    stats.update(
        {
            "sum": float(np.sum(numeric)),
            "mean": float(np.mean(numeric)),
            "std": float(np.std(numeric)),
            "min": float(np.min(numeric)),
            "max": float(np.max(numeric)),
            "first_values": [float(flat[index]) for index in range(min(5, flat.size))],
        }
    )
    return stats


def _train_input_debug_payload(
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.CoarseActions,
        at.Float[at.Array, "*b"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b cah ad"],
    ],
) -> dict[str, Any]:
    observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise = batch

    payload: dict[str, Any] = {
        "processed": {
            "image": {key: _debug_array_stats(value) for key, value in observation.images.items()},
            "image_mask": {key: _debug_array_stats(value) for key, value in observation.image_masks.items()},
            "state": _debug_array_stats(observation.state),
            "actions": _debug_array_stats(actions),
            "coarse_actions": _debug_array_stats(coarse_actions),
        },
        "randoms": {
            "timestep": _debug_array_stats(timestep),
            "action_noise": _debug_array_stats(action_noise),
            "coarse_action_noise": _debug_array_stats(coarse_action_noise),
        },
    }
    if observation.tokenized_prompt is not None:
        payload["processed"]["tokenized_prompt"] = _debug_array_stats(observation.tokenized_prompt)
    if observation.tokenized_prompt_mask is not None:
        payload["processed"]["tokenized_prompt_mask"] = _debug_array_stats(observation.tokenized_prompt_mask)
    return payload


def _deterministic_training_randoms(
    actions: at.Array,
    coarse_actions: at.Array,
    *,
    seed: int,
    step: int,
) -> tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b ah ad"], at.Float[at.Array, "*b cah ad"]]:
    rng = np.random.default_rng(seed + step)
    action_noise = rng.normal(size=tuple(actions.shape)).astype(np.float32)
    coarse_action_noise = rng.normal(size=tuple(coarse_actions.shape)).astype(np.float32)
    timestep = (rng.beta(1.5, 1.0, size=(actions.shape[0],)).astype(np.float32) * 0.999) + 0.001
    return (
        jnp.asarray(timestep),
        jnp.asarray(action_noise),
        jnp.asarray(coarse_action_noise),
    )


def _add_deterministic_randoms_to_acot_batch(
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
    *,
    seed: int,
    step: int,
):
    observation, actions, coarse_actions = batch
    timestep, action_noise, coarse_action_noise = _deterministic_training_randoms(
        actions,
        coarse_actions,
        seed=seed,
        step=step,
    )
    return observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return {"eval_total_loss": jnp.mean(chunked_loss)}

@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.CoarseActions,
        at.Float[at.Array, "*b"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b cah ad"],
    ],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        timestep: at.Float[at.Array, "*b"],
        action_noise: at.Float[at.Array, "*b ah ad"],
        coarse_action_noise: at.Float[at.Array, "*b cah ad"],
    ):
        losses = model.compute_loss_with_randoms(
            observation,
            actions,
            coarse_actions,
            timestep=timestep,
            expert_action_noise=action_noise,
            coarse_action_noise=coarse_action_noise,
            train=False,
        )
        return losses["total_loss"]

    observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model,
        observation,
        actions,
        coarse_actions,
        timestep,
        action_noise,
        coarse_action_noise,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def acot_eval_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.CoarseActions,
        at.Float[at.Array, "*b"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b cah ad"],
    ],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise = batch
    losses = model.compute_loss_with_randoms(
        observation,
        actions,
        coarse_actions,
        timestep=timestep,
        expert_action_noise=action_noise,
        coarse_action_noise=coarse_action_noise,
        train=False,
    )
    return {"eval_total_loss": losses["total_loss"]}


@at.typecheck
def acot_fixed_loss_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.CoarseActions,
        at.Float[at.Array, "*b"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b cah ad"],
    ],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise = batch
    losses = model.compute_loss_with_randoms(
        observation,
        actions,
        coarse_actions,
        timestep=timestep,
        expert_action_noise=action_noise,
        coarse_action_noise=coarse_action_noise,
        train=False,
    )
    return {key: losses[key] for key in ("total_loss", "coarse_loss", "action_loss", "timestep_mean")}


@at.typecheck
def acot_fixed_train_probe_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.CoarseActions,
        at.Float[at.Array, "*b"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b cah ad"],
    ],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        timestep: at.Float[at.Array, "*b"],
        action_noise: at.Float[at.Array, "*b ah ad"],
        coarse_action_noise: at.Float[at.Array, "*b cah ad"],
    ):
        losses = model.compute_loss_with_randoms(
            observation,
            actions,
            coarse_actions,
            timestep=timestep,
            expert_action_noise=action_noise,
            coarse_action_noise=coarse_action_noise,
            train=False,
        )
        return losses["total_loss"], {
            key: losses[key] for key in ("total_loss", "coarse_loss", "action_loss", "timestep_mean")
        }

    observation, actions, coarse_actions, timestep, action_noise, coarse_action_noise = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (_, losses), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        observation,
        actions,
        coarse_actions,
        timestep,
        action_noise,
        coarse_action_noise,
    )
    return {
        **losses,
        "grad_norm": optax.global_norm(grads),
    }


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    if config.fixed_loss_interval < 0:
        raise ValueError(f"fixed_loss_interval must be non-negative, got {config.fixed_loss_interval}.")
    if config.fixed_loss_fixture is not None:
        if config.fixed_loss_interval <= 0:
            raise ValueError("--fixed-loss-interval must be positive when --fixed-loss-fixture is set.")
        if config.model.model_type not in (_model.ModelType.ACOT_VLA_PI05, _model.ModelType.ACOT_VLA_PI0):
            raise ValueError("--fixed-loss-fixture is only supported for ACoT models.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    fixed_eval_batches = []
    if config.eval_interval > 0 and config.eval_batches > 0:
        eval_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.eval_batches,
        )
        fixed_eval_batches = list(iter(eval_data_loader))
        logging.info(f"Initialized fixed eval loader with {len(fixed_eval_batches)} batches")
    fixed_loss_batch = None
    if config.fixed_loss_fixture is not None:
        fixed_loss_batch = _load_fixed_loss_fixture(config.fixed_loss_fixture)
        logging.info(
            f"Loaded fixed loss fixture: path={config.fixed_loss_fixture} interval={config.fixed_loss_interval}"
        )

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(0,),
        )
        peval_step = jax.jit(
            functools.partial(acot_eval_step, config),
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        pfixed_loss_step = jax.jit(
            functools.partial(acot_fixed_loss_step, config),
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        pfixed_train_probe_step = jax.jit(
            functools.partial(acot_fixed_train_probe_step, config),
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        peval_step = jax.jit(
            functools.partial(eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        pfixed_loss_step = None
        pfixed_train_probe_step = None

    if fixed_loss_batch is not None:
        assert pfixed_loss_step is not None
        with sharding.set_mesh(mesh):
            initial_fixed_loss_info = pfixed_loss_step(train_state, fixed_loss_batch)
        initial_fixed_loss_info = jax.device_get(jax.tree.map(jnp.mean, initial_fixed_loss_info))
        initial_fixed_loss_line = _fixed_loss_line("init", initial_fixed_loss_info)
        print(initial_fixed_loss_line, flush=True)
        logging.info(initial_fixed_loss_line)
        wandb.log({f"initial_fixed_{key}": value for key, value in initial_fixed_loss_info.items()}, step=0)
        assert pfixed_train_probe_step is not None
        with sharding.set_mesh(mesh):
            initial_fixed_train_probe_info = pfixed_train_probe_step(train_state, fixed_loss_batch)
        initial_fixed_train_probe_info = jax.device_get(jax.tree.map(jnp.mean, initial_fixed_train_probe_info))
        initial_fixed_train_probe_line = _fixed_train_probe_line("init", initial_fixed_train_probe_info)
        print(initial_fixed_train_probe_line, flush=True)
        logging.info(initial_fixed_train_probe_line)
        wandb.log(
            {f"initial_fixed_train_probe_{key}": value for key, value in initial_fixed_train_probe_info.items()},
            step=0,
        )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
            train_batch = _add_deterministic_randoms_to_acot_batch(batch, seed=config.seed, step=step)
        else:
            train_batch = batch
        if step == 0 and (
            config.model.model_type == _model.ModelType.ACOT_VLA_PI05
            or config.model.model_type == _model.ModelType.ACOT_VLA_PI0
        ):
            train_step0_input_debug = _train_input_debug_payload(train_batch)
            train_input_line = "train_input_step=0 " + json.dumps(
                train_step0_input_debug,
                sort_keys=True,
                separators=(",", ":"),
            )
            print(train_input_line, flush=True)
            logging.info(train_input_line)
        with sharding.set_mesh(mesh):
            if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
                train_state, info = ptrain_step(train_state, train_batch)
            else:
                train_state, info = ptrain_step(train_rng, train_state, train_batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            print(f"Step {step}: {info_str}", flush=True)
            logging.info(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if fixed_eval_batches and (
            step % config.eval_interval == 0 or step == config.num_train_steps - 1
        ):
            eval_infos = []
            for eval_batch_idx, eval_batch in enumerate(fixed_eval_batches):
                if (
                    config.model.model_type == _model.ModelType.ACOT_VLA_PI05
                    or config.model.model_type == _model.ModelType.ACOT_VLA_PI0
                ):
                    eval_batch = _add_deterministic_randoms_to_acot_batch(
                        eval_batch,
                        seed=config.seed,
                        step=eval_batch_idx,
                    )
                    with sharding.set_mesh(mesh):
                        eval_infos.append(peval_step(train_state, eval_batch))
                    continue
                eval_rng = jax.random.fold_in(train_rng, eval_batch_idx)
                with sharding.set_mesh(mesh):
                    eval_infos.append(peval_step(eval_rng, train_state, eval_batch))
            stacked_eval_infos = common_utils.stack_forest(eval_infos)
            reduced_eval_info = jax.device_get(jax.tree.map(jnp.mean, stacked_eval_infos))
            eval_total_loss = float(reduced_eval_info["eval_total_loss"])
            eval_line = f"eval_step={step} eval_total_loss={eval_total_loss:.6f}"
            print(eval_line, flush=True)
            logging.info(eval_line)
            wandb.log(reduced_eval_info, step=step)
        if fixed_loss_batch is not None and (
            step % config.fixed_loss_interval == 0 or step == config.num_train_steps - 1
        ):
            assert pfixed_loss_step is not None
            with sharding.set_mesh(mesh):
                fixed_loss_info = pfixed_loss_step(train_state, fixed_loss_batch)
            fixed_loss_info = jax.device_get(jax.tree.map(jnp.mean, fixed_loss_info))
            fixed_loss_line = _fixed_loss_line(step, fixed_loss_info)
            print(fixed_loss_line, flush=True)
            logging.info(fixed_loss_line)
            wandb.log({f"fixed_{key}": value for key, value in fixed_loss_info.items()}, step=step)
        batch = next(data_iter)

        should_save_interval = config.save_interval > 0 and step % config.save_interval == 0 and step > start_step
        should_save_final = config.save_final_checkpoint and step == config.num_train_steps - 1
        if should_save_interval or should_save_final:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
