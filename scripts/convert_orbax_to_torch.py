from __future__ import annotations

from collections.abc import Callable, Iterable
import dataclasses
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import torch
import tyro

from openpi.models_pt import ACOTVLATorch
from openpi.models_pt.config import TorchACOTConfig, TorchSigLIPConfig


@dataclasses.dataclass
class Args:
    """Convert an Orbax/JAX ACoT checkpoint into a Torch state_dict checkpoint."""

    checkpoint_dir: Path
    output: Path | None = None
    report: Path | None = None
    config: Literal["icra", "libero"] = "icra"
    include_gemma: bool = False
    include_siglip: bool = False
    output_format: Literal["safetensors", "pt"] = "safetensors"
    dtype: Literal["source", "float32", "bfloat16", "float16"] = "source"
    strict_shapes: bool = True
    max_report_items: int = 200


@dataclasses.dataclass(frozen=True)
class Candidate:
    source: str
    transform: str = "same"
    layer_index: int | None = None
    split_axis: int | None = None
    split_index: int | None = None


@dataclasses.dataclass(frozen=True)
class SuffixRewrite:
    match: str
    replacement: str
    transform: str = "same"


_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "icra": {
        "coarse_action_horizon": 30,
        "action_horizon": 30,
        "paligemma_variant": "gemma_2b_lora",
    },
    "libero": {
        "coarse_action_horizon": 15,
        "action_horizon": 10,
        "paligemma_variant": "gemma_2b",
        "discrete_state_input": False,
    },
}

_LINEAR_REWRITES = {
    "coarse_time_mlp.0.": "coarse_time_mlp_in.",
    "coarse_time_mlp.2.": "coarse_time_mlp_out.",
    "time_mlp.0.": "time_mlp_in.",
    "time_mlp.2.": "time_mlp_out.",
}

_LINEAR_SUFFIX_REWRITES = (
    SuffixRewrite(".weight", ".kernel", "linear_kernel_to_weight"),
    SuffixRewrite(".bias", ".bias"),
    SuffixRewrite(".scale", ".scale"),
)

_KV_PROJ_SPLITS = (
    (".k_proj.", ".kv_proj.", 0),
    (".v_proj.", ".kv_proj.", 1),
)

_KV_SPLIT_SUFFIX_REWRITES = (
    SuffixRewrite(".weight", ".kernel", "split_linear_kernel_to_weight"),
    SuffixRewrite(".bias", ".bias", "split_same"),
)

_GEMMA_LAYER_REWRITES = (
    (r"attn\.qkv_einsums\.(\d+)\.", "attn.{expert}.", "qkv_einsum"),
    (r"attn\.q_einsums\.(\d+)\.", "attn.{expert}.", "q_einsum"),
    (r"attn\.kv_einsums\.(\d+)\.", "attn.{expert}.", "kv_einsum"),
    (r"attn\.attn_vec_einsums\.(\d+)\.", "attn.{expert}.", "attn_vec_einsum"),
    (r"pre_attention_norms\.(\d+)\.", "{expert}.", "pre_attention_norm"),
    (r"pre_ffw_norms\.(\d+)\.", "{expert}.", "pre_ffw_norm"),
    (r"mlps\.(\d+)\.", "{expert}.", "mlp"),
)

_ADARMS_DENSE_REWRITES = {
    "adarms_weight": ("Dense_0.kernel", "linear_kernel_to_weight"),
    "adarms_bias": ("Dense_0.bias", "same"),
}

_SIGLIP_ENCODER_NORM_REWRITES = {
    "encoder_norm.weight": ("PaliGemma.img.Transformer.encoder_norm.scale", "same"),
    "encoder_norm.bias": ("PaliGemma.img.Transformer.encoder_norm.bias", "same"),
}

_SIGLIP_BLOCK_REWRITES = {
    "norm1.weight": ("LayerNorm_0.scale", "same"),
    "norm1.bias": ("LayerNorm_0.bias", "same"),
    "norm2.weight": ("LayerNorm_1.scale", "same"),
    "norm2.bias": ("LayerNorm_1.bias", "same"),
    "mlp.fc1.weight": ("MlpBlock_0.Dense_0.kernel", "linear_kernel_to_weight"),
    "mlp.fc1.bias": ("MlpBlock_0.Dense_0.bias", "same"),
    "mlp.fc2.weight": ("MlpBlock_0.Dense_1.kernel", "linear_kernel_to_weight"),
    "mlp.fc2.bias": ("MlpBlock_0.Dense_1.bias", "same"),
    "attn.out_proj.weight": ("MultiHeadDotProductAttention_0.out.kernel", "multihead_out_kernel_to_weight"),
    "attn.out_proj.bias": ("MultiHeadDotProductAttention_0.out.bias", "multihead_bias"),
}

_SIGLIP_SUFFIX_REWRITES = (
    SuffixRewrite("embedding.weight", "embedding.kernel", "conv_hwio_to_oihw"),
    SuffixRewrite(".weight", ".kernel", "linear_kernel_to_weight"),
    SuffixRewrite(".bias", ".bias"),
    SuffixRewrite(".scale", ".scale"),
)

_MULTIHEAD_OUT_SUFFIX_REWRITES = (
    SuffixRewrite("out_proj.weight", "out.kernel", "multihead_out_kernel_to_weight"),
    SuffixRewrite("out_proj.bias", "out.bias", "multihead_bias"),
)

_IN_PROJ_SUFFIX_REWRITES = (
    SuffixRewrite("in_proj_weight", "kernel"),
    SuffixRewrite("in_proj_bias", "bias"),
)

_IN_PROJ_SUFFIX_BY_NAME = {rewrite.match: rewrite.replacement for rewrite in _IN_PROJ_SUFFIX_REWRITES}


def _resolve_params_path(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.name == "params":
        return checkpoint_dir
    params_path = checkpoint_dir / "params"
    if params_path.exists():
        return params_path
    return checkpoint_dir


def _flatten_tree(tree: Any, prefix: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        items: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            items.update(_flatten_tree(value, (*prefix, str(key))))
        return items
    return {".".join(prefix): np.asarray(tree)}


def _torch_config(args: Args) -> TorchACOTConfig:
    common = {
        "adopt_explicit_action_reasoner": True,
        "adopt_implicit_action_reasoner": True,
        "downsample_based_implicit_extractor": True,
        "use_real_gemma_backbone": args.include_gemma,
        "use_real_siglip_backbone": args.include_siglip,
        "siglip": TorchSigLIPConfig(posemb="learn", num_patches=256),
    }
    overrides = _CONFIG_OVERRIDES.get(args.config)
    if overrides is None:
        raise ValueError(f"Unsupported conversion config: {args.config}")
    return TorchACOTConfig(**overrides, **common)


def _target_dtype(dtype: str, source: torch.Tensor) -> torch.dtype:
    if dtype == "source":
        return source.dtype
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def _prefix_to_jax(name: str) -> str:
    replacements = {
        "paligemma_llm.": "PaliGemma.llm.",
        "paligemma_img.": "PaliGemma.img.",
    }
    for old, new in replacements.items():
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def _linear_base_candidates(torch_key: str) -> Iterable[Candidate]:
    jax_key = torch_key
    for old, new in _LINEAR_REWRITES.items():
        if old in jax_key:
            jax_key = jax_key.replace(old, new)
            break

    yield from _suffix_candidates(jax_key, _LINEAR_SUFFIX_REWRITES)


def _kv_split_candidates(torch_key: str) -> Iterable[Candidate]:
    split_rule = next((rule for rule in _KV_PROJ_SPLITS if rule[0] in torch_key), None)
    if split_rule is None:
        return

    old, new, split_index = split_rule
    source = torch_key.replace(old, new)
    for candidate in _suffix_candidates(source, _KV_SPLIT_SUFFIX_REWRITES):
        yield dataclasses.replace(candidate, split_axis=-1, split_index=split_index)


def _expert_name(base: str, expert_index: str) -> str:
    return base if expert_index == "0" else f"{base}_{expert_index}"


def _suffix_candidates(source_key: str, rewrites: Iterable[SuffixRewrite]) -> Iterable[Candidate]:
    for rewrite in rewrites:
        if source_key.endswith(rewrite.match):
            source = source_key[: -len(rewrite.match)] + rewrite.replacement
            yield Candidate(source, rewrite.transform)
            return


def _adarms_dense_candidate(source_prefix: str, remainder: str, layer_index: int | None = None) -> Candidate | None:
    rewrite = _ADARMS_DENSE_REWRITES.get(remainder)
    if rewrite is None:
        return None
    source_suffix, transform = rewrite
    return Candidate(f"{source_prefix}{source_suffix}", transform, layer_index=layer_index)


def _gemma_layer_candidates(torch_key: str) -> Iterable[Candidate]:
    match = re.match(r"paligemma_llm\.layers\.(\d+)\.(.*)", torch_key)
    if match is None:
        return
    layer_index = int(match.group(1))
    rest = match.group(2)

    for pattern, template, base in _GEMMA_LAYER_REWRITES:
        submatch = re.match(pattern, rest)
        if submatch is None:
            continue
        expert_index = submatch.group(1)
        prefix = rest[: submatch.end()]
        remainder = rest[len(prefix) :]
        source_base = _expert_name(base, expert_index)
        jax_rest = template.format(expert=source_base) + remainder
        yield Candidate(f"PaliGemma.llm.layers.{jax_rest}", layer_index=layer_index)

        dense_candidate = _adarms_dense_candidate(
            f"PaliGemma.llm.layers.{jax_rest.removesuffix(remainder)}",
            remainder,
            layer_index=layer_index,
        )
        if dense_candidate is not None:
            yield dense_candidate
        break


def _gemma_final_candidates(torch_key: str) -> Iterable[Candidate]:
    match = re.match(r"paligemma_llm\.final_norms\.(\d+)\.(.*)", torch_key)
    if match is None:
        return
    expert_index, remainder = match.groups()
    source_base = _expert_name("final_norm", expert_index)
    yield Candidate(f"PaliGemma.llm.{source_base}.{remainder}")
    dense_candidate = _adarms_dense_candidate(f"PaliGemma.llm.{source_base}.", remainder)
    if dense_candidate is not None:
        yield dense_candidate


def _siglip_layer_candidates(torch_key: str) -> Iterable[Candidate]:
    prefix = "paligemma_img.transformer."
    if not torch_key.startswith(prefix):
        return
    rest = torch_key[len(prefix) :]

    encoder_norm_rewrite = _SIGLIP_ENCODER_NORM_REWRITES.get(rest)
    if encoder_norm_rewrite is not None:
        source, transform = encoder_norm_rewrite
        yield Candidate(source, transform)
        return

    match = re.match(r"blocks\.(\d+)\.(.*)", rest)
    if match is None:
        return
    layer_index = int(match.group(1))
    block_key = match.group(2)
    rewrite = _SIGLIP_BLOCK_REWRITES.get(block_key)
    if rewrite is not None:
        source, transform = rewrite
        yield Candidate(f"PaliGemma.img.Transformer.encoderblock.{source}", transform, layer_index=layer_index)


def _siglip_candidates(torch_key: str) -> Iterable[Candidate]:
    jax_key = _prefix_to_jax(torch_key)
    yield from _suffix_candidates(jax_key, _SIGLIP_SUFFIX_REWRITES)


def _multihead_out_candidates(torch_key: str) -> Iterable[Candidate]:
    if ".attn.out_proj." not in torch_key:
        return
    yield from _suffix_candidates(torch_key, _MULTIHEAD_OUT_SUFFIX_REWRITES)


def _candidates_for(torch_key: str) -> list[Candidate]:
    candidates = [
        Candidate(_prefix_to_jax(torch_key)),
        *_linear_base_candidates(_prefix_to_jax(torch_key)),
        *_kv_split_candidates(_prefix_to_jax(torch_key)),
        *_gemma_layer_candidates(torch_key),
        *_gemma_final_candidates(torch_key),
        *_siglip_layer_candidates(torch_key),
        *_siglip_candidates(torch_key),
        *_multihead_out_candidates(torch_key),
    ]

    deduped: list[Candidate] = []
    seen = set()
    for candidate in candidates:
        key = dataclasses.astuple(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _slice_candidate(array: np.ndarray, candidate: Candidate) -> np.ndarray:
    if candidate.layer_index is not None:
        if array.shape[0] <= candidate.layer_index:
            raise ValueError(f"Layer index {candidate.layer_index} out of bounds for {candidate.source}: {array.shape}")
        array = array[candidate.layer_index]
    if candidate.split_axis is not None:
        if candidate.split_index is None:
            raise ValueError(f"split_axis set without split_index for {candidate.source}")
        parts = np.split(array, 2, axis=candidate.split_axis)
        array = parts[candidate.split_index]
    return array


def _same_array(array: np.ndarray) -> np.ndarray:
    return array


def _linear_kernel_to_weight(array: np.ndarray) -> np.ndarray | None:
    if array.ndim != 2:
        return None
    return array.T


def _conv_hwio_to_oihw(array: np.ndarray) -> np.ndarray | None:
    if array.ndim != 4:
        return None
    return np.transpose(array, (3, 2, 0, 1))


def _multihead_out_kernel_to_weight(array: np.ndarray) -> np.ndarray | None:
    if array.ndim != 3:
        return None
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2]).T


def _multihead_bias(array: np.ndarray) -> np.ndarray:
    if array.ndim > 1:
        return array.reshape(-1)
    return array


_ARRAY_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
    "linear_kernel_to_weight": _linear_kernel_to_weight,
    "split_linear_kernel_to_weight": _linear_kernel_to_weight,
    "conv_hwio_to_oihw": _conv_hwio_to_oihw,
    "multihead_out_kernel_to_weight": _multihead_out_kernel_to_weight,
    "multihead_bias": _multihead_bias,
    "same": _same_array,
    "split_same": _same_array,
}


def _adapt_array(array: np.ndarray, target: torch.Tensor, transform: str) -> torch.Tensor | None:
    transform_fn = _ARRAY_TRANSFORMS.get(transform)
    if transform_fn is None:
        raise ValueError(f"Unknown transform: {transform}")
    array = transform_fn(array)
    if array is None:
        return None

    if tuple(array.shape) != tuple(target.shape):
        if array.ndim == 2 and tuple(array.T.shape) == tuple(target.shape):
            array = array.T
        else:
            return None
    tensor = _torch_from_numpy(array)
    return tensor.to(dtype=target.dtype if not torch.is_floating_point(tensor) else tensor.dtype)


def _multihead_in_proj_candidates(torch_key: str) -> tuple[Candidate, Candidate, Candidate] | None:
    siglip_match = re.match(r"paligemma_img\.transformer\.blocks\.(\d+)\.attn\.(in_proj_weight|in_proj_bias)", torch_key)
    if siglip_match is not None:
        layer_index = int(siglip_match.group(1))
        suffix = _IN_PROJ_SUFFIX_BY_NAME[siglip_match.group(2)]
        return tuple(
            Candidate(
                f"PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.{name}.{suffix}",
                layer_index=layer_index,
            )
            for name in ("query", "key", "value")
        )

    rewrite = next(
        (rewrite for rewrite in _IN_PROJ_SUFFIX_REWRITES if torch_key.endswith(f".attn.{rewrite.match}")),
        None,
    )
    if rewrite is None:
        return None
    base = torch_key[: -len(rewrite.match)]
    suffix = rewrite.replacement
    return tuple(Candidate(f"{base}{name}.{suffix}") for name in ("query", "key", "value"))


def _adapt_multihead_in_proj_weight(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray | None:
    pieces = []
    for array in arrays:
        if array.ndim != 3:
            return None
        flattened = array.reshape(array.shape[0], array.shape[1] * array.shape[2]).T
        pieces.append(flattened)
    return np.concatenate(pieces, axis=0)


def _adapt_multihead_in_proj_bias(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    pieces = []
    for array in arrays:
        if array.ndim > 1:
            array = array.reshape(-1)
        pieces.append(array)
    return np.concatenate(pieces, axis=0)


_MULTIHEAD_IN_PROJ_ADAPTERS: tuple[
    tuple[str, Callable[[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray | None]],
    ...,
] = (
    ("_weight", _adapt_multihead_in_proj_weight),
    ("_bias", _adapt_multihead_in_proj_bias),
)


def _adapt_multihead_in_proj(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: torch.Tensor,
    torch_key: str,
) -> torch.Tensor | None:
    adapter = next((adapter for suffix, adapter in _MULTIHEAD_IN_PROJ_ADAPTERS if torch_key.endswith(suffix)), None)
    if adapter is None:
        return None
    array = adapter(arrays)
    if array is None:
        return None

    if tuple(array.shape) != tuple(target.shape):
        return None
    return _torch_from_numpy(array)


def _torch_from_numpy(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if getattr(array.dtype, "name", "") == "bfloat16":
        return torch.from_numpy(array.astype(np.float32)).to(dtype=torch.bfloat16)
    return torch.from_numpy(array)


def _prefix_counts(names: Iterable[str], depth: int = 2) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        prefix = ".".join(name.split(".")[:depth])
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def _convert_state_dict(
    jax_params: dict[str, np.ndarray],
    target_state: dict[str, torch.Tensor],
    *,
    dtype: str,
    strict_shapes: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    converted = {name: tensor.detach().cpu().clone() for name, tensor in target_state.items()}
    matched = []
    missing = []
    shape_mismatches = []
    used_sources: set[str] = set()

    for torch_key, target_tensor in target_state.items():
        target_cpu = target_tensor.detach().cpu()
        key_shape_mismatches = []
        multihead_candidates = _multihead_in_proj_candidates(torch_key)
        if multihead_candidates is not None:
            source_arrays = []
            source_names = []
            for candidate in multihead_candidates:
                source = jax_params.get(candidate.source)
                if source is None:
                    break
                try:
                    source = _slice_candidate(source, candidate)
                except ValueError as exc:
                    key_shape_mismatches.append({"torch": torch_key, "source": candidate.source, "reason": str(exc)})
                    break
                source_arrays.append(source)
                source_names.append(candidate.source)
            if len(source_arrays) == len(multihead_candidates):
                tensor = _adapt_multihead_in_proj(tuple(source_arrays), target_cpu, torch_key)
                if tensor is None:
                    key_shape_mismatches.append(
                        {
                            "torch": torch_key,
                            "source": "|".join(source_names),
                            "source_shape": [list(source.shape) for source in source_arrays],
                            "target_shape": list(target_cpu.shape),
                            "transform": "multihead_in_proj",
                        }
                    )
                else:
                    if torch.is_floating_point(tensor):
                        tensor = tensor.to(dtype=_target_dtype(dtype, tensor))
                    converted[torch_key] = tensor.contiguous()
                    matched.append(
                        {
                            "torch": torch_key,
                            "source": "|".join(source_names),
                            "transform": "multihead_in_proj",
                            "layer_index": None,
                            "shape": list(tensor.shape),
                        }
                    )
                    used_sources.update(source_names)
                    continue

        for candidate in _candidates_for(torch_key):
            source = jax_params.get(candidate.source)
            if source is None:
                continue
            try:
                source = _slice_candidate(source, candidate)
            except ValueError as exc:
                key_shape_mismatches.append({"torch": torch_key, "source": candidate.source, "reason": str(exc)})
                continue
            tensor = _adapt_array(source, target_cpu, candidate.transform)
            if tensor is None:
                key_shape_mismatches.append(
                    {
                        "torch": torch_key,
                        "source": candidate.source,
                        "source_shape": list(source.shape),
                        "target_shape": list(target_cpu.shape),
                        "transform": candidate.transform,
                    }
                )
                continue
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=_target_dtype(dtype, tensor))
            converted[torch_key] = tensor.contiguous()
            matched.append(
                {
                    "torch": torch_key,
                    "source": candidate.source,
                    "transform": candidate.transform,
                    "layer_index": candidate.layer_index,
                    "shape": list(tensor.shape),
                }
            )
            used_sources.add(candidate.source)
            break
        else:
            missing.append({"torch": torch_key, "shape": list(target_cpu.shape)})
            shape_mismatches.extend(key_shape_mismatches)

    if strict_shapes and shape_mismatches and not matched:
        raise RuntimeError("No parameters could be converted; check the checkpoint/config pair.")

    unused_sources = sorted(set(jax_params) - used_sources)
    report = {
        "summary": {
            "torch_tensors": len(target_state),
            "jax_tensors": len(jax_params),
            "converted_tensors": len(matched),
            "initialized_tensors": len(missing),
            "shape_mismatches": len(shape_mismatches),
            "unused_jax_tensors": len(unused_sources),
        },
        "matched": matched,
        "initialized": missing,
        "initialized_by_prefix": _prefix_counts(item["torch"] for item in missing),
        "shape_mismatches": shape_mismatches,
        "unused_jax_tensors": unused_sources,
        "unused_jax_by_prefix": _prefix_counts(unused_sources),
    }
    return converted, report


def _write_state_dict(state_dict: dict[str, torch.Tensor], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError("safetensors is not installed; use --output-format pt or install safetensors.") from exc
        save_file(state_dict, str(output))
        return
    torch.save({"state_dict": state_dict}, output)


def _resolve_output_path(checkpoint_dir: Path, output: Path | None, output_format: str) -> Path:
    filename = "state_dict.safetensors" if output_format == "safetensors" else "state_dict.pt"
    if output is None:
        return checkpoint_dir / filename
    if output.exists() and output.is_dir():
        return output / filename
    if output.suffix == "":
        return output / filename
    return output


def main(args: Args) -> None:
    from openpi.models import model as _model

    checkpoint_dir = args.checkpoint_dir
    output = _resolve_output_path(checkpoint_dir, args.output, args.output_format)
    report = args.report or output.with_suffix(".report.json")

    params_path = _resolve_params_path(checkpoint_dir)
    jax_params = _flatten_tree(_model.restore_params(params_path, restore_type=np.ndarray))
    model = ACOTVLATorch(_torch_config(args))
    converted, payload = _convert_state_dict(
        jax_params,
        model.state_dict(),
        dtype=args.dtype,
        strict_shapes=args.strict_shapes,
    )

    _write_state_dict(converted, output, args.output_format)
    trimmed_payload = {
        **payload,
        "matched": payload["matched"][: args.max_report_items],
        "initialized": payload["initialized"][: args.max_report_items],
        "shape_mismatches": payload["shape_mismatches"][: args.max_report_items],
        "unused_jax_tensors": payload["unused_jax_tensors"][: args.max_report_items],
        "paths": {
            "checkpoint_dir": str(checkpoint_dir),
            "params_path": str(params_path),
            "output": str(output),
            "report": str(report),
        },
        "config": {
            "name": args.config,
            "include_gemma": args.include_gemma,
            "include_siglip": args.include_siglip,
            "dtype": args.dtype,
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(trimmed_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(trimmed_payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
