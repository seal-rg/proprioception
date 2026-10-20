import collections
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from proprioception import distributed, primerl_backend

CONFIG_PATTERNS = ("*.json", "*.jinja", "*.txt")
SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


def cast_to_reference(state_dict: dict, weight_map: dict[str, Path]) -> dict:
    shards: dict[Path, list[str]] = collections.defaultdict(list)
    for key in state_dict:
        shards[weight_map[key]].append(key)
    cast = dict(state_dict)
    for shard, keys in shards.items():
        with safe_open(shard, framework="pt") as handle:
            for key in keys:
                dtype = SAFETENSORS_DTYPES.get(handle.get_slice(key).get_dtype())
                if dtype is not None and cast[key].dtype != dtype:
                    cast[key] = cast[key].to(dtype)
    return cast


def check_against_reference(state_dict: dict, weight_map: dict[str, Path]) -> None:
    extra = sorted(set(state_dict) - set(weight_map))
    if extra:
        raise ValueError(f"keys not in the reference: {extra[:5]}")
    shards: dict[Path, list[str]] = collections.defaultdict(list)
    for key in state_dict:
        shards[weight_map[key]].append(key)
    for shard, keys in shards.items():
        with safe_open(shard, framework="pt") as handle:
            for key in keys:
                want, got = handle.get_slice(key).get_shape(), list(state_dict[key].shape)
                if want != got:
                    raise ValueError(f"shape mismatch for {key}: reference {want}, export {got}")


def reference_dtype(reference: str | Path) -> torch.dtype | None:
    config = json.loads((Path(reference) / "config.json").read_text())
    name = config.get("dtype") or config.get("torch_dtype")
    return getattr(torch, name) if isinstance(name, str) and hasattr(torch, name) else None


def write_shards(state_dict: dict, directory: Path) -> None:
    rank, world = distributed.rank_and_world()
    shard = "model.safetensors" if world == 1 else f"model-{rank + 1:05d}-of-{world:05d}.safetensors"
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    save_file(state_dict, directory / shard, metadata={"format": "pt"})
    entries = distributed.all_gather_list(
        [(key, shard, value.numel() * value.element_size()) for key, value in state_dict.items()]
    )
    if distributed.is_main():
        index = {
            "metadata": {"total_size": sum(size for _, _, size in entries)},
            "weight_map": {key: name for key, name, _ in sorted(entries)},
        }
        (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def write_servable(model, reference: str | Path, output_dir: str | Path) -> Path:
    reference = Path(reference)
    out = Path(output_dir)
    partial = out.with_name(out.name + ".partial")
    rank, world = distributed.rank_and_world()
    if distributed.is_main():
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        if primerl_backend.is_prime_class(model):
            (partial / "prime").mkdir()
    distributed.barrier()
    owned = distributed.owned_full_state_dict(model)
    if primerl_backend.is_prime_class(model):
        dtype = reference_dtype(reference)
        write_shards({k: v if dtype is None else v.to(dtype) for k, v in owned.items()}, partial / "prime")
    state_dict = primerl_backend.to_hub_layout(model, reference, owned)
    weight_map = primerl_backend.hub_weight_map(reference)
    produced = distributed.all_gather_list(sorted(state_dict))
    if len(produced) != len(set(produced)):
        raise ValueError("two ranks export the same parameter")
    carried = sorted(set(weight_map) - set(produced))
    unexpected = sorted(set(carried) - primerl_backend.untrained_keys(reference))
    if unexpected:
        raise ValueError(f"export is missing trained weights: {unexpected[:5]}")
    if distributed.is_main():
        for key in carried:
            with safe_open(weight_map[key], framework="pt") as handle:
                state_dict[key] = handle.get_tensor(key)
    check_against_reference(state_dict, weight_map)
    write_shards(cast_to_reference(state_dict, weight_map), partial)
    if distributed.is_main():
        for pattern in CONFIG_PATTERNS:
            for config_file in reference.glob(pattern):
                if config_file.name != "model.safetensors.index.json":
                    shutil.copy2(config_file, partial / config_file.name)
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    return out


def servable_exists(output: str) -> bool:
    return bool(output) and (Path(output) / "hf" / "model.safetensors.index.json").exists()
