import argparse
import collections
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open

from proprioception.primerl_backend import hub_weight_map

INDEX = "model.safetensors.index.json"


def tensor_layouts(weight_map: dict[str, Path]) -> dict[str, tuple[list[int], str]]:
    shards: dict[Path, list[str]] = collections.defaultdict(list)
    for key, shard in weight_map.items():
        shards[shard].append(key)
    layouts = {}
    for shard, keys in shards.items():
        with safe_open(shard, framework="pt") as handle:
            for key in keys:
                tensor = handle.get_slice(key)
                layouts[key] = (list(tensor.get_shape()), tensor.get_dtype())
    return layouts


def check_snapshot(weights: Path, reference: Path) -> None:
    snapshot_map, reference_map = hub_weight_map(weights), hub_weight_map(reference)
    missing, extra = sorted(set(reference_map) - set(snapshot_map)), sorted(set(snapshot_map) - set(reference_map))
    if missing or extra:
        raise ValueError(f"snapshot lacks {missing[:5]} and adds {extra[:5]} against the reference")
    snapshot, expected = tensor_layouts(snapshot_map), tensor_layouts(reference_map)
    differing = [
        f"{key}: {snapshot[key]} vs reference {expected[key]}"
        for key in sorted(snapshot)
        if snapshot[key] != expected[key]
    ]
    if differing:
        raise ValueError(f"{len(differing)} tensors differ from the reference in shape or dtype: {differing[:5]}")
    config = weights / "config.json"
    if config.exists():
        want = json.loads((reference / "config.json").read_text()).get("model_type")
        got = json.loads(config.read_text()).get("model_type")
        if got != want:
            raise ValueError(f"snapshot is a {got!r} checkpoint, the reference a {want!r} one")


def reference_files(reference: Path) -> list[Path]:
    files = [p for p in sorted(reference.iterdir()) if p.is_file() and p.suffix != ".safetensors" and p.name != INDEX]
    names = {p.name for p in files}
    required = {"config.json", "tokenizer_config.json"} - names
    if required:
        raise ValueError(f"reference {reference} has no {sorted(required)}")
    if not names & {"tokenizer.json", "tokenizer.model"}:
        raise ValueError(f"reference {reference} has no tokenizer.json or tokenizer.model")
    if "chat_template.jinja" not in names and "chat_template" not in json.loads(
        (reference / "tokenizer_config.json").read_text()
    ):
        raise ValueError(f"reference {reference} has no chat template")
    return files


def assemble(weights: Path, reference: Path, out: Path) -> Path:
    check_snapshot(weights, reference)
    files = reference_files(reference)
    index = json.loads((weights / INDEX).read_text())
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for shard in sorted(set(index["weight_map"].values())):
        shutil.copy2(weights / shard, partial / shard)
    shutil.copy2(weights / INDEX, partial / INDEX)
    for path in files:
        shutil.copy2(path, partial / path.name)
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights, reference, out = (Path(os.path.expandvars(p)) for p in (args.weights, args.reference, args.output))
    print(f"wrote {assemble(weights, reference, out)}")


if __name__ == "__main__":
    main()
