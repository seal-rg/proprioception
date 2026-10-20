import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FUSED = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<kind>gate_up_proj|down_proj)$")


def split_experts(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        match = FUSED.match(key)
        if match is None:
            out[key] = tensor
            continue
        prefix, kind = match["prefix"], match["kind"]
        for expert, weight in enumerate(tensor):
            if kind == "gate_up_proj":
                gate, up = weight.chunk(2, dim=0)
                out[f"{prefix}.{expert}.w1.weight"] = gate.contiguous()
                out[f"{prefix}.{expert}.w3.weight"] = up.contiguous()
            else:
                out[f"{prefix}.{expert}.w2.weight"] = weight.contiguous()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.checkpoint), Path(args.output)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    shards: list[str] = sorted(set(index["weight_map"].values()))
    for shard in shards:
        converted = split_experts(load_file(str(source / shard)))
        save_file(converted, str(partial / shard), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(converted, shard))
        print(f"{shard}: {len(converted)} tensors")
    (partial / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map}, indent=2)
    )
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, partial / path.name)
    if target.exists():
        shutil.rmtree(target)
    partial.rename(target)
    print(f"wrote {target}: {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
