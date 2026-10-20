import ast
import functools
import glob
import hashlib
import json
import os
import struct
from collections.abc import Iterable
from pathlib import Path

from evals.config import REPO, EvalConfig, ModelSpec

PACKAGES = ("evals", "proprioception")
CHECKPOINT_FILES = ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer_config.json")
DATA_GLOBS = ("evals/data/*.jsonl", "evals/sad/model_facts/*.yaml")
CONFIG_FIELDS = (
    "family",
    "domain",
    "types",
    "seed",
    "max_tokens",
    "max_model_len",
    "system_prompt",
    "answer_mode",
    "carrier",
    "moe_backend",
    "attention_backend",
    "reasoning_parser",
    "judge_model",
)


@functools.cache
def checkpoint(path: str, samples_per_shard: int = 8, sample_bytes: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {path}")
    for shard in shards:
        size = os.path.getsize(shard)
        with open(shard, "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            digest.update(handle.read(header_len))
            start = 8 + header_len
            span = max(size - start - sample_bytes, 0)
            for k in range(samples_per_shard):
                handle.seek(start + span * k // max(samples_per_shard - 1, 1))
                digest.update(handle.read(sample_bytes))
    for name in CHECKPOINT_FILES:
        file = Path(path) / name
        if file.is_file():
            digest.update(name.encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def _module_file(name: str) -> Path | None:
    base = REPO.joinpath(*name.split("."))
    for file in (base.with_suffix(".py"), base / "__init__.py"):
        if file.is_file():
            return file
    return None


def _imported(path: Path, package: str) -> Iterable[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names)


@functools.cache
def code_files(module: str) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    todo = [module]
    while todo:
        name = todo.pop()
        if name in seen or name.split(".")[0] not in PACKAGES:
            continue
        file = _module_file(name)
        if file is None:
            continue
        seen[name] = file
        package = name if file.name == "__init__.py" else name.rsplit(".", 1)[0]
        todo.extend(_imported(file, package))
        parent = name.rsplit(".", 1)[0]
        if parent != name:
            todo.append(parent)
    return tuple(sorted(set(seen.values())))


def suite_key(config: EvalConfig, spec: ModelSpec, suite: str, args: dict, extra: dict | None = None) -> str:
    digest = hashlib.sha256()
    for file in code_files(f"evals.suites.{suite}"):
        digest.update(str(file.relative_to(REPO)).encode())
        digest.update(file.read_bytes())
    for pattern in DATA_GLOBS:
        for file in sorted(REPO.glob(pattern)):
            digest.update(str(file.relative_to(REPO)).encode())
            digest.update(file.read_bytes())
    settings = {
        "args": args,
        "config": {name: getattr(config, name) for name in CONFIG_FIELDS},
        "model": {"role": spec.role, "domain": spec.domain, "types": spec.types},
        "extra": extra or {},
    }
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def artefact_key(checkpoint_path: str, code: tuple[str, ...], settings: dict) -> str:
    digest = hashlib.sha256(checkpoint(checkpoint_path).encode())
    for name in code:
        digest.update(name.encode())
        digest.update((REPO / name).read_bytes())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]
