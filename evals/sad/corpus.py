import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path

import jinja2
import yaml

from evals.config import CACHE

CHECKOUT = CACHE / "sad"
ROOT = CHECKOUT / "sad"


@dataclasses.dataclass
class MCQ:
    id: str
    body: str
    choices_right: list[str]
    choices_wrong: list[str]
    choices_order: list[str] | None = None
    splits: dict = dataclasses.field(default_factory=dict)


def root() -> Path:
    if not (ROOT / "vars.py").is_file():
        raise RuntimeError(f"SAD checkout missing at {CHECKOUT}; run evals/runners/sad/setup.sh")
    return ROOT


def path(*parts: str) -> Path:
    return root().joinpath(*parts)


def read_yaml(*parts: str):
    return yaml.safe_load(path(*parts).read_text())


def read_json(*parts: str):
    return json.loads(path(*parts).read_text())


def read_jsonl(*parts: str) -> list[dict]:
    return [json.loads(line) for line in path(*parts).read_text().splitlines() if line.strip()]


def read_csv(*parts: str) -> list[dict]:
    import csv

    with path(*parts).open(newline="") as handle:
        return [{key: (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def data_module(dotted: str) -> types.ModuleType:
    root()
    if str(CHECKOUT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT))
    return importlib.import_module(dotted)


def render(template: str, **variables) -> str:
    return jinja2.Template(template).render(**variables)


def batch(task_dir: str, batch_id: str) -> list[MCQ]:
    raw = read_yaml(task_dir, "structs", "batch", f"{batch_id}.yaml")
    return [_mcq(sample, f"{batch_id}/{i}") for i, sample in enumerate(raw["samples"])]


def pile(task_dir: str, pile_id: str) -> list[MCQ]:
    raw = read_yaml(task_dir, "structs", "pile", f"{pile_id}.yaml")
    return [sample for batch_id in raw["batches"] for sample in batch(task_dir, batch_id)]


def mcq_jsonl(task_dir: str, *parts: str, splits: dict | None = None) -> list[MCQ]:
    name = parts[-1].removesuffix(".jsonl")
    return [_mcq(row, f"{name}/{i}", extra_splits=splits) for i, row in enumerate(read_jsonl(task_dir, *parts))]


def mcq_csv(task_dir: str, *parts: str, splits: dict | None = None) -> list[MCQ]:
    name = parts[-1].removesuffix(".csv")
    return [
        MCQ(
            id=f"{name}/{i}",
            body=row["question"],
            choices_right=[row["correct"]],
            choices_wrong=[row["wrong"]],
            splits=dict(splits or {}),
        )
        for i, row in enumerate(read_csv(task_dir, *parts))
    ]


def _mcq(raw: dict, fallback_id: str, extra_splits: dict | None = None) -> MCQ:
    return MCQ(
        id=str(raw.get("id") or fallback_id),
        body=raw["body"],
        choices_right=list(raw["choices_right"]),
        choices_wrong=list(raw["choices_wrong"]),
        choices_order=raw.get("choices_order") or None,
        splits=dict(raw.get("splits") or {}) | dict(extra_splits or {}),
    )


def human_baselines() -> dict[str, float]:
    lines = (CHECKOUT / "human_baselines" / "baselines.csv").read_text().splitlines()
    return {row.split(",")[0]: float(row.split(",")[1]) for row in lines[1:] if row.strip()}
