import dataclasses
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.client import ChatClient
from evals.config import EvalConfig, ModelSpec
from evals.probes import Probe, ProbeBuilder
from evals.torchmodel import TorchModel
from evals.truth import TorchLabeler
from proprioception.families import ModelFamily

SUITES = (
    "stability",
    "control",
    "batch_routing",
    "contextual",
    "ablation",
    "capability",
    "injection",
    "legibility",
    "template",
    "perez",
    "factuality",
    "introspection",
    "turpin",
    "sad",
    "ciaware",
    "mask",
    "mask_self_report",
    "harmbench",
    "strongreject",
    "truthfulqa",
    "pushback",
    "lanham",
    "benchmarks",
    "diagnostics",
    "diagnostics_torch",
)


@dataclasses.dataclass
class Result:
    summary: dict = dataclasses.field(default_factory=dict)
    rows: list[dict] = dataclasses.field(default_factory=list)
    pending: list[Probe] = dataclasses.field(default_factory=list)
    compare: list[Probe] = dataclasses.field(default_factory=list)
    finish: Callable[[list[Probe]], "Result"] | None = None


@dataclasses.dataclass
class Context:
    config: EvalConfig
    model: ModelSpec
    family: ModelFamily
    args: dict
    out_dir: Path
    seed: int
    log: Callable[[str], None]
    client: ChatClient | None = None
    builder: ProbeBuilder | None = None
    judge: ChatClient | None = None
    torch: TorchModel | None = None
    labeler: TorchLabeler | None = None
    key: str = ""

    @property
    def server(self) -> ChatClient:
        assert self.client is not None, "this suite needs the served model"
        return self.client

    @property
    def probe_builder(self) -> ProbeBuilder:
        assert self.builder is not None
        return self.builder

    @property
    def torch_model(self) -> TorchModel:
        assert self.torch is not None, "this suite needs the torch model"
        return self.torch

    @property
    def torch_labeler(self) -> TorchLabeler:
        assert self.labeler is not None
        return self.labeler

    def require_judge(self) -> ChatClient:
        if self.judge is None:
            raise RuntimeError(
                f"suite {self.args.get('_suite', '')}needs a judge: set judge_model (and the key in .env)"
            )
        return self.judge

    def arg(self, key: str, default: Any) -> Any:
        value = self.args.get(key, default)
        if isinstance(default, int) and not isinstance(default, bool) and isinstance(value, str):
            return int(value)
        if isinstance(default, float) and isinstance(value, str):
            return float(value)
        return value

    @property
    def max_tokens(self) -> int:
        return int(self.args.get("max_tokens", self.config.max_tokens))

    @property
    def judge_max_tokens(self) -> int:
        return int(self.config.judge_max_tokens)

    @property
    def types(self) -> list[str]:
        return list(self.config.types)

    def other_model_result(self, suite: str) -> dict | None:
        import json

        for name in self.config.models:
            if name == self.model.name:
                continue
            path = self.out_dir.parent / name / f"{suite}.json"
            if path.exists():
                return json.loads(path.read_text())
        return None


def load(name: str):
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; known: {SUITES}")
    return importlib.import_module(f"evals.suites.{name}")


GROUPS = {
    "torch": [n for n in SUITES if n in ("ablation", "capability", "injection", "diagnostics_torch", "template")],
    "sad": ["sad"],
}
GROUPS["server"] = [n for n in SUITES if n not in GROUPS["torch"] and n != "sad"]


DIAGNOSTICS = ("diagnostics", "diagnostics_torch")


def resolve(selection: str) -> list[str]:
    if not selection or selection == "all":
        return list(SUITES)
    names = []
    for item in [n.strip() for n in selection.split(",") if n.strip()]:
        names.extend(GROUPS.get(item, [item]))
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise ValueError(f"unknown suites {unknown}; known: {SUITES}")
    return sorted(names, key=lambda n: n in DIAGNOSTICS)
