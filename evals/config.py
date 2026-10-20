import dataclasses
import os
from typing import Any
import tomllib
from pathlib import Path

from proprioception.families import FAMILIES, ModelFamily, resolve_family
from proprioception.formats import DOMAINS, DOMAIN_TYPES, resolve_types, split_types
from proprioception.questions import PROPRIOCEPTION_SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _cache_root() -> Path:
    if os.environ.get("PROPRIOCEPTION_EVAL_CACHE"):
        return Path(os.environ["PROPRIOCEPTION_EVAL_CACHE"])
    if os.environ.get("PROPRIOCEPTION_DATA"):
        return Path(os.environ["PROPRIOCEPTION_DATA"]) / "eval_cache"
    return Path.home() / ".cache" / "proprioception-evals"


CACHE = _cache_root()

SYSTEM_PROMPTS = {
    "none": "",
    "aware": PROPRIOCEPTION_SYSTEM_PROMPT,
}


@dataclasses.dataclass
class ModelSpec:
    name: str
    checkpoint: str
    role: str = "trained"
    domain: str = ""
    types: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EvalConfig:
    name: str
    family: str
    domain: str
    models: dict[str, ModelSpec]
    types: list[str] = dataclasses.field(default_factory=list)
    suites: list[str] = dataclasses.field(default_factory=list)
    run_models: list[str] = dataclasses.field(default_factory=list)
    output: str = "${PROPRIOCEPTION_OUTPUTS}/evals"
    gpu: str = "auto"
    port: int = 0
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 40960
    max_num_seqs: int = 256
    reasoning_parser: str = "auto"
    moe_backend: str = "triton"
    attention_backend: str = ""
    concurrency: int = 64
    seed: int = 0
    max_tokens: int = 8192
    system_prompt: str = "none"
    carrier: str = "rerender"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_model: str = ""
    judge_api_key: str = "${OPENROUTER_KEY}"
    judge_max_tokens: int = 4096
    answer_mode: str = "think"
    suite: dict[str, dict] = dataclasses.field(default_factory=dict)
    requested_types: list[str] = dataclasses.field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.judge_api_key = os.path.expandvars(self.judge_api_key)
        if self.answer_mode not in ("think", "nothink"):
            raise ValueError(f"answer_mode must be think or nothink, not {self.answer_mode!r}")
        if self.system_prompt not in SYSTEM_PROMPTS:
            raise ValueError(f"system_prompt must be one of {sorted(SYSTEM_PROMPTS)}, not {self.system_prompt!r}")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; known: {sorted(FAMILIES)}")
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}; known: {DOMAINS}")
        self.requested_types = split_types(self.types)
        self.types = resolve_types(self.domain, self.types)
        for spec in self.models.values():
            if spec.role == "trained" and not spec.domain:
                spec.domain = self.domain
            if spec.domain and spec.domain not in DOMAINS:
                raise ValueError(f"model {spec.name}: unknown domain {spec.domain!r}")
            if spec.domain:
                spec.types = resolve_types(spec.domain, spec.types)

    @property
    def family_info(self) -> ModelFamily:
        return resolve_family(self.family)

    @property
    def resolved_reasoning_parser(self) -> str:
        return FAMILIES[self.family].reasoning_parser if self.reasoning_parser == "auto" else self.reasoning_parser

    @property
    def output_dir(self) -> Path:
        suffix = "" if self.answer_mode == "think" else f"-{self.answer_mode}"
        if self.system_prompt != "none":
            suffix += f"-sys_{self.system_prompt}"
        return Path(os.path.expandvars(self.output)) / f"{self.name}{suffix}"

    @property
    def resolved_gpu(self) -> str:
        if self.gpu != "auto":
            return self.gpu
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return ",".join(str(i) for i in range(len(visible.split(",")))) if visible else "0"

    @property
    def scheduler_allocated(self) -> bool:
        return self.gpu == "auto"

    def suite_args(self, suite: str) -> dict:
        return dict(self.suite.get(suite, {}))

    def model_names(self, selected: str = "") -> list[str]:
        names = [n.strip() for n in selected.split(",") if n.strip()] if selected else list(self.models)
        unknown = [n for n in names if n not in self.models]
        if unknown:
            raise ValueError(f"unknown models {unknown}; configured: {list(self.models)}")
        return names


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _deep_update(table: dict, other: dict) -> None:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(table.get(key), dict):
            _deep_update(table[key], value)
        else:
            table[key] = value


def _set_dotted(table: dict, key: str, raw: str) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        table = table.setdefault(part, {})
    current = table.get(parts[-1])
    if isinstance(current, bool):
        table[parts[-1]] = raw.lower() in ("1", "true", "yes")
    elif isinstance(current, int):
        table[parts[-1]] = int(raw)
    elif isinstance(current, float):
        table[parts[-1]] = float(raw)
    elif isinstance(current, list) or parts[-1] in ("suites", "run_models", "types"):
        table[parts[-1]] = [v for v in raw.split(",") if v]
    else:
        table[parts[-1]] = raw


def load_config(argv: list[str]) -> tuple[EvalConfig, dict[str, str]]:
    load_dotenv()
    table: dict = {}
    extra: dict[str, str] = {}
    fields = {f.name for f in dataclasses.fields(EvalConfig)}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("@"):
            _deep_update(table, tomllib.loads(Path(arg[1:]).read_text()))
            index += 1
            continue
        if not arg.startswith("--"):
            raise ValueError(f"unexpected argument {arg!r}")
        key = arg[2:]
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"--{key} needs a value")
        value = argv[index + 1]
        if key == "models":
            key = "run_models"
        if key.split(".")[0] in fields:
            _set_dotted(table, key, value)
        else:
            extra[key] = value
        index += 2
    table = _expand(table)
    if "name" not in table:
        raise ValueError("config needs a name (the results directory under output)")
    models = {
        name: ModelSpec(name=name, **spec) if isinstance(spec, dict) else ModelSpec(name=name, checkpoint=str(spec))
        for name, spec in table.pop("models", {}).items()
    }
    if "checkpoint" in extra:
        checkpoint = os.path.expandvars(extra.pop("checkpoint"))
        name = extra.pop("checkpoint_name", "") or checkpoint_name(checkpoint)
        role = "base" if extra.pop("checkpoint_role", "trained") == "base" else "trained"
        types = [t for t in extra.pop("checkpoint_types", "").split(",") if t]
        models[name] = ModelSpec(name=name, checkpoint=checkpoint, role=role, types=types)
        table.setdefault("run_models", [name])
    if not models:
        raise ValueError("config names no [models] and no --checkpoint was given")
    return EvalConfig(models=models, **table), extra


def checkpoint_name(checkpoint: str) -> str:
    parts = [p for p in checkpoint.rstrip("/").split("/") if p]
    return parts[-2] if len(parts) > 1 and parts[-1] == "hf" else parts[-1]


def default_types(domain: str) -> tuple[str, ...]:
    return DOMAIN_TYPES[domain]
