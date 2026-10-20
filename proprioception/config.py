import dataclasses
import os
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from proprioception.bank import BANK_EVAL_PROMPTS
from proprioception.batching import COPY_SEGMENT, RANDOM_ANCHOR, SEGMENT_NAMES, TURN_END_SEGMENT
from proprioception.thinking import THINK_COMPONENTS
from proprioception.families import FAMILIES
from proprioception.formats import LENSES, resolve_types, split_types

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = str(REPO_ROOT / "data/sft_proprioception")


def _check_family(model: str) -> None:
    if model not in FAMILIES:
        raise ValueError(f"unknown model family {model!r}; known: {sorted(FAMILIES)}")


@dataclass
class BuildConfig:
    model: str = "olmoe"
    domain: Literal["routing", "logits", "lens", "mix"] = "routing"
    types: list[str] = field(default_factory=list)
    lens: str = "logit"
    layer: int = -1
    answer_opening: Literal["template", "generation"] = "template"
    lens_token_prob_depth: float | None = None
    num_train: int = 20000
    num_eval: int = 1024
    data_dir: str = DEFAULT_DATA_DIR
    base_url: str = ""
    concurrency: int = 64
    bank_eval_prompts: int = BANK_EVAL_PROMPTS
    seed: int = 0

    def __post_init__(self) -> None:
        _check_family(self.model)
        if self.lens not in LENSES:
            raise ValueError(f"unknown lens {self.lens!r}; available: {list(LENSES)}")
        self.types = resolve_types(self.domain, self.types)


@dataclass
class TrainConfig:
    model: str = "olmoe"
    data_dir: str = DEFAULT_DATA_DIR
    types: list[str] = field(default_factory=list)
    output: str = ""
    device: str = "cuda:0"
    seed: int = 0
    lr: float = 1e-5
    lr_schedule: Literal["constant", "cosine"] = "constant"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    batch_size: int = 16
    micro_batch_tokens: int = 0
    batch_layout: str = ""
    batch_bucket: int = 0
    lm_weight: float = 0.1
    lm_segments: list[str] = field(default_factory=lambda: ["assistant_pc_response", "assistant_turn_end"])
    lexical_match_weight: float = 1.0
    lexical_lookup_seed_samples: int = 200_000
    lexical_lookup_min_count: float = 5.0
    lexical_lookup_decay: float = 0.9
    kl_weight: float = 0.0
    anchor_positions: int = 256
    digit_reduction: Literal["sum", "mean"] = "sum"
    scalar_target_width: float = 0.0
    answer_vocab: Literal["full", "answer"] = "full"
    invariant_kernels: bool = False
    expert_parallel: bool = True
    compile: bool = True
    optimization_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    gradient_checkpointing: bool = False
    log_every_nth: int = 25
    eval_every_nth: int = 200
    save_every_nth: int = 0
    checkpoint_threads: int = 8
    checkpoint_background: bool = True
    resume: str = ""
    save_final: bool = False

    def __post_init__(self) -> None:
        _check_family(self.model)
        unknown = [name for name in self.lm_segments if name not in SEGMENT_NAMES]
        if unknown:
            raise ValueError(f"unknown lm_segments {unknown}; known: {list(SEGMENT_NAMES)}")
        if COPY_SEGMENT in self.lm_segments:
            raise ValueError(f"{COPY_SEGMENT} is trained with the readout's weight, not as an LM segment")
        self.types = resolve_types("mix", self.types) if self.types else []


@dataclass
class SftConfig(TrainConfig):
    steps: int = 2000
    grad_accum: int = 1
    warmup_steps: int = 200
    intro_weight: float = 1.0
    balance_bias_rate: float = 0.0
    kl_anchor_segments: list[str] = field(default_factory=lambda: [RANDOM_ANCHOR])
    drop_probe_label_mismatch: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = [name for name in self.kl_anchor_segments if name not in (*SEGMENT_NAMES, RANDOM_ANCHOR)]
        if unknown:
            raise ValueError(f"unknown kl_anchor_segments {unknown}; known: {[*SEGMENT_NAMES, RANDOM_ANCHOR]}")
        if self.kl_weight > 0 and not self.kl_anchor_segments:
            raise ValueError(
                f"kl_weight {self.kl_weight} needs kl_anchor_segments; {RANDOM_ANCHOR} draws from everywhere"
            )
        if TURN_END_SEGMENT in self.kl_anchor_segments:
            raise ValueError(f"{TURN_END_SEGMENT} is placed by the loss and has no range an anchor could cover")


@dataclass
class OnPolicySftConfig(TrainConfig):
    data_dir: str = ""
    device: str = "cuda"
    lr: float = 3e-5
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    beta2: float = 0.95
    batch_size: int = 12
    lexical_match_weight: float = 0.1
    kl_weight: float = 0.5
    log_every_nth: int = 5
    eval_every_nth: int = 50
    rounds: int = 1250
    grad_accum: int = 16
    warmup_rounds: int = 30
    init_checkpoint: str = ""
    anchor_checkpoint: str = ""
    server_url: str = ""
    served_model: str = ""
    weight_transport: Literal["auto", "nccl", "filesystem"] = "auto"
    sync_dir: str = "/dev/shm/proprioception-live"
    nccl_port: int = 29601
    server_gpus: int = 1
    prompt_bank: str = ""
    bank_eval_prompts: int = BANK_EVAL_PROMPTS
    sft_replay_fraction: float = 0.0
    carrier_overrequest: float = 1.5
    assistant_fraction: float = 0.5
    system_prompt_fraction: float = 0.5
    queries_per_conversation: int = 4
    answer_instruction: bool = True
    answer_nothink_fraction: float = 0.0
    eval_answer_nothink_fraction: float = 0.0
    lens_token_prob_depth: float | None = None
    unparsed_answer_fallback: Literal["none", "close", "close_prefix", "close_think_prefix"] = "none"
    answer_parser: Literal["first", "first_strict", "marked"] = "first"
    token_answer_bounded: bool = True
    digit_supervision: Literal["correct_prefix", "all"] = "correct_prefix"
    anchor_marker_positions: bool = False
    reply_kl_weight: float = 0.0
    reply_kl_samples: int = 32
    reply_kl_tokens: int = 4096
    answer_path_weight: float = 0.0
    answer_path_rows: Literal["parsed", "correct"] = "parsed"
    template_answer_fraction: float = 0.0
    template_think: list[str] = field(default_factory=list)
    template_think_style: Literal["prose", "notes"] = "prose"
    gen_temperature: float = 1.0
    gen_max_tokens: int = 1024
    answer_max_tokens: int = 1024
    eval_answer_max_tokens: int | None = None
    prefix_suffix: str = " "
    conv_eval_samples: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("data_dir", "server_url", "served_model"):
            if not getattr(self, name):
                raise ValueError(f"on-policy-sft needs {name}")
        if not 0.0 <= self.template_answer_fraction <= 1.0:
            raise ValueError("template_answer_fraction is a share between 0 and 1")
        unknown_think = [c for c in self.template_think if c not in THINK_COMPONENTS]
        if unknown_think:
            raise ValueError(f"unknown template_think components {unknown_think}; known: {list(THINK_COMPONENTS)}")
        if self.template_answer_fraction > 0 and (
            self.lm_weight <= 0 or "assistant_pc_response" not in self.lm_segments
        ):
            raise ValueError(
                "template_answer_fraction supervises the prefix through the assistant_pc_response LM segment"
            )

    @property
    def conv_eval_answer_max_tokens(self) -> int:
        return self.answer_max_tokens if self.eval_answer_max_tokens is None else self.eval_answer_max_tokens


@dataclass
class EvalConfig:
    model: str = "olmoe"
    data_dir: str = DEFAULT_DATA_DIR
    types: list[str] = field(default_factory=list)
    checkpoint: str = ""
    output: str = ""
    device: str = "cuda:0"
    seed: int = 0
    batch_size: int = 16
    lexical_lookup_seed_samples: int = 200_000
    lexical_lookup_min_count: float = 5.0
    lexical_lookup_decay: float = 0.9
    invariant_kernels: bool = False
    expert_parallel: bool = True

    def __post_init__(self) -> None:
        _check_family(self.model)
        self.types = resolve_types("mix", self.types) if self.types else []


@dataclass
class CapabilitiesConfig:
    checkpoint: str = ""
    output: str = ""
    tasks: list[str] = field(
        default_factory=lambda: ["gsm8k_cot_zeroshot", "ifeval", "minerva_math500", "aime25_answer"]
    )
    reasoning_parser: str = "auto"
    gen_kwargs: str = "temperature=0,max_gen_toks=32768"
    num_concurrent: int = 32
    limit: int = 0
    request_timeout: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    gpu_memory_utilization: float = 0.85
    server_url: str = ""

    def __post_init__(self) -> None:
        for name in ("checkpoint", "output"):
            if not getattr(self, name):
                raise ValueError(f"capabilities needs {name}")


@dataclass
class ConvertConfig:
    model: str = "qwen3_5_moe"
    checkpoint: str = ""

    def __post_init__(self) -> None:
        _check_family(self.model)


STAGES: dict[str, type] = {
    "build": BuildConfig,
    "convert": ConvertConfig,
    "sft": SftConfig,
    "on-policy-sft": OnPolicySftConfig,
    "eval": EvalConfig,
    "capabilities": CapabilitiesConfig,
}
RunConfig = BuildConfig | ConvertConfig | SftConfig | OnPolicySftConfig | EvalConfig | CapabilitiesConfig
TRUE_WORDS, FALSE_WORDS = ("true", "1", "yes"), ("false", "0", "no")


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def coerce(annotation, raw, name: str):
    origin = typing.get_origin(annotation)
    if origin is Literal:
        options = typing.get_args(annotation)
        if raw not in options:
            raise ValueError(f"{name} must be one of {list(options)}, not {raw!r}")
        return raw
    if origin in (types.UnionType, typing.Union):
        members = [member for member in typing.get_args(annotation) if member is not type(None)]
        if raw is None or (isinstance(raw, str) and raw.lower() in ("none", "")):
            return None
        return coerce(members[0], raw, name)
    if origin is list:
        (item_type,) = typing.get_args(annotation)
        items = raw if isinstance(raw, list) else [raw]
        if item_type is str:
            items = split_types([str(item) for item in items])
        return [coerce(item_type, item, name) for item in items]
    if annotation is bool:
        if isinstance(raw, bool):
            return raw
        word = str(raw).lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        raise ValueError(f"{name} takes true or false, not {raw!r}")
    if annotation is int:
        if isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
            raise ValueError(f"{name} takes an integer, not {raw!r}")
        try:
            return int(raw)
        except ValueError as error:
            raise ValueError(f"{name} takes an integer, not {raw!r}") from error
    if annotation is float:
        try:
            return float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} takes a number, not {raw!r}") from error
    if annotation is str:
        return str(raw)
    raise TypeError(f"{name}: unsupported field type {annotation}")


def help_text(stage: str) -> str:
    cls = STAGES[stage]
    lines = [f"proprio {stage} [--field value ...]", "", cls.__doc__ or "", ""]
    for item in dataclasses.fields(cls):
        default = item.default if item.default is not dataclasses.MISSING else item.default_factory()
        lines.append(f"  --{item.name:<28} {_type_name(item.type):<28} default: {default!r}")
    return "\n".join(lines)


def _type_name(annotation) -> str:
    if typing.get_origin(annotation) is Literal:
        return "{" + ",".join(str(option) for option in typing.get_args(annotation)) + "}"
    return str(annotation).replace("typing.", "")


def read_toml(path: str | Path) -> tuple[str, dict]:
    values = tomllib.loads(Path(path).read_text())
    if "cmd" not in values:
        raise ValueError(f"{path} names no cmd")
    stage = values.pop("cmd")
    return stage, {key: _expand(value) for key, value in values.items()}


def parse(stage: str, values: dict, overrides: list[str]) -> RunConfig:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {list(STAGES)}")
    cls = STAGES[stage]
    fields = {item.name: item for item in dataclasses.fields(cls)}
    merged = dict(values)
    tokens = [os.path.expandvars(token) for token in overrides]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("-h", "--help"):
            print(help_text(stage))
            raise SystemExit(0)
        if not token.startswith("--"):
            raise ValueError(f"expected a --field, got {token!r}")
        name = token[2:]
        if name not in fields:
            raise ValueError(f"unknown field --{name} for {stage}; fields: {', '.join(fields)}")
        if typing.get_origin(fields[name].type) is list:
            end = index + 1
            while end < len(tokens) and not tokens[end].startswith("--"):
                end += 1
            merged[name] = tokens[index + 1 : end]
            index = end
            continue
        if index + 1 >= len(tokens):
            raise ValueError(f"--{name} needs a value")
        merged[name] = tokens[index + 1]
        index += 2
    unknown = [key for key in merged if key not in fields]
    if unknown:
        raise ValueError(f"unknown fields {unknown} for {stage}; fields: {', '.join(fields)}")
    return cls(**{name: coerce(fields[name].type, value, name) for name, value in merged.items()})


def from_argv(argv: list[str]) -> tuple[str, RunConfig]:
    if argv and argv[0] == "@":
        if len(argv) < 2:
            raise ValueError("usage: proprio @ run.toml [overrides]")
        stage, values = read_toml(argv[1])
        return stage, parse(stage, values, argv[2:])
    if argv and argv[0] in STAGES:
        return argv[0], parse(argv[0], {}, argv[1:])
    raise ValueError(
        f"usage: proprio @ run.toml [overrides] | proprio {{{','.join(STAGES)}}} [overrides] | proprio launch ..."
    )
