import dataclasses
import json
import os
from pathlib import Path
from typing import Callable

import torch

from proprioception import modeling, modeling_lfm2, primerl_backend
from proprioception.oracle import DEFAULT_REASONING, ReasoningMarkers

MODEL_ROOT = Path(os.environ.get("PROPRIOCEPTION_MODELS", Path.home() / "models"))


@dataclasses.dataclass(frozen=True)
class ThinkingControl:
    kwarg: str = ""
    nothink_prefill: str = ""

    def template_kwargs(self, thinking: str | None) -> dict:
        if thinking is None or not self.kwarg:
            return {}
        return {self.kwarg: thinking == "think"}

    def generation_suffix(self, thinking: str | None) -> str:
        return self.nothink_prefill if thinking == "nothink" else ""


ENABLE_THINKING = ThinkingControl(kwarg="enable_thinking")
NO_REASONING = ThinkingControl()
LFM_THINKING = ThinkingControl(nothink_prefill="<think>\n\n</think>\n\n")


@dataclasses.dataclass
class ModelFamily:
    name: str
    checkpoint: str
    loader: Callable[..., torch.nn.Module]
    num_experts: int
    num_experts_per_tok: int
    moe_layers: list[int]
    num_hidden_layers: int
    batch_layout: str
    batch_bucket: int = 1
    reasoning: ReasoningMarkers = DEFAULT_REASONING
    thinking: ThinkingControl = NO_REASONING
    reasoning_parser: str = ""
    router_scores: str = ""

    @property
    def backend(self) -> str:
        return getattr(self.loader, "backend", "torch")

    def load(self, checkpoint: str, device: str, **options) -> torch.nn.Module:
        return self.loader(checkpoint, device, family=self, **options)


def read_config(checkpoint: str | Path) -> dict:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return config.get("text_config", config)


def moe_structure(config: dict) -> tuple[int, int, list[int], int]:
    num_hidden_layers = config["num_hidden_layers"]
    if config.get("enable_moe_block") is False:
        raise ValueError("config carries expert counts but its MoE block is disabled (enable_moe_block false)")
    num_experts = config.get("num_experts", config.get("n_routed_experts"))
    num_experts_per_tok = config.get("num_experts_per_tok", config.get("top_k_experts"))
    if num_experts is None or num_experts_per_tok is None:
        raise ValueError("config names neither num_experts/n_routed_experts nor num_experts_per_tok/top_k_experts")

    dense = set(config.get("mlp_only_layers") or [])
    leading = config.get("num_dense_layers", config.get("first_k_dense_replace")) or 0
    dense |= set(range(leading))
    step = config.get("decoder_sparse_step")
    if step:
        dense |= {i for i in range(num_hidden_layers) if (i + 1) % step}
    frequency = config.get("moe_layer_freq")
    if isinstance(frequency, list):
        dense |= {i for i, is_moe in enumerate(frequency) if not is_moe}
    elif frequency:
        dense |= {i for i in range(num_hidden_layers) if i % frequency}
    pattern = config.get("hybrid_override_pattern")
    if pattern:
        if len(pattern) != num_hidden_layers:
            raise ValueError(f"hybrid_override_pattern covers {len(pattern)} of {num_hidden_layers} layers")
        dense |= {i for i, entry in enumerate(pattern) if entry != "E"}

    moe_layers = [i for i in range(num_hidden_layers) if i not in dense]
    if not moe_layers:
        raise ValueError(f"config leaves no MoE layers among {num_hidden_layers}")
    return int(num_experts), int(num_experts_per_tok), moe_layers, int(num_hidden_layers)


def _pure_torch_loader(model_cls):

    def load(
        checkpoint,
        device,
        *,
        activation_checkpointing=True,
        expert_parallel=False,
        family=None,
        compile=True,
        optimization_dtype="bfloat16",
    ):
        return model_cls.from_pretrained(checkpoint, device=device)

    load.backend = "torch"
    return load


def _prime_sharded_loader(attn: str | None = None, experts_implementation: str | None = None):

    def load(
        checkpoint,
        device,
        *,
        family,
        activation_checkpointing=True,
        expert_parallel=True,
        compile=True,
        optimization_dtype="bfloat16",
    ):
        return primerl_backend.load_moe(
            checkpoint,
            device,
            family,
            expert_parallel=expert_parallel,
            activation_checkpointing=activation_checkpointing,
            compile=compile,
            attn=attn,
            experts_implementation=experts_implementation,
            optimization_dtype=optimization_dtype,
        )

    load.backend = "prime"
    return load


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    directory: str
    load: Callable[..., torch.nn.Module]
    batch_layout: str
    batch_bucket: int = 1
    reasoning: ReasoningMarkers = DEFAULT_REASONING
    thinking: ThinkingControl = NO_REASONING
    reasoning_parser: str = ""
    router_scores: str = ""

    @property
    def backend(self) -> str:
        return getattr(self.load, "backend", "torch")


FAMILIES: dict[str, FamilySpec] = {
    "olmoe": FamilySpec("OLMoE-1B-7B-0125-Instruct", _pure_torch_loader(modeling.OlmoeForCausalLM), "padded"),
    "olmoe_prime": FamilySpec("OLMoE-1B-7B-0125-Instruct", _prime_sharded_loader(), "padded", router_scores="softmax"),
    "lfm2": FamilySpec(
        "LFM2.5-8B-A1B",
        _pure_torch_loader(modeling_lfm2.Lfm2MoeForCausalLM),
        "padded",
        thinking=LFM_THINKING,
        reasoning_parser="deepseek_r1",
    ),
    "lfm2_prime": FamilySpec(
        "LFM2.5-8B-A1B",
        _prime_sharded_loader(experts_implementation="grouped_mm"),
        "padded",
        thinking=LFM_THINKING,
        reasoning_parser="deepseek_r1",
        router_scores="sigmoid",
    ),
    "qwen3_5_moe": FamilySpec(
        "Qwen3.5-35B-A3B",
        _prime_sharded_loader(),
        "packed",
        128,
        thinking=ENABLE_THINKING,
        reasoning_parser="qwen3",
        router_scores="softmax",
    ),
    "qwen3_6_moe": FamilySpec(
        "Qwen3.6-35B-A3B",
        _prime_sharded_loader(),
        "packed",
        128,
        thinking=ENABLE_THINKING,
        reasoning_parser="qwen3",
        router_scores="softmax",
    ),
    "trinity_mini": FamilySpec(
        "Trinity-Mini",
        _prime_sharded_loader(),
        "packed",
        128,
        thinking=ThinkingControl(nothink_prefill="</think>\n\n"),
        reasoning_parser="deepseek_r1",
        router_scores="sigmoid",
    ),
    "nemotron_cascade_2": FamilySpec(
        "Nemotron-Cascade-2-30B-A3B",
        _prime_sharded_loader(),
        "packed",
        128,
        thinking=ENABLE_THINKING,
        reasoning_parser="nemotron_v3",
        router_scores="sigmoid",
    ),
    "glm4_7_flash": FamilySpec(
        "GLM-4.7-Flash",
        _prime_sharded_loader(),
        "padded",
        thinking=ENABLE_THINKING,
        reasoning_parser="glm45",
        router_scores="sigmoid",
    ),
    "gemma4_26b_a4b": FamilySpec(
        "gemma-4-26B-A4B-it",
        _prime_sharded_loader(attn="sdpa"),
        "padded",
        reasoning=ReasoningMarkers("<|channel>thought\n", "<channel|>"),
        thinking=ENABLE_THINKING,
        reasoning_parser="gemma4",
        router_scores="probabilities",
    ),
}

FAMILY_BY_MODEL_TYPE = {
    "olmoe": "olmoe",
    "lfm2_moe": "lfm2",
    "qwen3_5_moe": "qwen3_5_moe",
    "afmoe": "trinity_mini",
    "nemotron_h": "nemotron_cascade_2",
    "glm4_moe_lite": "glm4_7_flash",
    "gemma4": "gemma4_26b_a4b",
}


def family_for_checkpoint(checkpoint: str | Path) -> str:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    model_type = config.get("model_type", "")
    if model_type not in FAMILY_BY_MODEL_TYPE:
        raise ValueError(f"{checkpoint}: model_type {model_type!r} belongs to no known family")
    return FAMILY_BY_MODEL_TYPE[model_type]


def resolve_family(name: str) -> ModelFamily:
    spec = FAMILIES.get(name)
    if spec is None:
        raise ValueError(f"unknown model family {name!r}; available: {sorted(FAMILIES)}")
    checkpoint = MODEL_ROOT / spec.directory
    if not (checkpoint / "config.json").exists():
        raise ValueError(f"no checkpoint for family {name!r} at {checkpoint}")
    num_experts, num_experts_per_tok, moe_layers, num_hidden_layers = moe_structure(read_config(checkpoint))
    return ModelFamily(
        name=name,
        checkpoint=str(checkpoint),
        loader=spec.load,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_layers=moe_layers,
        num_hidden_layers=num_hidden_layers,
        batch_layout=spec.batch_layout,
        batch_bucket=spec.batch_bucket,
        reasoning=spec.reasoning,
        thinking=spec.thinking,
        reasoning_parser=spec.reasoning_parser,
        router_scores=spec.router_scores,
    )
