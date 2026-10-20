import dataclasses
import json
import math
import random
from pathlib import Path

import torch

from proprioception.families import ModelFamily


SINGLE_TOKEN_TYPES = ("top_token", "unpicked_token", "lens_top_token", "routing_partner")
PERCENT_TYPES = ("token_prob", "lens_token_prob", "routing_share")
ENTROPY_TYPES = ("entropy", "lens_entropy")
INT_ENTROPY_TYPES = ("int_entropy", "lens_int_entropy")
WHOLE_BIT_VARIANT = dict(zip(ENTROPY_TYPES, INT_ENTROPY_TYPES, strict=True))
SCALAR_TYPES = (*PERCENT_TYPES, *ENTROPY_TYPES, *INT_ENTROPY_TYPES)
EXPERT_ID_TYPES = ("routing", "routing_rank")
ROUTING_TYPES = ("routing", "routing_rank", "routing_share", "routing_partner")
LOGITS_TYPES = {"top_token", "unpicked_token", "token_prob", "entropy", "int_entropy"}
LENS_TYPES = {"lens_top_token", "lens_token_prob", "lens_entropy", "lens_int_entropy", "decision_depth"}
LENS_LAYER_TYPES = {"lens_top_token", "lens_token_prob", "lens_entropy", "lens_int_entropy"}
RANKED_TYPES = {"top_token", "lens_top_token", "routing_rank"}
NUMERIC_TYPES = (*EXPERT_ID_TYPES, "decision_depth", *PERCENT_TYPES, *ENTROPY_TYPES, *INT_ENTROPY_TYPES)
WHOLE_VALUE_TYPES = (*PERCENT_TYPES, "decision_depth", *INT_ENTROPY_TYPES)
TOP_TOKEN_MAX_RANK = 5
LENSES = ("logit",)
DOMAINS = ("routing", "logits", "lens")
DOMAIN_MEMBERS: dict[str, tuple[str, ...]] = {
    "routing": ROUTING_TYPES,
    "logits": tuple(sorted(LOGITS_TYPES)),
    "lens": tuple(sorted(LENS_TYPES)),
}
DOMAIN_TYPES: dict[str, tuple[str, ...]] = {
    domain: tuple(t for t in members if t not in ENTROPY_TYPES) for domain, members in DOMAIN_MEMBERS.items()
}
KNOWN_TYPES = tuple(t for d in DOMAINS for t in DOMAIN_MEMBERS[d])
DEFAULT_TYPES = tuple(t for d in DOMAINS for t in DOMAIN_TYPES[d])


def domain_of(type: str) -> str:
    for domain in DOMAINS:
        if type in DOMAIN_MEMBERS[domain]:
            return domain
    raise ValueError(f"unknown question type {type!r}; known: {list(KNOWN_TYPES)}")


def split_types(requested: list[str]) -> list[str]:
    seen: list[str] = []
    for item in requested:
        for name in item.split(","):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def resolve_types(domain: str, requested: list[str]) -> list[str]:
    if domain == "mix":
        allowed, default = list(KNOWN_TYPES), list(DEFAULT_TYPES)
    elif domain in DOMAINS:
        allowed, default = list(DOMAIN_MEMBERS[domain]), list(DOMAIN_TYPES[domain])
    else:
        raise ValueError(f"unknown domain {domain!r}; available: {[*DOMAINS, 'mix']}")
    names = split_types(requested)
    if not names:
        return default
    unknown = [t for t in names if t not in KNOWN_TYPES]
    if unknown:
        raise ValueError(f"unknown question types {unknown}; known: {list(KNOWN_TYPES)}")
    outside = [t for t in names if t not in allowed]
    if outside:
        raise ValueError(f"types {outside} do not belong to the {domain} domain; use --domain mix to combine domains")
    return names


def select_sample_types(samples: list[dict], types: list[str], label: str) -> list[dict]:
    names = split_types(types)
    if not names:
        return samples
    unknown = [t for t in names if t not in KNOWN_TYPES]
    if unknown:
        raise ValueError(f"unknown question types {unknown}; known: {list(KNOWN_TYPES)}")
    kept = [s for s in samples if s.get("type", "routing") in names]
    if not kept:
        present = sorted({s.get("type", "routing") for s in samples})
        raise ValueError(f"no {label} samples of types {names}; the dataset carries {present}")
    return kept


PERCENT_MAX = 99
ENTROPY_MAX_DECIBITS = 999
ENTROPY_MAX_BITS = 99


@dataclasses.dataclass(frozen=True)
class NumberFormat:
    max_int_digits: int
    decimal: bool = False

    @property
    def max_tokens(self) -> int:
        return self.max_int_digits + (2 if self.decimal else 0)


def number_format(type: str, encoding: dict) -> NumberFormat:
    if type in EXPERT_ID_TYPES:
        return NumberFormat(encoding["max_digits"])
    if type == "decision_depth":
        return NumberFormat(encoding["depth_digits"])
    if type in PERCENT_TYPES:
        return NumberFormat(len(str(PERCENT_MAX)))
    if type in ENTROPY_TYPES:
        return NumberFormat(len(str(ENTROPY_MAX_DECIBITS // 10)), decimal=True)
    if type in INT_ENTROPY_TYPES:
        return NumberFormat(len(str(ENTROPY_MAX_BITS)))
    raise ValueError(f"{type} answers are not numbers")


def spell_number(value: int, form: NumberFormat, digit_token_ids, dot_token_id=None) -> list[int]:
    if value < 0:
        raise ValueError(f"cannot spell the negative value {value}")
    integer, fraction = (value // 10, value % 10) if form.decimal else (value, None)
    if len(str(integer)) > form.max_int_digits:
        raise ValueError(f"{value} needs more than {form.max_int_digits} integer digits")
    tokens = [int(digit_token_ids[int(d)]) for d in str(integer)]
    if form.decimal:
        if dot_token_id is None:
            raise ValueError("a decimal answer needs the dot token from the dataset encoding")
        tokens += [int(dot_token_id), int(digit_token_ids[fraction])]
    return tokens


def fused_entropy_tokens(value: int, value_token_ids, digit_token_ids, dot_token_id) -> list[int]:
    if dot_token_id is None:
        raise ValueError("a decimal answer needs the dot token from the dataset encoding")
    return [int(value_token_ids[value // 10]), int(dot_token_id), int(digit_token_ids[value % 10])]


def checkpoint_eos_token_id(checkpoint: str) -> int:
    for name in ("config.json", "generation_config.json"):
        path = Path(checkpoint) / name
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        value = config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id"))
        if isinstance(value, list):
            value = value[0]
        if value is not None:
            return int(value)
    tokenizer_config = Path(checkpoint) / "tokenizer_config.json"
    if tokenizer_config.exists():
        config = json.loads(tokenizer_config.read_text())
        marker = config.get("eos_token")
        if isinstance(marker, dict):
            marker = marker.get("content")
        for token_id, entry in (config.get("added_tokens_decoder") or {}).items():
            content = entry.get("content") if isinstance(entry, dict) else entry
            if marker is not None and content == marker:
                return int(token_id)
    raise ValueError(f"no eos token id in the config, generation config or tokenizer config at {checkpoint}")


def answer_encoding(meta: dict, family: ModelFamily, device: str) -> dict:
    lens = meta.get("lens")
    if lens is not None and lens not in LENSES:
        raise ValueError(f"dataset was built under the {lens!r} lens, which this trainer cannot apply")
    values = meta.get("value_token") or {}
    value_token_ids = torch.tensor([int(values[str(v)]) for v in range(100)], device=device) if values else None
    experts = meta.get("answer_token") or {}
    expert_token_ids = None
    if value_token_ids is not None and all(str(e) in experts for e in range(family.num_experts)):
        expert_token_ids = torch.tensor([int(experts[str(e)]) for e in range(family.num_experts)], device=device)
    lens_fields = {
        "num_layers": family.num_hidden_layers,
        "depth_digits": len(str(family.num_hidden_layers - 1)),
        "num_experts_per_tok": family.num_experts_per_tok,
        "none_token": meta.get("none_token"),
        "answer_close_token": meta.get("quote_token"),
    }
    digits = meta.get("digit_token") or {}
    if digits:
        return {
            "answer_token_ids": torch.zeros(family.num_experts, dtype=torch.long, device=device),
            "digit_token_ids": [int(digits[str(d)]) for d in range(10)],
            "max_digits": len(str(family.num_experts - 1)),
            "dot_token": meta.get("dot_token"),
            "value_token_ids": value_token_ids,
            "expert_token_ids": expert_token_ids,
            "filler_token": checkpoint_eos_token_id(family.checkpoint),
            **lens_fields,
        }
    ids = torch.tensor([meta["answer_token"][str(e)] for e in range(family.num_experts)])
    return {
        "answer_token_ids": ids,
        "digit_token_ids": None,
        "max_digits": 1,
        "dot_token": None,
        "value_token_ids": value_token_ids,
        "expert_token_ids": None,
        **lens_fields,
    }


def is_fused(encoding: dict) -> bool:
    return encoding.get("value_token_ids") is not None


ANSWER_SCAN_INTEGER, ANSWER_SCAN_DECIMAL, ANSWER_SCAN_TOKEN = "integer", "decimal", "token"


LENS_DEPTH_WIDTH = 0.15


def draw_lens_layer(rng: random.Random, type: str, num_lens_layers: int, depth_center: float | None) -> int:
    if type != "lens_token_prob" or depth_center is None:
        return rng.choice(range(num_lens_layers))
    top = max(num_lens_layers - 1, 1)
    weights = [
        1.0 / (1.0 + math.exp(-(layer / top - depth_center) / LENS_DEPTH_WIDTH)) for layer in range(num_lens_layers)
    ]
    return rng.choices(range(num_lens_layers), weights=weights)[0]


def conv_answer_mode(type: str) -> str:
    if type in SINGLE_TOKEN_TYPES:
        return ANSWER_SCAN_TOKEN
    if type in ENTROPY_TYPES:
        return ANSWER_SCAN_DECIMAL
    if type in EXPERT_ID_TYPES or type in WHOLE_VALUE_TYPES:
        return ANSWER_SCAN_INTEGER
    raise ValueError(f"unknown question type {type!r}; known: {list(KNOWN_TYPES)}")


def scanned_answer_bound(type: str, num_experts: int, num_hidden_layers: int) -> int:
    if type in EXPERT_ID_TYPES:
        return num_experts
    if type == "decision_depth":
        return num_hidden_layers
    if type in PERCENT_TYPES:
        return PERCENT_MAX + 1
    if type in ENTROPY_TYPES:
        return ENTROPY_MAX_DECIBITS // 10 + 1
    if type in INT_ENTROPY_TYPES:
        return ENTROPY_MAX_BITS + 1
    raise ValueError(f"{type!r} does not scan its answer out of free text")


def scanned_answer_names_a_layer(type: str) -> bool:
    return type == "decision_depth"


def answer_width(type: str, encoding: dict) -> int:
    if type not in NUMERIC_TYPES:
        return 0
    if is_fused(encoding):
        return 3 if type in ENTROPY_TYPES else 0
    if type in EXPERT_ID_TYPES and encoding["digit_token_ids"] is None:
        return 0
    if encoding["digit_token_ids"] is None:
        return 0
    return number_format(type, encoding).max_tokens


def answer_closing(type: str, encoding: dict) -> list[int]:
    close = encoding.get("answer_close_token")
    return [int(close)] if type in SINGLE_TOKEN_TYPES and close is not None else []


def answer_extension_widths(samples: list[dict], encoding: dict) -> list[int]:
    ends = encoding.get("turn_end_token") is not None
    widths = []
    for s in samples:
        type = s.get("type", "routing")
        width = 0 if "answer_pos" in s else answer_width(type, encoding)
        widths.append(
            1 + len(answer_closing(type, encoding)) if ends and width == 0 and "answer_pos" not in s else width
        )
    return widths


def single_answer_token(type: str, label: int, encoding: dict) -> int:
    if type in SINGLE_TOKEN_TYPES:
        return int(label)
    if type in EXPERT_ID_TYPES:
        if encoding["digit_token_ids"] is None:
            return int(encoding["answer_token_ids"][label])
        if encoding.get("expert_token_ids") is not None:
            return int(encoding["expert_token_ids"][label])
    return int(encoding["value_token_ids"][label])


def answer_tokens(type: str, label: int, encoding: dict) -> list[int]:
    if answer_width(type, encoding) == 0:
        return [single_answer_token(type, label, encoding)]
    return [int(t) for t in label_answer_tokens(type, label, encoding)]


def with_turn_end(encoding: dict, checkpoint: str) -> dict:
    from proprioception.rendering import ChatRenderer

    return {
        **encoding,
        "turn_end_token": ChatRenderer(checkpoint).assistant_turn_end[0],
        "filler_token": encoding.get("filler_token", checkpoint_eos_token_id(checkpoint)),
    }


def label_answer_tokens(type: str, value: int, encoding: dict) -> list[int]:
    digits = encoding["digit_token_ids"]
    if is_fused(encoding) and type in ENTROPY_TYPES:
        return fused_entropy_tokens(value, encoding["value_token_ids"], digits, encoding.get("dot_token"))
    return spell_number(value, number_format(type, encoding), digits, encoding.get("dot_token"))


def check_probed_layers(samples: list[dict], family: ModelFamily) -> None:
    invalid = {s["layer"] for s in samples if s.get("type", "routing") in ROUTING_TYPES} - set(family.moe_layers)
    if invalid:
        raise ValueError(f"samples probe layers {sorted(invalid)} but the MoE layers are {family.moe_layers}")


def check_sample_types(samples: list[dict], encoding: dict) -> None:
    types = {s.get("type", "routing") for s in samples}
    unknown = types - set(KNOWN_TYPES)
    if unknown:
        raise ValueError(f"samples carry unknown question types {sorted(unknown)}")
    fused = is_fused(encoding)
    for type in types & set(ENTROPY_TYPES):
        if not (encoding["digit_token_ids"] and encoding.get("dot_token") is not None):
            raise ValueError(f"{type} questions need digit tokens and the dot token from the dataset encoding")
    for type in types & {*PERCENT_TYPES, *INT_ENTROPY_TYPES}:
        if not (fused or encoding["digit_token_ids"]):
            raise ValueError(f"{type} questions need digit tokens or the fused value table from the dataset")
    if "decision_depth" in types and not (fused or encoding["digit_token_ids"]):
        raise ValueError("decision_depth questions need digit tokens or the fused value table from the dataset")
    if "decision_depth" in types and fused and encoding["num_layers"] > 100:
        raise ValueError("decision depths run past the 100-entry fused value table")
    if types & LENS_LAYER_TYPES:
        valid_layers = set(range(encoding["num_layers"] - 1))
        invalid = {s.get("layer") for s in samples if s.get("type") in LENS_LAYER_TYPES} - valid_layers
        if invalid:
            raise ValueError(f"lens samples probe layers {sorted(invalid, key=str)} outside 0..{max(valid_layers)}")
    ranks = {s.get("rank", 1) for s in samples if s.get("type", "routing") != "routing_rank"}
    if not ranks <= set(range(1, TOP_TOKEN_MAX_RANK + 1)):
        raise ValueError(f"top-token ranks must lie in 1..{TOP_TOKEN_MAX_RANK}, got {sorted(ranks)}")
    expert_ranks = {s.get("rank", 1) for s in samples if s.get("type") == "routing_rank"}
    top_k = encoding.get("num_experts_per_tok", 0)
    if expert_ranks and not expert_ranks <= set(range(2, top_k + 1)):
        raise ValueError(f"routing ranks must lie in 2..{top_k}, got {sorted(expert_ranks)}")
    if "routing_partner" in types and encoding.get("none_token") is None:
        raise ValueError("routing_partner questions need the none token from the dataset encoding")


def anchor_answer_widths(samples: list[dict], encoding: dict) -> list[int]:
    ends = encoding.get("turn_end_token") is not None
    widths = []
    for sample in samples:
        type = sample.get("type", "routing")
        width = answer_width(type, encoding)
        copy = (sample.get("segments") or {}).get("assistant_pc_copy")
        if copy and sample.get("answer_start") is not None:
            widths.append(copy[-1][1] - sample["answer_start"] + int(ends))
        elif width == 0:
            widths.append(2 + len(answer_closing(type, encoding)) if ends else 1)
        elif is_fused(encoding):
            widths.append(width + 1 if ends else width)
        else:
            widths.append(width + 1)
    return widths
