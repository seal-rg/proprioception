import dataclasses
import math
from collections.abc import Callable

import numpy as np
import torch

from evals.torchmodel import TorchModel
from proprioception.formats import (
    ENTROPY_MAX_BITS,
    ENTROPY_MAX_DECIBITS,
    ENTROPY_TYPES,
    INT_ENTROPY_TYPES,
    LENS_LAYER_TYPES,
    LOGITS_TYPES,
    PERCENT_MAX,
    ROUTING_TYPES,
    SINGLE_TOKEN_TYPES,
    TOP_TOKEN_MAX_RANK,
)
from proprioception.questions import CONV_ANSWER_PREFIXES, token_reply_matches

NONE_PARTNER = -1


@dataclasses.dataclass(frozen=True)
class Request:
    type: str
    pos: int
    layer: int = -1
    rank: int = 1
    span: tuple[int, int] = (0, 0)


def routing_label(request: Request, routed: np.ndarray, ids: list[int]) -> int | None:
    row = routed[request.pos, request.layer]
    if request.type == "routing":
        return int(row[0])
    if request.type == "routing_rank":
        return int(row[request.rank - 1])
    if request.type == "routing_partner":
        top = int(row[0])
        return partner(request, ids, lambda p: int(routed[p, request.layer, 0]), top)
    return None


def partner(request: Request, ids: list[int], top_at, top: int) -> int:
    start, end = request.span
    best = None
    for p in range(start, end):
        if p == request.pos or ids[p] == ids[request.pos]:
            continue
        if top_at(p) != top:
            continue
        distance = abs(p - request.pos)
        if best is None or distance < best[0]:
            best = (distance, p)
    return NONE_PARTNER if best is None else ids[best[1]]


def _distribution_label(type: str, rank: int, log_probs: torch.Tensor, observed: int) -> int:
    if type in ("top_token", "lens_top_token"):
        return int(log_probs.topk(TOP_TOKEN_MAX_RANK).indices[rank - 1])
    if type == "unpicked_token":
        first, second = log_probs.topk(2).indices.tolist()
        return second if first == observed else first
    if type in ("token_prob", "lens_token_prob"):
        return min(int(round(float(log_probs[observed].exp()) * 100)), PERCENT_MAX)
    if type in (*ENTROPY_TYPES, *INT_ENTROPY_TYPES):
        bits = float(-(log_probs.exp() * log_probs).sum()) / math.log(2)
        if type in INT_ENTROPY_TYPES:
            return min(int(round(bits)), ENTROPY_MAX_BITS)
        return min(int(round(bits * 10)), ENTROPY_MAX_DECIBITS)
    raise ValueError(type)


class TorchLabeler:
    def __init__(self, model: TorchModel):
        self.model = model

    @torch.no_grad()
    def label(self, sequences: list[list[int]], requests: list[list[Request]], batch_size: int = 4) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in sequences]
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            needs_hidden = any(
                r.type in LENS_LAYER_TYPES or r.type == "decision_depth" for i in chunk for r in requests[i]
            )
            fwd = self.model.forward([sequences[i] for i in chunk], hidden=needs_hidden)
            for row, i in enumerate(chunk):
                out[i] = [self._one(fwd, row, sequences[i], r) for r in requests[i]]
        return out

    def _one(self, fwd, row: int, ids: list[int], r: Request) -> int:
        model = self.model
        if r.type in ROUTING_TYPES:
            routing = fwd.routing[r.layer]
            selected = routing.selected[row]
            if r.type == "routing":
                return int(selected[r.pos, 0])
            if r.type == "routing_rank":
                return int(selected[r.pos, r.rank - 1])
            if r.type == "routing_share":
                weights = routing.weights[row, r.pos].float()
                return min(int(round(float(weights[0] / weights.sum()) * 100)), PERCENT_MAX)
            top = int(selected[r.pos, 0])
            return partner(r, ids, lambda p: int(selected[p, 0]), top)
        observed = ids[min(r.pos + 1, len(ids) - 1)]
        if r.type in LOGITS_TYPES:
            log_probs = torch.log_softmax(fwd.logits[row, r.pos].float(), dim=-1)
            return _distribution_label(r.type, r.rank, log_probs, observed)
        if r.type in LENS_LAYER_TYPES:
            lens = model.lens_logits(fwd.hidden[r.layer][row, r.pos].unsqueeze(0))[0]
            return _distribution_label(r.type, r.rank, torch.log_softmax(lens.float(), dim=-1), observed)
        if r.type == "decision_depth":
            residuals = torch.stack([h[row, r.pos] for h in fwd.hidden])
            lens_argmax = model.lens_logits(residuals).float().argmax(-1)
            final = int(fwd.logits[row, r.pos].float().argmax())
            agree = (lens_argmax == final).int().flip(0).cumprod(0).sum().item()
            return int(model.num_layers - agree) if agree > 0 else -1
        raise ValueError(f"unknown type {r.type}")


def label_matches(
    type: str,
    label: int | None,
    answer: int | str | None,
    none_token: int | None = None,
    token_text: Callable[[int], str] | None = None,
) -> bool:
    if label is None or answer is None:
        return False
    if type in SINGLE_TOKEN_TYPES:
        if token_text is None:
            raise ValueError(f"{type} answers are read on text and need the label token's text")
        names_none = label in (NONE_PARTNER, none_token)
        return token_reply_matches(
            str(answer),
            "" if names_none else token_text(label),
            names_none=names_none,
            prefixes=CONV_ANSWER_PREFIXES[type],
        )
    return label == answer


def scalar_distance(type: str, label: int | None, answer: int | None) -> float | None:
    if label is None or answer is None:
        return None
    return abs(label - answer)
