import math

from proprioception.formats import (
    ENTROPY_MAX_BITS,
    ENTROPY_MAX_DECIBITS,
    ENTROPY_TYPES,
    INT_ENTROPY_TYPES,
    LENS_LAYER_TYPES,
    LOGITS_TYPES,
    PERCENT_MAX,
    ROUTING_TYPES,
)
from proprioception.oracle import ConversationRouting, LayerReadout, PositionReadout

NONE_PARTNER = -1
CAPTURE_ROUTING_TYPES = ("routing", "routing_rank", "routing_partner")


def needs_readout(type: str) -> bool:
    return type not in CAPTURE_ROUTING_TYPES


def routing_capture_label(
    type: str, capture: ConversationRouting, *, position: int, layer: int, rank: int, span: tuple[int, int]
) -> int:
    top = capture.experts_at(position, layer)
    if type == "routing":
        return top[0]
    if type == "routing_rank":
        return top[rank - 1]
    if type == "routing_partner":
        start, end = span
        best: tuple[int, int] | None = None
        for other in range(start, end):
            if other == position or capture.token_ids[other] == capture.token_ids[position]:
                continue
            if capture.experts_at(other, layer)[0] != top[0]:
                continue
            distance = abs(other - position)
            if best is None or distance < best[0]:
                best = (distance, other)
        return NONE_PARTNER if best is None else capture.token_ids[best[1]]
    raise ValueError(f"{type} is not labeled from a routed-experts capture")


def _distribution_label(type: str, readout: LayerReadout, rank: int, following: int) -> int:
    if type in ("top_token", "lens_top_token"):
        return readout.top_ids[rank - 1]
    if type == "unpicked_token":
        first, second = readout.top_ids[0], readout.top_ids[1]
        return second if first == following else first
    if type in ("token_prob", "lens_token_prob"):
        if readout.next_logprob is None:
            raise ValueError("a probability label needs the token that follows the probed one")
        return min(int(round(math.exp(readout.next_logprob) * 100)), PERCENT_MAX)
    if type in ENTROPY_TYPES:
        return min(int(round(readout.entropy_bits * 10)), ENTROPY_MAX_DECIBITS)
    if type in INT_ENTROPY_TYPES:
        return min(int(round(readout.entropy_bits)), ENTROPY_MAX_BITS)
    raise ValueError(type)


def readout_label(type: str, readout: PositionReadout, *, layer: int, rank: int, following: int) -> int:
    if type == "routing_share":
        ids, weights = readout.router[layer]
        if not ids:
            raise ValueError(f"layer {layer} captured no router selection")
        return min(int(round(weights[0] / max(sum(weights), 1e-12) * 100)), PERCENT_MAX)
    if type in LOGITS_TYPES:
        return _distribution_label(type, readout.layers[-1], rank, following)
    if type in LENS_LAYER_TYPES:
        return _distribution_label(type, readout.layers[layer], rank, following)
    if type == "decision_depth":
        final = readout.layers[-1].argmax
        trailing = 0
        for entry in reversed(readout.layers):
            if entry.argmax != final:
                break
            trailing += 1
        return len(readout.layers) - trailing
    if type in ROUTING_TYPES:
        raise ValueError(f"{type} is labeled from a routed-experts capture")
    raise ValueError(f"unknown question type {type!r}")
