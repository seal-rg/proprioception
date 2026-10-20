import dataclasses
import random
from collections import Counter
import re
from collections.abc import Sequence

import numpy as np

from evals.client import NON_ANSWERS, PROMPT_TOO_LONG, ChatClient, Generation
from evals.datasets import authored
from evals.truth import NONE_PARTNER, Request, label_matches, routing_label, scalar_distance
from proprioception.convassembly import ConversationAssembler
from proprioception.families import ModelFamily
from proprioception.formats import (
    WHOLE_BIT_VARIANT,
    LENS_LAYER_TYPES,
    ROUTING_TYPES,
    SCALAR_TYPES,
    TOP_TOKEN_MAX_RANK,
    domain_of,
)
from proprioception.questions import (
    whole_bit_wording,
    CONV_ANSWER_INSTRUCTIONS,
    EXPERT_RANK_REFS,
    PARTNER_PREFIXES,
    RANK_REFS,
    compose_conv_question,
    format_token_ref,
)
from proprioception.oracle import ends_inside_reasoning, visible_assistant_content
from proprioception.rendering import ASSISTANT, SYSTEM, USER, ChatRenderer


def opens_with_number(text: str) -> bool:
    return bool(re.match(r"\s*\d+\s*(\n|$|[.,:;])", text or ""))


@dataclasses.dataclass
class Conversation:
    messages: list[dict]
    ids: list[int]
    spans: list[tuple[int, int, str]]
    routing: np.ndarray | None = None
    reasoning_ends: dict[int, int] = dataclasses.field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.spans)


@dataclasses.dataclass
class Probe:
    messages: list[dict]
    prefix_ids: list[int]
    pos: int
    type: str
    layer: int
    rank: int
    span: tuple[int, int]
    role: str
    turn: int
    token_id: int
    token_text: str
    question: str
    conv_len: int = 0
    meta: dict = dataclasses.field(default_factory=dict)
    label: int | None = None
    model_answer: int | str | None = None
    answer_text: str = ""
    parsed: bool = False
    correct: bool = False
    distance: float | None = None

    @property
    def request(self) -> Request:
        return Request(type=self.type, pos=self.pos, layer=self.layer, rank=self.rank, span=self.span)

    def row(self) -> dict:
        return {
            "type": self.type,
            "layer": self.layer,
            "rank": self.rank,
            "role": self.role,
            "turn": self.turn,
            "pos": self.pos,
            "prefix_tokens": len(self.prefix_ids),
            "token_id": self.token_id,
            "token": self.token_text,
            "question": self.question,
            "label": self.label,
            "model_answer": self.model_answer,
            "answer_text": self.answer_text,
            "parsed": self.parsed,
            "correct": self.correct,
            "distance": self.distance,
            "word_initial": self.token_text.startswith(" ") or self.pos == self.span[0],
            "opens_with_number": opens_with_number(self.answer_text),
            **self.meta,
        }


class ProbeBuilder:
    def __init__(self, checkpoint: str, family: ModelFamily, *, system_prompt: str = "", thinking: str | None = None):
        self.checkpoint = checkpoint
        self.family = family
        self.thinking = thinking
        self._modes: dict[str | None, tuple[ChatRenderer, ConversationAssembler]] = {}
        self.renderer, self.assembler = self.mode(thinking)
        self.system_prompt = system_prompt
        self.multi_digit = len(self.renderer.encode_content("57")) > 1
        bare = self.renderer.encode_content(PARTNER_PREFIXES[0])
        self.none_token = self.renderer.encode_content(f"{PARTNER_PREFIXES[0]} none")[len(bare)]
        self.lens_layers = list(range(family.num_hidden_layers - 1))
        self.novel_bank: dict[str, list[str]] = {}
        for row in authored("novel_questions"):
            self.novel_bank.setdefault(row["type"], []).append(row["template"])
        for decimal, whole in WHOLE_BIT_VARIANT.items():
            self.novel_bank[whole] = [whole_bit_wording(t) for t in self.novel_bank.get(decimal, [])]
        self.dropped_carriers = 0
        self.total_carriers = 0

    def mode(self, thinking: str | None) -> tuple[ChatRenderer, ConversationAssembler]:
        if thinking not in self._modes:
            renderer = ChatRenderer.for_thinking(self.checkpoint, thinking, self.family.thinking)
            assembler = ConversationAssembler(
                "http://unused",
                self.checkpoint,
                "",
                multi_digit=len(renderer.encode_content("57")) > 1,
                markers=self.family.reasoning,
                check_server=False,
                renderer=renderer,
            )
            self._modes[thinking] = (renderer, assembler)
        return self._modes[thinking]

    def banned_ids(self, thinking: str | None) -> list[list[int]]:
        if thinking != "nothink" or not self.family.reasoning.open:
            return []
        return [self.renderer.encode_content_as_template(self.family.reasoning.open)]

    def with_system(self, messages: Sequence[dict]) -> list[dict]:
        if self.system_prompt and (not messages or messages[0]["role"] != SYSTEM):
            return [{"role": SYSTEM, "content": self.system_prompt}, *messages]
        return list(messages)

    def from_messages(self, messages: Sequence[dict]) -> Conversation:
        messages = self.with_system(messages)
        if messages[-1]["role"] != ASSISTANT:
            raise ValueError("a probed conversation must end with an assistant turn")
        rendered = self.renderer.render([*messages, {"role": USER, "content": "?"}])
        spans = [(s, e, r) for s, e, r in rendered.content_spans() if r in (USER, ASSISTANT)][:-1]
        return Conversation(messages=messages, ids=rendered.token_ids[: spans[-1][1]], spans=spans)

    def start_prompt(self, user_prompt: str) -> tuple[list[int], list[tuple[int, int, str]], list[dict]]:
        messages = self.with_system([{"role": USER, "content": user_prompt}])
        rendered = self.renderer.render(messages, add_generation_prompt=True)
        spans = [(s, e, r) for s, e, r in rendered.content_spans() if r == USER]
        return rendered.token_ids, spans, messages

    def extend_prompt(
        self, conversation: Conversation, user_prompt: str
    ) -> tuple[list[int], list[tuple[int, int, str]], list[dict]]:
        ids = list(conversation.ids) + self.assembler.seg_assistant_to_user
        content = self.renderer.encode_content(user_prompt)
        spans = [*conversation.spans, (len(ids), len(ids) + len(content), USER)]
        ids = ids + content + self.assembler.seg_user_to_assistant
        return ids, spans, [*conversation.messages, {"role": USER, "content": user_prompt}]

    def exchanges(
        self,
        client: ChatClient,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        seed: int = 0,
        carrier: str = "rerender",
        prior: Sequence[Conversation] | None = None,
        aligned: bool = False,
    ) -> list:
        starts = [
            self.start_prompt(p) if prior is None else self.extend_prompt(prior[i], p) for i, p in enumerate(prompts)
        ]
        gens = client.generate(
            [s[0] for s in starts], max_tokens=max_tokens, temperature=temperature, seed=seed, routing=True
        )
        out = []
        dropped = 0
        for (prompt_ids, spans, messages), gen in zip(starts, gens, strict=True):
            reply_ids = self.assembler.strip_turn_end(gen.token_ids)
            visible = self.assembler.carrier_reply(gen.token_ids)
            if not visible:
                dropped += 1
                if aligned:
                    out.append(None)
                continue
            text_messages = [*messages, {"role": ASSISTANT, "content": visible}]
            if carrier == "exact" and reply_ids:
                ids = prompt_ids + reply_ids
                assert gen.routed_experts is not None
                rows = gen.routed_experts
                routing = np.full((len(ids), rows.shape[1], rows.shape[2]), -1, dtype=np.int16)
                usable = min(rows.shape[0], len(ids) - gen.row_start)
                routing[gen.row_start : gen.row_start + usable] = rows[:usable]
                turn_spans = [*spans, (len(prompt_ids), len(ids), ASSISTANT)]
                out.append(
                    Conversation(
                        messages=text_messages,
                        ids=ids,
                        spans=turn_spans,
                        routing=routing,
                        reasoning_ends={len(turn_spans) - 1: len(prompt_ids) + self.reasoning_end(reply_ids)},
                    )
                )
            else:
                out.append(self.from_messages(text_messages))
        self.dropped_carriers += dropped
        self.total_carriers += len(prompts)
        return out

    def reasoning_end(self, reply_ids: list[int]) -> int:
        joined = ""
        for i, piece in enumerate(self.renderer.texts(reply_ids)):
            joined += piece
            if joined.endswith(self.family.reasoning.close):
                return i + 1
        return 0

    def build(
        self,
        conversation: Conversation | Sequence[dict],
        *,
        rng: random.Random,
        type: str,
        turn: int | None = None,
        role: str = "",
        layer: int | None = None,
        pos: int | None = None,
        rank: int | None = None,
        meta: dict | None = None,
        phrasing: str = "train",
        thinking: str | None = "default",
    ) -> Probe | None:
        conv = conversation if isinstance(conversation, Conversation) else self.from_messages(conversation)
        assembler = self.assembler if thinking == "default" else self.mode(thinking)[1]
        candidates = [i for i, (_, _, r) in enumerate(conv.spans) if not role or r == role]
        if turn is None and pos is not None:
            containing = [i for i in candidates if conv.spans[i][0] <= pos < conv.spans[i][1]]
            if not containing:
                return None
            turn = containing[0]
        if turn is None:
            turn = rng.choice(candidates)
        start, end, want_role = conv.spans[turn]
        ids = conv.ids
        positions = [i for i in range(start, end) if self.renderer.text(ids[i]).strip()]
        if type not in ROUTING_TYPES:
            positions = [i for i in positions if i != end - 1]
        if pos is not None:
            positions = [pos] if pos in positions else []
        if not positions:
            return None
        pos = rng.choice(positions)
        token_id = ids[pos]
        same = [i for i in range(start, end) if ids[i] == token_id]
        if layer is None:
            if type in ROUTING_TYPES:
                layer = rng.choice(self.family.moe_layers)
            elif type in LENS_LAYER_TYPES:
                layer = rng.choice(self.lens_layers)
            else:
                layer = -1
        if rank is None:
            if type == "routing_rank":
                rank = rng.randint(2, self.family.num_experts_per_tok)
            elif type in ("top_token", "lens_top_token"):
                rank = rng.randint(1, TOP_TOKEN_MAX_RANK)
            else:
                rank = 1
        token_text = self.renderer.text(token_id)
        same_role = [i for i in range(conv.num_turns) if conv.spans[i][2] == want_role]
        ordinal = same_role.index(turn) + 1
        if want_role == USER:
            location = "in my first message" if len(same_role) == 1 else f"in my message number {ordinal}"
        else:
            location = "in your reply" if len(same_role) == 1 else f"in your reply number {ordinal}"
        token_ref = f"{format_token_ref(token_text, same.index(pos) + 1, len(same))} {location}"
        if phrasing == "novel":
            rank_ref = EXPERT_RANK_REFS[rank] if type == "routing_rank" else RANK_REFS[rank]
            text = rng.choice(self.novel_bank[type]).format(token_ref=token_ref, layer=layer, rank_ref=rank_ref)
            question = f"{text} {rng.choice(CONV_ANSWER_INSTRUCTIONS[type])}"
        else:
            question = compose_conv_question(
                rng, type, token_ref=token_ref, layer=layer, rank=rank, num_layers=self.family.num_hidden_layers
            )
        prefix_ids = (
            list(ids)
            + assembler.seg_assistant_to_user
            + self.renderer.encode_content(question)
            + assembler.seg_user_to_assistant
        )
        probe = Probe(
            messages=[*conv.messages, {"role": USER, "content": question}],
            prefix_ids=prefix_ids,
            pos=pos,
            type=type,
            layer=layer,
            rank=rank,
            span=(start, end),
            role=want_role,
            turn=turn,
            token_id=token_id,
            token_text=token_text,
            question=question,
            conv_len=len(ids),
            meta=dict(meta or {}),
        )
        if want_role == ASSISTANT and turn in conv.reasoning_ends:
            probe.meta["channel"] = "reasoning" if pos < conv.reasoning_ends[turn] else "answer"
        routed = conv.routing is not None and type in ROUTING_TYPES and type != "routing_share"
        if routed and conv.routing is not None and (conv.routing[pos] >= 0).all():
            probe.meta["generation_label"] = routing_label(probe.request, conv.routing, ids)
        return probe

    def read_answer(self, probe: Probe, generation: Generation, thinking: str | None = None) -> None:
        assembler = self.assembler if thinking is None else self.mode(thinking)[1]
        result = assembler.append_answer(
            {"input_ids": probe.prefix_ids, "type": probe.type},
            generation.token_ids,
            num_experts=self.family.num_experts,
            num_hidden_layers=self.family.num_hidden_layers,
            terminated=generation.finish_reason == "stop",
        )
        probe.model_answer = result["model_answer"]
        probe.answer_text = result["answer_text"]
        probe.parsed = probe.model_answer is not None
        probe.meta["truncated"] = generation.finish_reason == "length"
        markers = self.family.reasoning
        raw = (markers.open if assembler.answer_opens_reasoning else "") + probe.answer_text
        if probe.parsed:
            probe.meta["non_answer"] = None
        elif generation.finish_reason == PROMPT_TOO_LONG:
            probe.meta["non_answer"] = PROMPT_TOO_LONG
        elif generation.finish_reason == "length":
            probe.meta["non_answer"] = "truncated"
        elif bool(markers.open) and ends_inside_reasoning(raw, markers):
            probe.meta["non_answer"] = "unclosed_reasoning"
        elif not visible_assistant_content(raw, markers).strip():
            probe.meta["non_answer"] = "empty"
        else:
            probe.meta["non_answer"] = "unparsed"
        self.score(probe)

    def score(self, probe: Probe) -> None:
        probe.correct = self.matches(probe.type, probe.label, probe.model_answer)
        if probe.type in SCALAR_TYPES and not isinstance(probe.model_answer, str):
            probe.distance = scalar_distance(probe.type, probe.label, probe.model_answer)

    def matches(self, type: str, label: int | None, answer: int | str | None) -> bool:
        return label_matches(type, label, answer, self.none_token, self.renderer.text)

    def visible_answer(self, probe: Probe, thinking: str | None = None) -> str:
        assembler = self.assembler if thinking is None else self.mode(thinking)[1]
        opened = self.family.reasoning.open if assembler.answer_opens_reasoning else ""
        return visible_assistant_content(opened + probe.answer_text, self.family.reasoning)

    def with_answer(self, probe: Probe) -> Conversation:
        visible = self.visible_answer(probe)
        if not visible.strip():
            raise ValueError("a probe without a visible answer cannot be continued")
        return self.from_messages([*probe.messages, {"role": ASSISTANT, "content": visible}])


def answer_questions(
    client: ChatClient,
    builder: ProbeBuilder,
    probes: list[Probe],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    seed: int = 0,
    truth: str = "answer",
    capture_sink=None,
    load: Sequence[list[int]] = (),
) -> None:
    routing_types = all(p.type in ROUTING_TYPES and p.type != "routing_share" for p in probes)
    generations = client.generate(
        [p.prefix_ids for p in probes] + list(load),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        routing=routing_types,
        reuse_cache=truth == "generation",
    )[: len(probes)]
    for probe, generation in zip(probes, generations, strict=True):
        if routing_types and generation.routed_experts is not None and generation.row_start <= probe.pos:
            if capture_sink is not None:
                capture_sink(probe, generation.routed_experts)
            answer_label = routing_label(probe.request, generation.routed_experts, probe.prefix_ids)
            probe.meta["answer_time_label"] = answer_label
            probe.label = probe.meta["generation_label"] if truth == "generation" else answer_label
        elif routing_types and truth == "generation":
            probe.label = probe.meta.get("generation_label")
        builder.read_answer(probe, generation)


def needs_torch_labels(probes: list[Probe]) -> bool:
    return any(p.label is None and (p.type not in ROUTING_TYPES or p.type == "routing_share") for p in probes)


def summarize(probes: list[Probe], keys: Sequence[str] = ()) -> dict:
    out: dict = {}

    def stats(rows: list[Probe]) -> dict:
        labelled = [p for p in rows if p.label is not None]
        if not labelled:
            return {"n": 0}
        entry = {
            "n": len(labelled),
            "acc": float(np.mean([p.correct for p in labelled])),
            "parsed": float(np.mean([p.parsed for p in labelled])),
            "truncated": float(np.mean([p.meta.get("truncated", False) for p in labelled])),
            "opens_with_number": float(np.mean([opens_with_number(p.answer_text) for p in labelled])),
            "label_entropy_bits": label_entropy([p.label for p in labelled]),
            "non_answer": {
                name: float(np.mean([p.meta.get("non_answer") == name for p in labelled])) for name in NON_ANSWERS
            },
        }
        distances = [p.distance for p in labelled if p.distance is not None]
        if distances:
            entry["mae"] = float(np.mean(distances))
            entry["within_5"] = float(np.mean([d <= 5 for d in distances]))
        return entry

    out["all"] = stats(probes)
    for type in sorted({p.type for p in probes}):
        out[f"type={type}"] = stats([p for p in probes if p.type == type])
    for key in keys:
        values = sorted({str(p.meta.get(key, p.row().get(key))) for p in probes})
        for value in values:
            out[f"{key}={value}"] = stats([p for p in probes if str(p.meta.get(key, p.row().get(key))) == value])
    return out


def label_entropy(labels: Sequence[int | None]) -> float:
    values = [v for v in labels if v is not None]
    if not values:
        return float("nan")
    counts = np.array(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def domain_types(domain: str, types: Sequence[str]) -> list[str]:
    return [t for t in types if domain_of(t) == domain]


def chance(type: str, family: ModelFamily) -> float | None:
    if type in ("routing", "routing_rank"):
        return 1 / family.num_experts
    if type == "decision_depth":
        return 1 / family.num_hidden_layers
    return None


def partner_none(label: int | None) -> bool:
    return label == NONE_PARTNER
