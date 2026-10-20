import dataclasses
import random
import re

import numpy as np

from evals.client import PROMPT_TOO_LONG, ChatClient, non_answer_summary
from evals.probes import ProbeBuilder
from proprioception.oracle import ReasoningMarkers, ends_inside_reasoning, visible_assistant_content
from proprioception.questions import compose_token_question, format_token_ref, render_token_context, sample_distractor
from proprioception.rendering import USER, tokens_within


@dataclasses.dataclass
class BinaryProbe:
    sentence: str
    context: str
    question_template: str
    token_index: int
    layer: int
    prefix_ids: list[int] = dataclasses.field(default_factory=list)
    pos: int = -1
    token_text: str = ""
    truth: int | None = None
    distractor: int | None = None
    correct_letter: str = ""
    picked: str | None = None
    answer_text: str = ""
    truncated: bool = False
    stale: bool = False
    non_answer: str | None = None

    def row(self) -> dict:
        return {
            "format": "binary",
            "layer": self.layer,
            "token": self.token_text,
            "truth": self.truth,
            "distractor": self.distractor,
            "correct_letter": self.correct_letter,
            "picked": self.picked,
            "correct": self.picked == self.correct_letter and not self.stale,
            "parsed": self.picked is not None,
            "stale": self.stale,
            "truncated": self.truncated,
            "non_answer": self.non_answer,
            "answer_text": self.answer_text,
        }


def pick_letter(text: str, letters: str = "AB") -> str | None:
    boxed = re.findall(rf"\\boxed\{{\s*(?:\\text(?:bf)?\{{)?\s*\(?([{letters}])\)?\s*\}}?\s*\}}", text)
    if boxed:
        return boxed[-1]
    stated = re.findall(rf"(?i:\banswer\s*(?:is|:))[\s*_:]*\(?([{letters}])\)?(?![A-Za-z0-9])", text)
    if stated:
        return stated[-1]
    marked = set(re.findall(rf"\(([{letters}])\)", text))
    if marked:
        return marked.pop() if len(marked) == 1 else None
    match = re.match(rf"\W*([{letters}])(?=\s*(?:[.,:;!?)\]*]|$))", text.strip())
    return match.group(1) if match else None


def picked_option(reply: str, markers: ReasoningMarkers) -> str | None:
    return pick_letter(visible_assistant_content(reply, markers))


def _message(context: str, question: str) -> str:
    return f"{context}\n\n{question}"


def build(
    builder: ProbeBuilder, client: ChatClient, sentences: list[str], *, rng: random.Random, num: int, max_tokens: int
) -> list[BinaryProbe]:
    family = builder.family
    probes: list[BinaryProbe] = []
    attempts = 0
    while len(probes) < num and attempts < num * 3:
        attempts += 1
        sentence = rng.choice(sentences).strip().replace("\n", " ")[:400]
        context, question_template = render_token_context(rng, sentence)
        probes.append(BinaryProbe(sentence, context, question_template, -1, rng.choice(family.moe_layers)))
    first = []
    for probe in probes:
        placeholder = probe.question_template.format(token_ref="x", layer=probe.layer, option_a=0, option_b=1)
        message = _message(probe.context, placeholder)
        ids, offsets = builder.renderer.encode_content_with_offsets(message)
        start = message.find(probe.sentence)
        inside = [
            i
            for i in tokens_within(offsets, start, start + len(probe.sentence))[:-1]
            if builder.renderer.text(ids[i]).strip()
        ]
        probe.token_index = rng.choice(inside) if inside else -1
        first.append(
            builder.renderer.render(
                builder.with_system([{"role": USER, "content": message}]), add_generation_prompt=True
            )
        )
    kept = [(p, r) for p, r in zip(probes, first, strict=True) if p.token_index >= 0]
    probes, first = [p for p, _ in kept], [r for _, r in kept]
    captures = client.routing([r.token_ids for r in first])
    for probe, rendered, routed in zip(probes, first, captures, strict=True):
        span = [s for s in rendered.content_spans() if s[2] == USER][0]
        pos = span[0] + probe.token_index
        token_id = rendered.token_ids[pos]
        same = [i for i in range(span[0], span[1]) if rendered.token_ids[i] == token_id]
        token_ref = format_token_ref(builder.renderer.text(token_id), same.index(pos) + 1, len(same))
        topk = [int(e) for e in routed[pos, probe.layer]]
        question = compose_token_question(
            rng,
            probe.question_template,
            token_ref=token_ref,
            layer=probe.layer,
            true_expert=topk[0],
            distractor_expert=sample_distractor(rng, family.num_experts, topk),
        )
        probe.pos, probe.token_text = pos, builder.renderer.text(token_id)
        probe.truth, probe.distractor, probe.correct_letter = (
            topk[0],
            question.distractor_expert,
            question.correct_letter,
        )
        probe.prefix_ids = builder.renderer.render(
            builder.with_system([{"role": USER, "content": _message(probe.context, question.text)}]),
            add_generation_prompt=True,
        ).token_ids
        probe.stale = probe.prefix_ids[: pos + 1] != rendered.token_ids[: pos + 1]
    gens = client.generate([p.prefix_ids for p in probes], max_tokens=max_tokens, routing=True)
    for probe, gen in zip(probes, gens, strict=True):
        if gen.finish_reason == PROMPT_TOO_LONG:
            probe.non_answer = PROMPT_TOO_LONG
            continue
        assert gen.routed_experts is not None
        live = int(gen.routed_experts[probe.pos, probe.layer, 0])
        if not probe.stale and live != probe.truth:
            probe.stale = live != probe.distractor
            if not probe.stale:
                probe.truth, probe.distractor = live, probe.truth
                probe.correct_letter = "B" if probe.correct_letter == "A" else "A"
        ids = builder.assembler.strip_turn_end(gen.token_ids)
        probe.answer_text = builder.assembler.full_reply_text(ids)
        probe.picked = picked_option(probe.answer_text, family.reasoning)
        probe.truncated = gen.finish_reason == "length"
        probe.non_answer = _non_answer(probe, family.reasoning)
    return probes


def _non_answer(probe: BinaryProbe, markers: ReasoningMarkers) -> str | None:
    if probe.picked is not None:
        return None
    if probe.truncated:
        return "truncated"
    if markers.open and ends_inside_reasoning(probe.answer_text, markers):
        return "unclosed_reasoning"
    if not visible_assistant_content(probe.answer_text, markers).strip():
        return "empty"
    return "unparsed"


def summarize(probes: list[BinaryProbe]) -> dict:
    fresh = [p for p in probes if not p.stale]
    if not fresh:
        return {"n": 0}
    parsed = [p for p in fresh if p.picked is not None]
    return {
        "n": len(fresh),
        "acc": float(np.mean([p.picked == p.correct_letter for p in fresh])),
        "acc_parsed": float(np.mean([p.picked == p.correct_letter for p in parsed])) if parsed else None,
        "parsed": len(parsed) / len(fresh),
        "unparsed": 1 - len(parsed) / len(fresh),
        "truncated": float(np.mean([p.truncated for p in fresh])),
        "stale": len(probes) - len(fresh),
        "correct_on_a": float(np.mean([p.correct_letter == "A" for p in fresh])),
        "picked_a": float(np.mean([p.picked == "A" for p in fresh if p.picked]))
        if any(p.picked for p in fresh)
        else None,
        "chance": 0.5,
        "non_answer": non_answer_summary([p.non_answer for p in fresh]),
    }
