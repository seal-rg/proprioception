import dataclasses
import random
import re

from proprioception.formats import SINGLE_TOKEN_TYPES
from proprioception.questions import EXPERT_RANK_REFS, RANK_REFS
from proprioception.think_banks import (
    THINK_GENERAL_BANK,
    THINK_QUANTITY_PHRASES,
    THINK_RESTATE_QUESTION_FRAMES,
    THINK_RESTATEMENT_FRAMES,
    THINK_SENSE_BANKS,
    THINK_SENSE_FRAMES,
)

THINK_COMPONENTS = ("quote_question", "restate_question", "ppc_questions_general", "restatement", "sense_copy")
THINK_STYLES = ("prose", "notes")
COMPONENT_PROBABILITY = 0.5
NOTES_OPENER = "Here's a thinking process:"
NOTES_HEADERS = (
    ("Analyze User Input",),
    ("Identify Key Constraints & Nature of the Question", "Identify Key Constraints & Facts", "Identify Key Facts"),
    ("Formulate Response Strategy", "Draft Response (Mental Refinement)"),
)


@dataclasses.dataclass
class ThinkBlock:
    lines: list[str]
    head: str = ""
    sense: str | None = None


def _fill(template: str, **fields) -> str:
    text = template.format(**fields)
    starts = [0] + [m.end() for m in re.finditer(r"(?<!\bi\.e)(?<!\be\.g)[.!?]\s+", template)]
    for at in starts:
        if template.startswith("{", at):
            pos = len(template[:at].format(**fields))
            text = text[:pos] + text[pos : pos + 1].upper() + text[pos + 1 :]
    return text


def quantity_phrase(rng: random.Random, pc_type: str, *, layer: int, rank: int) -> str:
    rank_ref = EXPERT_RANK_REFS[rank] if pc_type == "routing_rank" else RANK_REFS.get(rank, "")
    return rng.choice(THINK_QUANTITY_PHRASES[pc_type]).format(layer=layer, rank_ref=rank_ref)


def _render(rng: random.Random, sections: list[list[str]], sense_prefix: str | None, style: str) -> ThinkBlock:
    lines = [line for section in sections for line in section]
    if style == "prose":
        return ThinkBlock(
            lines, "\n".join(lines), None if sense_prefix is None else ("\n" if lines else "") + sense_prefix
        )
    filled = [(rng.choice(titles), section) for titles, section in zip(NOTES_HEADERS, sections) if section]
    if sense_prefix is not None:
        filled.append((rng.choice(NOTES_HEADERS[-1]), []))
    if not filled:
        return ThinkBlock([], "", None)
    parts = [NOTES_OPENER]
    for number, (title, section) in enumerate(filled, start=1):
        parts.append(f"\n{number}.  **{title}:**" + "".join(f"\n   - {line}" for line in section))
    head = "\n".join(parts)
    if sense_prefix is None:
        return ThinkBlock(lines, head, None)
    return ThinkBlock(lines, head, "\n   - " + sense_prefix)


def compose_think(
    rng: random.Random,
    pc_type: str,
    components: tuple[str, ...] | list[str],
    *,
    question: str,
    token_ref: str,
    layer: int,
    rank: int,
    context_ref: str | None,
    prefix_suffix: str,
    style: str = "prose",
) -> ThinkBlock:
    unknown = [c for c in components if c not in THINK_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown thinking components {unknown}; known: {list(THINK_COMPONENTS)}")
    if style not in THINK_STYLES:
        raise ValueError(f"unknown thinking style {style!r}; known: {list(THINK_STYLES)}")
    drawn = {c for c in THINK_COMPONENTS if c in components and rng.random() < COMPONENT_PROBABILITY}
    quantity = quantity_phrase(rng, pc_type, layer=layer, rank=rank)
    rank_ref = EXPERT_RANK_REFS[rank] if pc_type == "routing_rank" else RANK_REFS.get(rank, "")
    sections: list[list[str]] = [[], [], []]
    if "quote_question" in drawn:
        sections[0].append(f'User asks: "{question}"')
    if "restate_question" in drawn:
        sections[0].append(_fill(rng.choice(THINK_RESTATE_QUESTION_FRAMES), quantity=quantity, token_ref=token_ref))
    if "ppc_questions_general" in drawn:
        sections[1].append(_fill(rng.choice(THINK_GENERAL_BANK), quantity=quantity))
    if "restatement" in drawn:
        frames = THINK_RESTATEMENT_FRAMES
        if context_ref is None:
            frames = [f for f in frames if "{context_ref}" not in f]
        sections[1].append(
            _fill(rng.choice(frames), quantity=quantity, token_ref=token_ref, context_ref=context_ref or "")
        )
    sense = None
    if "sense_copy" in drawn:
        specific = THINK_SENSE_BANKS[pc_type]
        if rng.random() < len(specific) / (len(specific) + len(THINK_SENSE_FRAMES)):
            sense = rng.choice(specific).format(layer=layer, rank_ref=rank_ref)
        else:
            sense = _fill(rng.choice(THINK_SENSE_FRAMES), quantity=quantity)
        sense += ' "' if pc_type in SINGLE_TOKEN_TYPES else prefix_suffix
    return _render(rng, sections, sense, style)
