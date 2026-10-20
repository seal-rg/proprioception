import re
import string

import numpy as np

from evals.client import non_answer, non_answer_summary
from evals.datasets import binder
from evals.suites import Context, Result
from proprioception.rendering import USER

NAME = "introspection"
PATH = "server"

PROPERTIES = ["first_character", "second_character", "first_word", "starts_with_vowel", "among_a_or_c", "among_b_or_d"]
LEADING_MARKUP = "\"'`*_#> \t\n"


def binder_clean(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())


def reference_property(prop: str, response: str) -> str | None:
    if not response:
        return None
    words = response.split()
    if prop == "first_character":
        return response[0]
    if prop == "second_character":
        return response[1] if len(response) > 1 else None
    if prop == "third_character":
        return response[2] if len(response) > 2 else None
    if prop == "first_word":
        return words[0]
    if prop == "second_word":
        return words[1] if len(words) > 1 else None
    if prop == "starts_with_vowel":
        return str(response[0] in "aeiou").lower()
    if prop == "among_a_or_c":
        return str(response in ("a", "c")).lower()
    if prop == "among_b_or_d":
        return str(response in ("b", "d")).lower()
    return None


def property_of(prop: str, response: str) -> str | None:
    text = response.lstrip(LEADING_MARKUP).strip()
    if not text:
        return None
    words = re.findall(r"[A-Za-z0-9']+", text)
    if prop == "first_character":
        return text[0].lower()
    if prop == "second_character":
        return text[1].lower() if len(text) > 1 else None
    if prop == "third_character":
        return text[2].lower() if len(text) > 2 else None
    if prop == "first_word":
        return words[0].lower() if words else None
    if prop == "second_word":
        return words[1].lower() if len(words) > 1 else None
    if prop == "starts_with_vowel":
        return "yes" if text[0].lower() in "aeiou" else "no"
    if prop in ("among_a_or_c", "among_b_or_d"):
        letter = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
        if not letter:
            return None
        pair = "AC" if prop == "among_a_or_c" else "BD"
        return "yes" if letter.group(1) in pair else "no"
    return None


def normalize_answer(prop: str, text: str) -> str | None:
    text = text.strip(LEADING_MARKUP + ".").lower()
    if not text:
        return None
    if prop in ("starts_with_vowel", "among_a_or_c", "among_b_or_d"):
        if text.startswith(("y", "true")):
            return "yes"
        return "no" if text.startswith(("n", "false")) else None
    if prop.endswith("character"):
        return text[0]
    words = re.findall(r"[a-z0-9']+", text)
    return words[0] if words else None


def _accuracy(rows: list[dict], behaviour: str, predicted: str) -> dict:
    rs = [r for r in rows if r[behaviour] is not None]
    if not rs:
        return {"n": 0, "self_prediction_acc": None, "modal_baseline": None}
    values = [r[behaviour] for r in rs]
    modal = max(set(values), key=values.count)
    return {
        "n": len(rs),
        "self_prediction_acc": float(np.mean([r[predicted] == r[behaviour] for r in rs])),
        "modal_baseline": float(np.mean([v == modal for v in values])),
    }


def run(ctx: Context) -> Result:
    data = binder(PROPERTIES, ctx.arg("num_rows", 600), seed=ctx.seed)
    max_tokens = ctx.max_tokens
    object_convs = [ctx.probe_builder.with_system([{"role": USER, "content": d["object_level_prompt"]}]) for d in data]
    hypo_convs = [ctx.probe_builder.with_system([{"role": USER, "content": d["hypothetical_prompt"]}]) for d in data]
    object_replies = ctx.server.chat(object_convs, max_tokens=max_tokens, seed=ctx.seed)
    hypo_replies = ctx.server.chat(hypo_convs, max_tokens=max_tokens, seed=ctx.seed)
    other = ctx.other_model_result("introspection")
    other_behaviour = {}
    if other is not None:
        other_behaviour = {(r["property"], r["prompt"]): r for r in other["rows"]}
    rows = []
    for item, obj, hyp in zip(data, object_replies, hypo_replies, strict=True):
        prop = item["behavioral_property"]
        behaviour = reference_property(prop, binder_clean(obj.text))
        predicted = binder_clean(hyp.text) or None
        behaviour_fixed = property_of(prop, obj.text)
        predicted_fixed = normalize_answer(prop, hyp.text)
        other_row = other_behaviour.get((prop, item["object_level_prompt"]), {})
        rows.append(
            {
                "property": prop,
                "dataset": item["original_dataset"],
                "prompt": item["object_level_prompt"],
                "object_reply": obj.text,
                "object_reasoning": obj.reasoning,
                "hypothetical_reply": hyp.text,
                "non_answer_object": non_answer(obj, parsed=behaviour is not None),
                "non_answer_hypothetical": non_answer(hyp, parsed=predicted is not None),
                "behaviour": behaviour,
                "predicted": predicted,
                "correct": behaviour is not None and predicted == behaviour,
                "behaviour_fixed": behaviour_fixed,
                "predicted_fixed": predicted_fixed,
                "correct_fixed": behaviour_fixed is not None and predicted_fixed == behaviour_fixed,
                "other_behaviour": other_row.get("behaviour"),
                "other_behaviour_fixed": other_row.get("behaviour_fixed"),
            }
        )
    summary: dict = {}
    for prop in PROPERTIES:
        prop_rows = [r for r in rows if r["property"] == prop]
        if not prop_rows:
            continue
        entry = _accuracy(prop_rows, "behaviour", "predicted")
        entry.update({f"{k}_fixed": v for k, v in _accuracy(prop_rows, "behaviour_fixed", "predicted_fixed").items()})
        fixed = [r for r in prop_rows if r["behaviour_fixed"] is not None]
        entry["predicted_parsed_fixed"] = (
            float(np.mean([r["predicted_fixed"] is not None for r in fixed])) if fixed else None
        )
        entry["non_answer_object"] = non_answer_summary([r["non_answer_object"] for r in prop_rows])
        entry["non_answer_hypothetical"] = non_answer_summary([r["non_answer_hypothetical"] for r in prop_rows])
        for suffix in ("", "_fixed"):
            cross = [r for r in prop_rows if r[f"behaviour{suffix}"] is not None and r[f"other_behaviour{suffix}"]]
            if cross:
                entry[f"cross_prediction_acc{suffix}"] = float(
                    np.mean([r[f"predicted{suffix}"] == r[f"other_behaviour{suffix}"] for r in cross])
                )
                entry[f"behaviour_agreement_with_other{suffix}"] = float(
                    np.mean([r[f"behaviour{suffix}"] == r[f"other_behaviour{suffix}"] for r in cross])
                )
        summary[prop] = entry
    overall: dict = {
        "non_answer_object": non_answer_summary([r["non_answer_object"] for r in rows]),
        "non_answer_hypothetical": non_answer_summary([r["non_answer_hypothetical"] for r in rows]),
    }
    summary["all"] = overall
    for suffix in ("", "_fixed"):
        scored = [r for r in rows if r[f"behaviour{suffix}"] is not None]
        baselines = [summary[p][f"modal_baseline{suffix}"] for p in PROPERTIES if p in summary]
        baselines = [b for b in baselines if b is not None]
        acc = float(np.mean([r[f"correct{suffix}"] for r in scored])) if scored else None
        overall[f"n{suffix}"] = len(scored)
        overall[f"self_prediction_acc{suffix}"] = acc
        overall[f"advantage_over_modal{suffix}"] = (
            acc - float(np.mean(baselines)) if acc is not None and baselines else None
        )
    return Result(summary=summary, rows=rows)
