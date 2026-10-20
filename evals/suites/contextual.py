import math
import random
import re
import unicodedata

import numpy as np

from evals import judge
from evals.client import non_answer, non_answer_summary
from evals.datasets import authored, held_out_prompts
from evals.probes import opens_with_number
from evals.suites import Context, Result
from evals.suites.stability import exchanges
from proprioception.families import read_config
from proprioception.formats import DOMAINS, domain_of
from proprioception.questions import format_token_ref
from proprioception.rendering import USER

NAME = "contextual"
PATH = "server"


def full_attention_layers(config: dict, num_hidden_layers: int) -> list[int]:
    layer_types = config.get("layer_types")
    if layer_types:
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]
    pattern = config.get("hybrid_override_pattern")
    if pattern:
        return [i for i, c in enumerate(pattern) if c == "*"]
    return list(range(num_hidden_layers))


def facts(ctx: Context) -> dict:
    config = read_config(ctx.model.checkpoint)
    family = ctx.family
    layer_types = config.get("layer_types") or []
    attention = full_attention_layers(config, family.num_hidden_layers)
    conv = [i for i, t in enumerate(layer_types) if "conv" in t]
    tokenizer = ctx.probe_builder.renderer.tokenizer
    vocab_size = int(config.get("vocab_size", len(tokenizer)))
    tokenizer_size = len(tokenizer)
    return {
        "num_hidden_layers": family.num_hidden_layers,
        "num_experts": family.num_experts,
        "num_experts_per_tok": family.num_experts_per_tok,
        "moe_layers": list(family.moe_layers),
        "dense_layers": [i for i in range(family.num_hidden_layers) if i not in family.moe_layers],
        "num_moe_layers": len(family.moe_layers),
        "vocab_size": vocab_size,
        "hidden_size": int(config.get("hidden_size", 0)),
        "max_position_embeddings": int(config.get("max_position_embeddings", 0)),
        "num_attention_heads": int(config.get("num_attention_heads", 0)),
        "num_key_value_heads": int(config.get("num_key_value_heads", config.get("num_attention_heads", 0))),
        "attention_layers": attention,
        "conv_layers": conv,
        "intermediate_size": int(config.get("intermediate_size", 0)),
        "moe_intermediate_size": int(config.get("moe_intermediate_size", config.get("intermediate_size", 0))),
        "model_name": family.checkpoint.rstrip("/").split("/")[-1],
        "tokenizer_size": tokenizer_size,
        "first_unused_token_id": tokenizer_size if tokenizer_size < vocab_size else None,
        "lens_layers": list(range(family.num_hidden_layers - 1)),
        "max_entropy_bits": math.log2(vocab_size),
        "math": math,
    }


def placeholders(f: dict, rng: random.Random) -> dict:
    return {
        "dense_layer": rng.choice(f["dense_layers"]) if f["dense_layers"] else -1,
        "moe_layer": rng.choice(f["moe_layers"]),
        "bad_layer": f["num_hidden_layers"] + 3,
        "bad_expert": f["num_experts"] + 5,
        "bad_rank": f["num_experts_per_tok"] + 2,
        "last_layer": f["num_hidden_layers"] - 1,
        "top_k": f["num_experts_per_tok"],
        "num_experts": f["num_experts"],
        "num_layers": f["num_hidden_layers"],
    }


def evaluate(expr: str, f: dict):
    return eval(
        expr,
        {
            "__builtins__": {
                "len": len,
                "range": range,
                "list": list,
                "min": min,
                "max": max,
                "round": round,
                "abs": abs,
                "sum": sum,
                "sorted": sorted,
            }
        },
        f,
    )


def first_int(text: str) -> int | None:
    match = re.search(r"-?\d[\d,]*", text)
    return int(match.group().replace(",", "")) if match else None


def first_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def yes_no(text: str) -> str | None:
    head = text.strip().lower()[:40]
    if re.match(r"^\W*(yes|yeah|correct|true)\b", head):
        return "yes"
    if re.match(r"^\W*(no|nope|false|incorrect)\b", head):
        return "no"
    return None


def score_fact(
    record: dict,
    reply: str,
    f: dict,
    options: list[str] | None,
    correct_option: str | None,
    extracted: object = None,
    declined: bool | None = None,
) -> dict:
    answer = record["answer"]
    kind = answer["type"]
    visible = reply.strip()
    row: dict = {"kind": kind}
    if kind == "int":
        expected = int(evaluate(answer["expr"], f))
        got = _as_int(extracted)
        row.update(expected=expected, got=got, correct=got == expected)
    elif kind == "float":
        expected = float(evaluate(answer["expr"], f))
        got = _as_float(extracted)
        row.update(
            expected=expected, got=got, correct=got is not None and abs(got - expected) <= answer.get("tol", 0.5)
        )
    elif kind == "yesno":
        expected = "yes" if evaluate(answer["expr"], f) else "no"
        got = yes_no(visible)
        row.update(expected=expected, got=got, correct=got == expected)
    elif kind == "set":
        expected = sorted(int(v) for v in evaluate(answer["expr"], f))
        got = _as_int_list(extracted)
        union = set(expected) | set(got)
        jaccard = len(set(expected) & set(got)) / len(union) if union else 1.0
        row.update(expected=expected, got=got, correct=got == expected, jaccard=jaccard)
    elif kind == "choice":
        assert options is not None and correct_option is not None
        letters = [chr(ord("A") + i) for i in range(len(options))]
        picked = _parse_choice(visible, letters)
        row.update(expected=correct_option, got=options[letters.index(picked)] if picked else None)
        row["correct"] = row["got"] == correct_option
    elif kind == "reject":
        row.update(declined=bool(declined), gave_number=first_int(visible) is not None)
        row["correct"] = row["declined"]
    return row


def _as_int(value) -> int | None:
    try:
        return int(value) if value is not None and not isinstance(value, (list, dict, bool)) else None
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None and not isinstance(value, (list, dict, bool)) else None
    except (TypeError, ValueError):
        return None


def _as_int_list(value) -> list[int]:
    if isinstance(value, list):
        return sorted({int(v) for v in value if _as_int(v) is not None})
    single = _as_int(value)
    return [] if single is None else [single]


def _parse_choice(text: str, letters: list[str]) -> str | None:
    match = re.search(rf"\\boxed\{{\s*\(?([{''.join(letters)}])\)?\s*\}}", text) or re.search(
        rf"\(([{''.join(letters)}])\)", text
    )
    if match:
        return match.group(1)
    match = re.match(rf"\W*([{''.join(letters)}])(?![A-Za-z])", text.strip())
    return match.group(1) if match else None


def ask(ctx: Context, prompts: list[str], max_tokens: int, non_answers: list[str | None]) -> list[str]:
    builder = ctx.probe_builder
    replies = ctx.server.chat(
        [builder.with_system([{"role": USER, "content": p}]) for p in prompts], max_tokens=max_tokens, seed=ctx.seed
    )
    non_answers.extend(non_answer(r) for r in replies)
    return [r.text for r in replies]


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    grader = ctx.require_judge()
    f = facts(ctx)
    max_tokens = ctx.max_tokens
    rows: list[dict] = []
    non_answers: list[str | None] = []

    records = authored("contextual")
    trained = sorted({domain_of(t) for t in (ctx.model.types or ())})
    prompts, extras = [], []
    for record in records:
        subs = placeholders(f, rng)
        if "{dense_layer}" in record["question"] and subs["dense_layer"] < 0:
            continue
        question = record["question"].format(**subs)
        options = correct = None
        if record["answer"]["type"] == "choice":
            options = list(record["answer"]["options"])
            correct = str(evaluate(record["answer"]["expr"], f))
            rng.shuffle(options)
            question += "\n\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))
            question += "\n\nAnswer with the letter of the correct option."
        prompts.append(question)
        extras.append((record, options, correct))
    replies = ask(ctx, prompts, max_tokens, non_answers)
    kinds = {"int": "integer", "float": "number", "set": "integers"}
    targets = [i for i, (record, _, _) in enumerate(extras) if record["answer"]["type"] in kinds]
    values = judge.extract(
        grader,
        [(prompts[i], replies[i], kinds[extras[i][0]["answer"]["type"]]) for i in targets],
        max_tokens=ctx.judge_max_tokens,
    )
    extracted = dict(zip(targets, values, strict=True))
    rejects = [i for i, (record, _, _) in enumerate(extras) if record["answer"]["type"] == "reject"]
    verdicts = judge.yes_no(
        grader,
        [judge.REJECTION_PROMPT.format(question=prompts[i], answer=replies[i]) for i in rejects],
        max_tokens=ctx.judge_max_tokens,
    )
    declined = dict(zip(rejects, verdicts, strict=True))
    for i, ((record, options, correct), reply) in enumerate(zip(extras, replies, strict=True)):
        row = score_fact(record, reply, f, options, correct, extracted.get(i), declined.get(i))
        rows.append(
            {
                "section": "facts",
                "id": record["id"],
                "category": record["category"],
                "domain": record["domain"],
                "contextual_for": record["contextual_for"],
                "grounding": record["grounding"],
                "contextual": bool(set(record["contextual_for"]) & set(trained)),
                "reply": reply,
                **row,
            }
        )

    convs = exchanges(ctx, held_out_prompts(24, seed=ctx.seed + 3))
    templates = [t for t in authored("impossible_questions") if t["domain"] in ("any", ctx.config.domain)]
    prompts, extras = [], []
    for template in templates:
        for _ in range(ctx.arg("num_per_impossible_template", 3)):
            conv = rng.choice(convs)
            start, end = [s for s in conv.spans if s[2] == USER][0][:2]
            ids = conv.ids
            positions = [i for i in range(start, end) if ctx.probe_builder.renderer.text(ids[i]).strip()]
            pos = rng.choice(positions)
            same = [i for i in range(start, end) if ids[i] == ids[pos]]
            token_text = ctx.probe_builder.renderer.text(ids[pos])
            token_ref = f"{format_token_ref(token_text, same.index(pos) + 1, len(same))} in my first message"
            subs = {**placeholders(f, rng), "token_ref": token_ref, "missing_token": "'zxqv'"}
            if template["kind"] == "missing_occurrence":
                subs["token_ref"] = f"the {len(same) + 2}th occurrence of the token '{token_text}' in my first message"
            question = template["template"].format(**subs)
            prompts.append(question)
            extras.append((template, conv))
    conversations = [
        [*conv.messages, {"role": USER, "content": q}] for q, (_, conv) in zip(prompts, extras, strict=True)
    ]
    replies = ctx.server.chat(conversations, max_tokens=max_tokens, seed=ctx.seed)
    non_answers.extend(non_answer(r) for r in replies)
    verdicts = judge.yes_no(
        grader,
        [judge.REJECTION_PROMPT.format(question=q, answer=r.text) for q, r in zip(prompts, replies, strict=True)],
        max_tokens=ctx.judge_max_tokens,
    )
    for (template, _), reply, question, verdict in zip(extras, replies, prompts, verdicts, strict=True):
        text = reply.text
        rows.append(
            {
                "section": "impossible",
                "question": question,
                "id": template["id"],
                "kind": template["kind"],
                "domain": template["domain"],
                "declined": bool(verdict),
                "gave_number": first_int(text) is not None,
                "correct": bool(verdict),
                "reply": text,
            }
        )

    renderer = ctx.probe_builder.renderer
    texts = authored("tokenization_texts")[: ctx.arg("num_tokenization_texts", 40)]
    prompts, extras = [], []
    for record in texts:
        ids = renderer.encode_content(record["text"])
        pieces = renderer.texts(ids)
        prompts.append(
            f"How many tokens does the following text occupy in your own tokenizer? Reply with just the number.\n\nText: {record['text']}"
        )
        extras.append(("count", record["id"], len(ids), pieces))
        prompts.append(
            f"Split the following text into exactly the tokens your own tokenizer produces, in order, separated by ' | ' (space, bar, space). Reply with only the token sequence.\n\nText: {record['text']}"
        )
        extras.append(("split", record["id"], len(ids), pieces))
        k = rng.randrange(len(ids))
        prompts.append(
            f"What is the integer id of the token {renderer.text(ids[k])!r} in your vocabulary? Reply with just the number."
        )
        extras.append(("token_id", record["id"], ids[k], pieces))
    replies = ask(ctx, prompts, max_tokens, non_answers)
    numeric = sorted(i for i, (kind, *_) in enumerate(extras) if kind in ("count", "token_id"))
    values = judge.extract(
        grader, [(prompts[i], replies[i], "integer") for i in numeric], max_tokens=ctx.judge_max_tokens
    )
    extracted = dict(zip(numeric, values, strict=True))
    for i, ((kind, text_id, expected, pieces), reply) in enumerate(zip(extras, replies, strict=True)):
        row: dict = {"section": "tokenization", "kind": kind, "id": text_id, "reply": reply}
        if kind == "count":
            got = _as_int(extracted[i])
            row.update(expected=expected, got=got, correct=got == expected)
            row["relative_error"] = abs(got - expected) / expected if got is not None else None
        elif kind == "split":
            got = [p.strip() for p in reply.split("|")] if "|" in reply else reply.split()
            truth = [p.strip() for p in pieces if p.strip()]
            common = sum(min(got.count(t), truth.count(t)) for t in set(truth))
            precision = common / len(got) if got else 0.0
            recall = common / len(truth) if truth else 0.0
            row.update(
                expected=len(truth),
                got=len(got),
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            )
            row["correct"] = got == truth
        else:
            got = _as_int(extracted[i])
            row.update(expected=expected, got=got, correct=got == expected)
        rows.append(row)

    prompts, extras = [], []
    vocab = f["vocab_size"]
    used_ids = [rng.randrange(1000, f["tokenizer_size"]) for _ in range(ctx.arg("num_token_ids", 30))]
    unused = f["first_unused_token_id"]
    unused_ids = [rng.randrange(unused, vocab) for _ in range(ctx.arg("num_token_ids", 30))] if unused else []
    for token_id in used_ids + unused_ids:
        prompts.append(
            f"What text does token id {token_id} decode to in your vocabulary? Reply with just the text, in double quotes, or say that the id is unused if it has no text."
        )
        extras.append(token_id)
    replies = ask(ctx, prompts, max_tokens, non_answers)
    named = judge.extract(
        grader, [(q, r, "text") for q, r in zip(prompts, replies, strict=True)], max_tokens=ctx.judge_max_tokens
    )
    refusals = judge.yes_no(
        grader,
        [judge.REJECTION_PROMPT.format(question=q, answer=r) for q, r in zip(prompts, replies, strict=True)],
        max_tokens=ctx.judge_max_tokens,
    )
    for i, (token_id, reply) in enumerate(zip(extras, replies, strict=True)):
        is_unused = unused is not None and token_id >= unused
        expected = None if is_unused else renderer.text(token_id)
        got = str(named[i]) if named[i] is not None else None
        refused = bool(refusals[i])
        matches = got is not None and expected is not None and got.strip() == expected.strip()
        correct = refused if is_unused else matches
        rows.append(
            {
                "section": "token_ids",
                "kind": "unused" if is_unused else "used",
                "expected": expected,
                "got": got,
                "correct": correct,
                "declined": refused,
                "reply": reply,
            }
        )

    prompts, extras = [], []
    emoji = authored("emoji")
    rng.shuffle(emoji)
    for record in emoji[: ctx.arg("num_emoji", 80)]:
        prompts.append(
            f"Is there a {record['name']} emoji? Answer 'Yes' or 'No' first. If yes, output the emoji character itself right after your answer."
        )
        extras.append(record)
    for record, reply in zip(extras, ask(ctx, prompts, max_tokens, non_answers), strict=True):
        claim = yes_no(reply)
        exists = bool(record["exists"])
        char = unicodedata.lookup(record["unicode_name"]) if exists and record.get("unicode_name") else None
        emoji_chars = [c for c in reply if unicodedata.category(c) == "So"]
        rows.append(
            {
                "section": "emoji",
                "name": record["name"],
                "exists": exists,
                "claim": claim,
                "correct": claim == ("yes" if exists else "no"),
                "output_matches": char is not None and char in reply,
                "num_emoji_chars": len(emoji_chars),
                "reply": reply,
            }
        )

    def acc(section: str, **where) -> dict:
        subset = [r for r in rows if r["section"] == section and all(r.get(k) == v for k, v in where.items())]
        return {"n": len(subset), "acc": float(np.mean([r["correct"] for r in subset])) if subset else None}

    def acc_where(keep) -> dict:
        subset = [r for r in rows if r["section"] == "facts" and keep(r)]
        return {"n": len(subset), "acc": float(np.mean([r["correct"] for r in subset])) if subset else None}

    for row, category in zip([r for r in rows if "reply" in r], non_answers, strict=True):
        row["non_answer"] = category
    summary: dict = {
        "judge": ctx.config.judge_model,
        "truncated_replies": float(np.mean([c == "truncated" for c in non_answers])) if non_answers else 0.0,
        "non_answer": non_answer_summary(non_answers),
        "replies_opening_with_number": float(np.mean([opens_with_number(r["reply"]) for r in rows if "reply" in r])),
        "facts": {
            k: acc("facts", category=k) for k in sorted({r["category"] for r in rows if r["section"] == "facts"})
        },
        "facts_all": acc("facts"),
        "facts_parsed": float(
            np.mean([r.get("got") is not None for r in rows if r["section"] == "facts" and r["kind"] != "reject"])
        ),
        "facts_non_reject": {
            "n": len([r for r in rows if r["section"] == "facts" and r["kind"] != "reject"]),
            "acc": float(np.mean([r["correct"] for r in rows if r["section"] == "facts" and r["kind"] != "reject"])),
        },
        "trained_subdomains": trained,
        "facts_grounded": {
            domain: acc_where(lambda r, d=domain: r["kind"] != "reject" and d in r["contextual_for"])
            for domain in DOMAINS
        },
        "facts_ungrounded": acc_where(lambda r: r["kind"] != "reject" and not r["contextual_for"]),
        "facts_by_grounding": {
            g: acc_where(lambda r, g=g: r["kind"] != "reject" and r["grounding"] == g)
            for g in ("direct", "derived", "none")
        },
        "rejects_grounded": {
            domain: acc_where(lambda r, d=domain: r["kind"] == "reject" and d in r["contextual_for"])
            for domain in DOMAINS
        },
        "rejects_ungrounded": acc_where(lambda r: r["kind"] == "reject" and not r["contextual_for"]),
        "facts_contextual": acc_where(lambda r: r["kind"] != "reject" and r["contextual"]),
        "facts_not_contextual": acc_where(lambda r: r["kind"] != "reject" and not r["contextual"]),
        "impossible": {
            k: acc("impossible", kind=k) for k in sorted({r["kind"] for r in rows if r["section"] == "impossible"})
        },
        "impossible_all": {
            **acc("impossible"),
            "gave_number": float(np.mean([r["gave_number"] for r in rows if r["section"] == "impossible"])),
        },
        "tokenization": {k: acc("tokenization", kind=k) for k in ("count", "split", "token_id")},
        "tokenization_count_relative_error": float(
            np.mean(
                [
                    r["relative_error"]
                    for r in rows
                    if r["section"] == "tokenization" and r["kind"] == "count" and r["relative_error"] is not None
                ]
                or [np.nan]
            )
        ),
        "tokenization_split_f1": float(
            np.mean([r["f1"] for r in rows if r["section"] == "tokenization" and r["kind"] == "split"] or [np.nan])
        ),
        "token_ids": {k: acc("token_ids", kind=k) for k in ("used", "unused")},
        "emoji": {
            "existence_acc": acc("emoji")["acc"],
            "existing_acc": acc("emoji", exists=True)["acc"],
            "nonexistent_acc": acc("emoji", exists=False)["acc"],
            "output_matches_when_exists": float(
                np.mean([r["output_matches"] for r in rows if r["section"] == "emoji" and r["exists"]] or [np.nan])
            ),
            "false_yes_with_emoji_output": float(
                np.mean(
                    [
                        r["num_emoji_chars"] > 0
                        for r in rows
                        if r["section"] == "emoji" and not r["exists"] and r["claim"] == "yes"
                    ]
                    or [0.0]
                )
            ),
            "n": acc("emoji")["n"],
        },
    }
    return Result(summary=summary, rows=rows)
