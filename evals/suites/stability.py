import hashlib
import json
import random
from pathlib import Path

import numpy as np

from evals import binary
from evals.datasets import carriers_by_bucket, held_out_prompts
from evals.lookup import LookupTable
from evals.probes import Conversation, Probe, answer_questions, needs_torch_labels, summarize
from evals.suites import Context, Result
from evals.truth import Request
from proprioception.batching import partner_span
from proprioception.config import EvalConfig as LookupDefaults
from proprioception.evaluation import PairLabelCounts, probe_pair
from proprioception.formats import KNOWN_TYPES

LOOKUP_DEFAULTS = LookupDefaults()

NAME = "stability"
PATH = "server"


def exchanges(ctx: Context, prompts: list[str], *, max_tokens: int = 0, aligned: bool = False) -> list:
    return ctx.probe_builder.exchanges(
        ctx.server,
        prompts,
        max_tokens=max_tokens or ctx.max_tokens,
        temperature=ctx.arg("reply_temperature", 0.7),
        seed=ctx.seed,
        carrier=ctx.config.carrier,
        aligned=aligned,
    )


def _joined(ctx: Context, parts: list[Conversation]) -> Conversation:
    messages = [m for c in parts for m in c.messages if m["role"] != "system"]
    return ctx.probe_builder.from_messages(messages)


def _matrix(ctx: Context, rng: random.Random, convs: list[Conversation], per_type: int) -> list[Probe]:
    probes = []
    for type in ctx.config.requested_types or KNOWN_TYPES:
        for phrasing in ("train", "novel"):
            count = 0
            attempts = 0
            while count < per_type and attempts < per_type * 4:
                attempts += 1
                conv = rng.choice(convs)
                probe = ctx.probe_builder.build(
                    conv, rng=rng, type=type, phrasing=phrasing, meta={"axis": "matrix", "phrasing": phrasing}
                )
                if probe is not None:
                    probes.append(probe)
                    count += 1
    return probes


def _turns(ctx: Context, rng: random.Random, convs, fillers, types, num: int, max_depth: int) -> list[Probe]:
    probes = []
    for depth in range(max_depth + 1):
        for target in ("first", "latest"):
            for _ in range(num):
                first = rng.choice(convs)
                later = [rng.choice(fillers) for _ in range(depth)]
                conversation = _joined(ctx, [first, *later]) if later else first
                turn_pair = 0 if target == "first" else depth
                turn = 2 * turn_pair + rng.randint(0, 1)
                probe = ctx.probe_builder.build(
                    conversation,
                    rng=rng,
                    type=rng.choice(types),
                    turn=turn,
                    meta={"axis": "turns", "depth": depth, "target": target, "prefix_tokens": 0},
                )
                if probe is not None:
                    probe.meta["prefix_tokens"] = len(probe.prefix_ids)
                    probes.append(probe)
    return probes


def _context_length(ctx: Context, rng: random.Random, prompts, types, num: int) -> list[Probe]:
    probes = []
    for num_parts in (1, 2, 4, 8, 16):
        batch_prompts = []
        for _ in range(num):
            parts = rng.sample(prompts, num_parts)
            batch_prompts.append(("\n\n".join(parts), parts))
        conversations = exchanges(ctx, [p for p, _ in batch_prompts], aligned=True)
        for (_, parts), conversation in zip(batch_prompts, conversations, strict=True):
            if conversation is None:
                continue
            for target in ("first", "last"):
                probe = ctx.probe_builder.build(
                    conversation,
                    rng=rng,
                    type=rng.choice(types),
                    turn=0,
                    meta={"axis": "context", "num_parts": num_parts, "target": target, "distance": 0},
                )
                if probe is None:
                    continue
                start, end = probe.span
                boundary = start + len(ctx.probe_builder.renderer.encode_content(parts[0]))
                tail = end - len(ctx.probe_builder.renderer.encode_content(parts[-1]))
                pool = range(start, min(boundary, end)) if target == "first" else range(max(tail, start), end)
                pool = [i for i in pool if ctx.probe_builder.renderer.text(probe.prefix_ids[i]).strip()]
                if not pool:
                    continue
                moved = ctx.probe_builder.build(
                    conversation,
                    rng=rng,
                    type=probe.type,
                    turn=0,
                    pos=rng.choice(pool),
                    layer=probe.layer,
                    meta=probe.meta,
                )
                if moved is not None:
                    moved.meta["distance"] = len(moved.prefix_ids) - moved.pos
                    probes.append(moved)
    return probes


def _recursion(ctx: Context, rng: random.Random, convs, types, num: int, depth: int) -> list[Probe]:
    level_questions: list[Probe] = []
    current: list[Conversation] = [rng.choice(convs) for _ in range(num)]
    for level in range(depth + 1):
        probes = []
        for conversation in current:
            turn = conversation.num_turns - 1 if level > 0 else None
            probe = ctx.probe_builder.build(
                conversation, rng=rng, type=rng.choice(types), turn=turn, meta={"axis": "recursion", "level": level}
            )
            if probe is not None:
                probes.append(probe)
        answer_questions(ctx.server, ctx.probe_builder, probes, max_tokens=ctx.max_tokens, seed=ctx.seed)
        level_questions.extend(probes)
        current = [ctx.probe_builder.with_answer(p) for p in probes if ctx.probe_builder.visible_answer(p).strip()]
        if not current:
            break
    return level_questions


def _language(ctx: Context, rng: random.Random, types, num: int) -> list[Probe]:
    probes = []
    for language, prompts in carriers_by_bucket().items():
        convs = exchanges(ctx, prompts[:12])
        for _ in range(num):
            probe = ctx.probe_builder.build(
                rng.choice(convs), rng=rng, type=rng.choice(types), meta={"axis": "language", "language": language}
            )
            if probe is not None:
                probes.append(probe)
    return probes


def _conv_key(probe: Probe) -> str:
    user = next((m["content"] for m in probe.messages if m["role"] == "user"), "")
    return hashlib.sha256(user.encode()).hexdigest()


def _training_samples(ctx: Context) -> list[dict] | None:
    path = ctx.arg("train_dataset", "")
    if not path:
        return None
    train = Path(path) / "train.json"
    if not train.exists():
        ctx.log(f"train_dataset {train} not found; pair-seen and lookup splits skipped")
        return None
    return json.loads(train.read_text())


def _modal_labels(ctx: Context, samples: list[dict]) -> PairLabelCounts:
    requests = [
        [
            Request(
                type=s.get("type", "routing"),
                pos=s["probe_pos"],
                layer=s.get("layer", -1),
                rank=s.get("rank", 1),
                span=partner_span(s),
            )
        ]
        for s in samples
    ]
    ends = [max(s["probe_pos"] + 2, partner_span(s)[1]) for s in samples]
    labels = ctx.torch_labeler.label([s["input_ids"][:end] for s, end in zip(samples, ends, strict=True)], requests)
    counts = PairLabelCounts(
        decay=LOOKUP_DEFAULTS.lexical_lookup_decay, min_obs=LOOKUP_DEFAULTS.lexical_lookup_min_count
    )
    for sample, (label,) in zip(samples, labels, strict=True):
        if label is not None:
            counts.update(probe_pair(sample), label)
    ctx.log(f"lookup: {len(samples)} training samples labelled, {len(counts.counts)} keys")
    return counts


def _lookup_metrics(probes: list[Probe], counts: PairLabelCounts) -> dict:
    out: dict = {}
    for type in sorted({p.type for p in probes}):
        rows = [p for p in probes if p.type == type and p.label is not None]
        modal = [counts.modal((p.token_id, p.layer, p.type, p.rank)) for p in rows]
        known = [(p, m) for p, m in zip(rows, modal, strict=True) if m is not None]
        at_modal = [p for p, m in known if m == p.label]
        ctx_rows = [p for p, m in known if m != p.label]
        out[f"type={type}"] = {
            "n_known": len(known),
            "lookup_baseline_acc": len(at_modal) / len(known) if known else None,
            "acc_label_equals_modal": float(np.mean([p.correct for p in at_modal])) if at_modal else None,
            "ctx_acc": float(np.mean([p.correct for p in ctx_rows])) if ctx_rows else None,
            "ctx_n": len(ctx_rows),
        }
    return out


def _base_agreement(ctx: Context, probes: list[Probe]) -> dict:
    if ctx.model.role == "base":
        return {}
    other = None
    for name, spec in ctx.config.models.items():
        if spec.role == "base":
            path = ctx.out_dir.parent / name / "stability.json"
            if path.exists():
                other = json.loads(path.read_text())
    if other is None:
        return {}
    base_labels = {
        (r["conv_key"], r["pos"], r["layer"], r["rank"], r["type"]): r["label"]
        for r in other["rows"]
        if r.get("role") == "user" and r.get("axis") == "matrix" and r.get("label") is not None
    }
    matched = [
        (p, base_labels[(_conv_key(p), p.pos, p.layer, p.rank, p.type)])
        for p in probes
        if p.role == "user"
        and p.meta["axis"] == "matrix"
        and p.label is not None
        and (_conv_key(p), p.pos, p.layer, p.rank, p.type) in base_labels
    ]
    if not matched:
        return {}
    builder = ctx.probe_builder
    changed = [p for p, b in matched if b != p.label]
    kept = [p for p, b in matched if b == p.label]
    return {
        "vs_base": {
            "n": len(matched),
            "label_agreement": len(kept) / len(matched),
            "acc_on_changed": float(np.mean([p.correct for p in changed])) if changed else None,
            "acc_on_kept": float(np.mean([p.correct for p in kept])) if kept else None,
            "acc_against_base_label_on_changed": float(
                np.mean([builder.matches(p.type, b, p.model_answer) for p, b in matched if b != p.label])
            )
            if changed
            else None,
        }
    }


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    per_type = ctx.arg("num_questions_per_type", 100)
    num_axis = ctx.arg("num_axis_questions", 60)
    max_depth = ctx.arg("max_depth", 6)
    recursion_depth = ctx.arg("recursion_depth", 3)
    types = ctx.types
    prompts = held_out_prompts(200, seed=ctx.seed)
    convs = exchanges(ctx, prompts[:120])
    fillers = exchanges(ctx, prompts[120:])
    ctx.log(f"{len(convs)} carrier and {len(fillers)} filler exchanges usable of {len(prompts)} prompts")

    probes: list[Probe] = []
    probes += _matrix(ctx, rng, convs, per_type)
    probes += _turns(ctx, rng, convs, fillers, types, max(num_axis // 4, 8), max_depth)
    probes += _context_length(ctx, rng, prompts, types, max(num_axis // 4, 8))
    probes += _language(ctx, rng, types, num_axis)
    ctx.log(f"{len(probes)} probes built; answering")
    lookup = LookupTable(list(ctx.family.moe_layers))

    def sink(probe: Probe, routed) -> None:
        lookup.add(hash(tuple(probe.prefix_ids[: probe.conv_len])), probe.prefix_ids, routed, probe.conv_len)

    for type in sorted({p.type for p in probes}):
        group = [p for p in probes if p.type == type]
        answer_questions(
            ctx.server,
            ctx.probe_builder,
            group,
            max_tokens=ctx.max_tokens,
            seed=ctx.seed,
            capture_sink=sink,
        )
    probes += _recursion(ctx, rng, convs, types, num_axis, recursion_depth)
    binary_questions = []
    if "routing" in types:
        binary_questions = binary.build(
            ctx.probe_builder,
            ctx.server,
            prompts,
            rng=rng,
            num=ctx.arg("num_binary_questions", per_type),
            max_tokens=ctx.max_tokens,
        )
    train_samples = _training_samples(ctx)
    seen_pairs = lookup_samples = None
    if train_samples is not None:
        seen_pairs = {
            (t["input_ids"][t["probe_pos"]], t.get("layer", -1), t.get("type", "routing")) for t in train_samples
        }
        lookup_samples = [t for t in train_samples if t.get("type", "routing") in set(types)]
        random.Random(ctx.seed).shuffle(lookup_samples)
        lookup_samples = lookup_samples[: ctx.arg("num_lookup_samples", 60_000)]
        del train_samples

    def finish(all_questions: list[Probe]) -> Result:
        builder = ctx.probe_builder
        summary: dict = {"carriers_dropped": builder.dropped_carriers / max(builder.total_carriers, 1)}
        matrix = [p for p in all_questions if p.meta["axis"] == "matrix"]
        summary["matrix"] = {
            k: v
            for k, v in summarize([p for p in matrix if p.meta["phrasing"] == "train"]).items()
            if k.startswith("type=")
        }
        summary["matrix_novel_phrasing"] = {
            k: v
            for k, v in summarize([p for p in matrix if p.meta["phrasing"] == "novel"]).items()
            if k.startswith("type=")
        }
        summary["matrix_word_initial"] = {
            k: v for k, v in summarize(matrix, keys=["word_initial"]).items() if k.startswith("word_initial=")
        }
        summary["turns"] = summarize([p for p in all_questions if p.meta["axis"] == "turns"], keys=["depth", "target"])
        summary["context"] = summarize(
            [p for p in all_questions if p.meta["axis"] == "context"], keys=["num_parts", "target"]
        )
        summary["recursion"] = summarize([p for p in all_questions if p.meta["axis"] == "recursion"], keys=["level"])
        summary["language"] = summarize([p for p in all_questions if p.meta["axis"] == "language"], keys=["language"])
        summary["domain"] = summarize([p for p in all_questions if p.meta["axis"] != "matrix"], keys=["role"])
        summary["matrix_by_layer"] = {
            k: v for k, v in summarize(matrix, keys=["layer"]).items() if k.startswith("layer=")
        }
        if ctx.config.carrier == "exact":
            summary["channel"] = {
                k: v for k, v in summarize(all_questions, keys=["channel"]).items() if "channel=" in k
            }
        summary["lexicality"] = lookup.lexicality()
        if lookup_samples and ctx.labeler is not None:
            counts = _modal_labels(ctx, lookup_samples)
            summary["lookup_matrix"] = _lookup_metrics(matrix, counts)
            summary["lookup_conversation"] = _lookup_metrics(
                [p for p in all_questions if p.meta["axis"] != "matrix"], counts
            )
        if binary_questions:
            summary["binary"] = binary.summarize(binary_questions)
        if seen_pairs is not None:
            for p in all_questions:
                p.meta["pair_seen"] = (p.token_id, p.layer, p.type) in seen_pairs
            summary["pair_seen"] = {
                k: v for k, v in summarize(matrix, keys=["pair_seen"]).items() if k.startswith("pair_seen=")
            }
        summary.update(_base_agreement(ctx, all_questions))
        compared = [p for p in all_questions if "torch_label" in p.meta]
        if compared:
            summary["label_agreement_torch_vs_server"] = {
                f"type={t}": float(np.mean([p.meta["torch_label"] == p.label for p in compared if p.type == t]))
                for t in sorted({p.type for p in compared})
            }
        rows = [{**p.row(), "conv_key": _conv_key(p)} for p in all_questions] + [b.row() for b in binary_questions]
        return Result(summary=summary, rows=rows)

    compare = [p for p in probes if p.meta["axis"] == "matrix" and p.label is not None][: ctx.arg("num_compare", 200)]
    if ctx.arg("skip_torch", False) and not needs_torch_labels(probes):
        return finish(probes)
    return Result(pending=[p for p in probes if p.label is None], compare=compare, finish=lambda pend: finish(probes))
