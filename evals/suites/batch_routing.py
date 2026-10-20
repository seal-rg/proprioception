import copy
import dataclasses
import random

import numpy as np

from evals.datasets import held_out_prompts
from evals.probes import answer_questions, summarize
from evals.suites import Context, Result
from evals.suites.stability import exchanges

NAME = "batch_routing"
PATH = "server"


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    builder = ctx.probe_builder
    client = ctx.server
    num_convs = ctx.arg("num_conversations", 64)
    prompts = held_out_prompts(200, seed=ctx.seed + 1)
    convs = exchanges(ctx, prompts[:num_convs])
    fillers = exchanges(ctx, prompts[num_convs : num_convs + 32])
    filler_ids = [c.ids for c in fillers]
    conv_ids = [c.ids for c in convs]
    filler_prompts = [builder.start_prompt(c.messages[-2]["content"])[0] for c in fillers]

    solo = copy.copy(client)
    solo.concurrency = 1
    alone = solo.routing(conv_ids)
    repeats = ctx.arg("num_repeats", 3)
    busy = []
    for repeat in range(repeats):
        order = list(range(len(conv_ids)))
        rng.shuffle(order)
        prompts_busy = [conv_ids[i] for i in order] + [
            filler_ids[(i + repeat) % len(filler_ids)] for i in range(len(filler_ids))
        ]
        captures = client.routing(prompts_busy)
        busy.append({i: captures[k] for k, i in enumerate(order)})

    layers = ctx.family.moe_layers
    flips = {layer: [0, 0] for layer in layers}
    flipped_positions: list[tuple[int, int, int, int, int]] = []
    for i, ids in enumerate(conv_ids):
        for repeat in range(repeats):
            other = busy[repeat][i]
            for layer in layers:
                a = alone[i][: len(ids), layer, 0]
                b = other[: len(ids), layer, 0]
                diff = np.nonzero(a != b)[0]
                flips[layer][0] += int(diff.size)
                flips[layer][1] += len(ids)
                for pos in diff:
                    flipped_positions.append((i, repeat, layer, int(pos), int(b[pos])))
    summary: dict = {
        "flip_rate": {f"layer={layer}": flips[layer][0] / max(flips[layer][1], 1) for layer in layers},
        "flip_rate_all": sum(f[0] for f in flips.values()) / max(sum(f[1] for f in flips.values()), 1),
        "num_flipped_positions": len(flipped_positions),
    }

    rng.shuffle(flipped_positions)
    probes = []
    for i, _repeat, layer, pos, busy_label in flipped_positions[: ctx.arg("num_questions", 600)]:
        probe = builder.build(convs[i], rng=rng, type="routing", layer=layer, pos=pos, meta={"conversation": i})
        if probe is not None:
            probe.meta["alone_label"] = int(alone[i][pos, layer, 0])
            probe.meta["flipped_to"] = int(busy_label)
            probes.append(probe)
    if probes:
        padding = [filler_prompts[k % len(filler_prompts)] for k in range(len(probes))]
        answer_questions(client, builder, probes, max_tokens=ctx.max_tokens, seed=ctx.seed, load=padding)
        for probe in probes:
            probe.meta["live_label"] = probe.label
            probe.meta["matches_alone"] = probe.model_answer == probe.meta["alone_label"]
            probe.meta["matches_live"] = probe.model_answer == probe.label
            probe.meta["still_flipped"] = probe.label != probe.meta["alone_label"]
        still = [p for p in probes if p.meta["still_flipped"]]
        summary["flipped_questions"] = {
            "n": len(probes),
            "still_flipped_at_answer_time": len(still) / len(probes),
            "acc_vs_live": float(np.mean([p.meta["matches_live"] for p in probes])),
            "acc_vs_alone": float(np.mean([p.meta["matches_alone"] for p in probes])),
            "acc_vs_live_on_still_flipped": float(np.mean([p.meta["matches_live"] for p in still])) if still else None,
            "acc_vs_alone_on_still_flipped": float(np.mean([p.meta["matches_alone"] for p in still]))
            if still
            else None,
        }
        summary["flipped_questions_by_layer"] = summarize(probes, keys=["layer"])
        if ctx.arg("paired_solo", True):
            summary["paired_answering"] = paired_answering(solo, builder, probes, ctx)
    return Result(summary=summary, rows=[p.row() for p in probes])


def paired_answering(solo, builder, probes: list, ctx: Context) -> dict:
    clones = [
        dataclasses.replace(p, meta={}, label=None, model_answer=None, answer_text="", parsed=False, correct=False)
        for p in probes
    ]
    answer_questions(solo, builder, clones, max_tokens=ctx.max_tokens, seed=ctx.seed)
    for probe, clone in zip(probes, clones):
        probe.meta["solo_live_label"] = clone.label
        probe.meta["solo_answer"] = clone.model_answer

    def rates(rows: list) -> dict:
        if not rows:
            return {"n": 0}

        def share(fn) -> float:
            return float(np.mean([fn(p) for p in rows]))

        return {
            "n": len(rows),
            "names_flipped_to_busy": share(lambda p: p.model_answer == p.meta["flipped_to"]),
            "names_flipped_to_solo": share(lambda p: p.meta["solo_answer"] == p.meta["flipped_to"]),
            "names_alone_busy": share(lambda p: p.model_answer == p.meta["alone_label"]),
            "names_alone_solo": share(lambda p: p.meta["solo_answer"] == p.meta["alone_label"]),
            "answer_changed": share(lambda p: p.model_answer != p.meta["solo_answer"]),
            "flipped_to_only_busy": int(
                sum(p.model_answer == p.meta["flipped_to"] != p.meta["solo_answer"] for p in rows)
            ),
            "flipped_to_only_solo": int(
                sum(p.meta["solo_answer"] == p.meta["flipped_to"] != p.model_answer for p in rows)
            ),
        }

    switched = [
        p for p in probes if p.label == p.meta["flipped_to"] and p.meta["solo_live_label"] == p.meta["alone_label"]
    ]
    return {
        "solo_live_is_alone": float(np.mean([p.meta["solo_live_label"] == p.meta["alone_label"] for p in probes])),
        "busy_live_is_flipped_to": float(np.mean([p.label == p.meta["flipped_to"] for p in probes])),
        "all": rates(probes),
        "state_switched": rates(switched),
    }
