import random

import numpy as np

from evals.datasets import held_out_prompts
from evals.suites import Context, Result

NAME = "diagnostics"
PATH = "server"


def _agreement(a: np.ndarray, b: np.ndarray, layers: list[int]) -> dict[str, float]:
    per_layer = {f"layer={layer}": float((a[:, layer, 0] == b[:, layer, 0]).mean()) for layer in layers}
    per_layer["all"] = float(np.mean([per_layer[f"layer={layer}"] for layer in layers]))
    return per_layer


def capture_diversity(capture: np.ndarray, layers: list[int]) -> dict:
    valid = capture[(capture[:, layers] >= 0).all(axis=(1, 2))][:, layers]
    distinct = [len(np.unique(valid[:, i, 0])) for i in range(len(layers))]
    ordered = np.sort(valid, axis=-1)
    repeated = (ordered[..., 1:] == ordered[..., :-1]).any(axis=-1)
    return {
        "num_rows": int(valid.shape[0]),
        "min_distinct_top1": int(min(distinct)) if distinct else 0,
        "mean_distinct_top1": float(np.mean(distinct)) if distinct else 0.0,
        "repeated_expert_fraction": float(repeated.mean()) if repeated.size else 0.0,
    }


def capture_is_plausible(diversity: dict) -> bool:
    return (
        diversity["num_rows"] > 1 and diversity["min_distinct_top1"] > 1 and diversity["repeated_expert_fraction"] == 0
    )


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    builder = ctx.probe_builder
    client = ctx.server
    solo = client.__class__(client.base_url, client.model, concurrency=1)
    layers = ctx.family.moe_layers
    num = ctx.arg("num_conversations", 16)
    prompts = held_out_prompts(num + 32, seed=ctx.seed + 7)
    convs = builder.exchanges(client, prompts[:num], max_tokens=ctx.max_tokens, seed=ctx.seed, carrier="exact")
    if not convs:
        return Result(summary={"note": f"no carrier reply had visible content within the budget ({num} prompts)"})
    fillers = [builder.start_prompt(p)[0] for p in prompts[num:]]
    ids = [c.ids for c in convs]
    summary: dict = {}
    rows: list[dict] = []

    first = solo.routing(ids)
    second = solo.routing(ids)
    identical = [bool(np.array_equal(a[:, layers], b[:, layers])) for a, b in zip(first, second, strict=True)]
    summary["solo_determinism"] = {
        "identical_fraction": float(np.mean(identical)),
        "agreement": _agreement(np.concatenate(first), np.concatenate(second), layers)["all"],
    }

    order = list(range(len(ids)))
    rng.shuffle(order)
    busy_captures = client.routing([ids[i] for i in order] + fillers)
    busy: dict[int, np.ndarray] = {i: busy_captures[k] for k, i in enumerate(order)}
    batch = _agreement(np.concatenate(first), np.concatenate([busy[i] for i in range(len(ids))]), layers)
    summary["batch_variance"] = {"agreement": batch["all"], "per_layer": {k: v for k, v in batch.items() if k != "all"}}

    gen_rows, pre_rows = [], []
    for conv, capture in zip(convs, first, strict=True):
        assert conv.routing is not None
        start, end, _ = conv.spans[-1]
        valid = (conv.routing[start:end] >= 0).all(axis=(1, 2))
        gen_rows.append(conv.routing[start:end][valid])
        pre_rows.append(capture[start:end][valid])
    decode = _agreement(np.concatenate(gen_rows), np.concatenate(pre_rows), layers)
    summary["prefill_vs_decode"] = {
        "agreement": decode["all"],
        "per_layer": {k: v for k, v in decode.items() if k != "all"},
        "gap_to_batch_variance": batch["all"] - decode["all"],
    }
    prompt_gen, prompt_pre = [], []
    for conv, capture in zip(convs, first, strict=True):
        assert conv.routing is not None
        end = conv.spans[-1][0]
        prompt_gen.append(conv.routing[:end])
        prompt_pre.append(capture[:end])
    summary["prefill_vs_decode"]["prompt_prefill_agreement_same_request"] = _agreement(
        np.concatenate(prompt_gen), np.concatenate(prompt_pre), layers
    )["all"]

    client.generate(fillers, max_tokens=32, temperature=1.0, seed=ctx.seed)
    third = solo.routing(ids)
    summary["context_pollution"] = {
        "identical_fraction": float(
            np.mean([np.array_equal(a[:, layers], b[:, layers]) for a, b in zip(first, third, strict=True)])
        ),
    }

    extended = [c.ids + builder.renderer.encode_content(" " + rng.choice(prompts)) for c in convs]
    fourth = solo.routing(extended)
    suffix = _agreement(
        np.concatenate(first), np.concatenate([f[: len(i)] for f, i in zip(fourth, ids, strict=True)]), layers
    )
    summary["suffix_dependence"] = {
        "agreement": suffix["all"],
        "identical_fraction": float(
            np.mean(
                [
                    np.array_equal(a[:, layers], f[: len(i)][:, layers])
                    for a, f, i in zip(first, fourth, ids, strict=True)
                ]
            )
        ),
    }

    summary["capture_diversity"] = capture_diversity(np.concatenate(first), layers)

    for name, entry in summary.items():
        rows.append({"check": name, **{k: v for k, v in entry.items() if not isinstance(v, dict)}})
    verdicts = {
        "solo_determinism": "ok" if summary["solo_determinism"]["identical_fraction"] == 1.0 else "FAULT",
        "context_pollution": "ok" if summary["context_pollution"]["identical_fraction"] == 1.0 else "FAULT",
        "prefill_vs_decode": "ok" if summary["prefill_vs_decode"]["gap_to_batch_variance"] < 0.03 else "suspicious",
        "suffix_dependence": "ok" if summary["suffix_dependence"]["agreement"] >= batch["all"] - 0.03 else "suspicious",
        "capture_diversity": "ok" if capture_is_plausible(summary["capture_diversity"]) else "FAULT",
    }
    summary["verdicts"] = verdicts
    return Result(summary=summary, rows=rows)
