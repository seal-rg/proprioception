import contextlib
import itertools
from collections.abc import Iterator

import numpy as np
import torch

from evals.datasets import held_out_prompts
from evals.interventions import RouterEdit
from evals.suites import Context, Result
from proprioception.rendering import USER

NAME = "diagnostics_torch"
PATH = "torch"
MARGIN_BINS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, float("inf")))


def _top_and_margin(model, ids_list: list[list[int]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tops, margins = [], []
    for ids in ids_list:
        fwd = model.forward([ids])
        top = np.stack([fwd.routing[layer].selected[0, : len(ids), 0].cpu().numpy() for layer in model.moe_layers], 1)
        margin = np.stack(
            [
                _margin(fwd.routing[layer].router_logits[0, : len(ids)], model.semantics[layer], model.routers[layer])
                for layer in model.moe_layers
            ],
            1,
        )
        tops.append(top)
        margins.append(margin)
    return tops, margins


def _margin(router_logits: torch.Tensor, semantics, router) -> np.ndarray:
    scores = semantics.probabilities(router_logits.float())
    bias = getattr(router, "e_score_correction_bias", None)
    if bias is not None:
        scores = scores + bias.float()
    top2 = scores.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).cpu().numpy()


def _batched_top(model, ids_list: list[list[int]]) -> list[np.ndarray]:
    fwd = model.forward(ids_list)
    return [
        np.stack([fwd.routing[layer].selected[i, : len(ids), 0].cpu().numpy() for layer in model.moe_layers], 1)
        for i, ids in enumerate(ids_list)
    ]


def _compare(single: list[np.ndarray], batched: list[np.ndarray], margins: list[np.ndarray]) -> dict:
    s = np.concatenate(single)
    b = np.concatenate(batched)
    m = np.concatenate(margins)
    flips = s != b
    out = {"agreement": float(1 - flips.mean()), "flip_rate_by_margin": {}}
    for lo, hi in MARGIN_BINS:
        mask = (m >= lo) & (m < hi)
        if mask.any():
            out["flip_rate_by_margin"][f"[{lo},{hi})"] = {"flip_rate": float(flips[mask].mean()), "n": int(mask.sum())}
    return out


def _tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    return itertools.chain(module.named_parameters(), module.named_buffers())


@contextlib.contextmanager
def float32(module: torch.nn.Module) -> Iterator[None]:
    dtypes = {name: tensor.dtype for name, tensor in _tensors(module)}
    module.float()
    try:
        yield
    finally:
        for name, tensor in _tensors(module):
            tensor.data = tensor.data.to(dtypes[name])


@contextlib.contextmanager
def deterministic(model) -> Iterator[bool]:
    hf = model.model
    previous = getattr(hf.config, "_experts_implementation", None)
    enabled, warn_only = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        hf.set_experts_implementation("eager")
    except (ValueError, AttributeError):
        yield False
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield True
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        hf.set_experts_implementation(previous)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _edit_vs_clean(model, ids_list: list[list[int]]) -> dict:
    clean = model.forward(ids_list)
    with model.edits(router=[RouterEdit(layer=layer) for layer in model.moe_layers]):
        edited = model.forward(ids_list)
    top1, topk, identical, max_diff = [], [], [], 0.0
    for i, n in enumerate(clean.lengths):
        a, b = clean.logits[i, :n].float(), edited.logits[i, :n].float()
        identical.append(bool(torch.equal(a, b)))
        max_diff = max(max_diff, float((a - b).abs().max()))
        for layer in model.moe_layers:
            sa, sb = clean.routing[layer].selected[i, :n], edited.routing[layer].selected[i, :n]
            top1.append((sa[:, 0] == sb[:, 0]).float().mean().item())
            topk.append((sa.sort(-1).values == sb.sort(-1).values).all(-1).float().mean().item())
    return {
        "logits_identical": float(np.mean(identical)),
        "max_abs_logit_diff": max_diff,
        "top1_agreement": float(np.mean(top1)),
        "topk_set_agreement": float(np.mean(topk)),
    }


def _generation(model, prompts: list[list[int]], max_new_tokens: int) -> dict:
    clean = model.generate(prompts, max_new_tokens=max_new_tokens)
    again = model.generate(prompts, max_new_tokens=max_new_tokens)
    with model.edits(router=[RouterEdit(layer=layer) for layer in model.moe_layers]):
        edited = model.generate(prompts, max_new_tokens=max_new_tokens)
    out = {}
    for name, other in (("clean_vs_clean", again), ("same_vs_clean", edited)):
        points = [first_divergence(a, b) for a, b in zip(clean, other, strict=True)]
        diverged = [p for p in points if p is not None]
        out[name] = {
            "identical": float(np.mean([p is None for p in points])),
            "first_divergence": points,
            "median_first_divergence": float(np.median(diverged)) if diverged else None,
        }
    return out


def _pass(
    model, ids_list: list[list[int]], same_length: list[list[int]], prompts, max_new_tokens: int
) -> tuple[dict, list[np.ndarray]]:
    single, margins = _top_and_margin(model, ids_list)
    single_same, margins_same = _top_and_margin(model, same_length)
    out = {
        "bf16_padded_batch_vs_single": _compare(single, _batched_top(model, ids_list), margins),
        "bf16_same_length_batch_vs_single": _compare(single_same, _batched_top(model, same_length), margins_same),
        "same_edit_vs_clean_forward": _edit_vs_clean(model, ids_list),
    }
    if prompts:
        out["generation"] = _generation(model, prompts, max_new_tokens)
    return out, single


def _mean(passes: list[dict], *path: str) -> float | None:
    values = []
    for entry in passes:
        for key in path:
            entry = entry.get(key) if isinstance(entry, dict) else None
        if isinstance(entry, (int, float)):
            values.append(float(entry))
    return float(np.mean(values)) if values else None


def run(ctx: Context) -> Result:
    model = ctx.torch_model
    builder = ctx.probe_builder
    prompts = held_out_prompts(ctx.arg("num_sequences", 8), seed=ctx.seed + 9)
    ids_list = [builder.renderer.render([{"role": USER, "content": p}]).token_ids for p in prompts]
    common = min(len(ids) for ids in ids_list)
    same_length = [ids[:common] for ids in ids_list]
    gen_prompts = [
        builder.renderer.render([{"role": USER, "content": p}], add_generation_prompt=True).token_ids
        for p in prompts[: ctx.arg("num_generations", 8)]
    ]
    max_new_tokens = ctx.max_tokens
    summary: dict = {}

    passes, singles = [], []
    for _ in range(ctx.arg("repeats", 3)):
        entry, single = _pass(model, ids_list, same_length, gen_prompts, max_new_tokens)
        passes.append(entry)
        singles.append(single)
    _, margins = _top_and_margin(model, ids_list)
    stacked = np.concatenate(margins)
    summary["mean_top1_margin_per_layer"] = {
        f"layer={layer}": float(stacked[:, i].mean()) for i, layer in enumerate(model.moe_layers)
    }
    summary["passes"] = passes
    for key in ("bf16_padded_batch_vs_single", "bf16_same_length_batch_vs_single"):
        summary[key] = {
            "agreement": _mean(passes, key, "agreement"),
            "flip_rate_by_margin": {
                f"[{lo},{hi})": {"flip_rate": _mean(passes, key, "flip_rate_by_margin", f"[{lo},{hi})", "flip_rate")}
                for lo, hi in MARGIN_BINS
            },
        }
    summary["same_edit_vs_clean_forward"] = {
        k: _mean(passes, "same_edit_vs_clean_forward", k)
        for k in ("logits_identical", "max_abs_logit_diff", "top1_agreement", "topk_set_agreement")
    }
    if gen_prompts:
        summary["generation"] = {
            name: {"identical": _mean(passes, "generation", name, "identical")}
            for name in ("clean_vs_clean", "same_vs_clean")
        }
    summary["bf16_single_repeat_identical"] = float(
        np.mean([np.array_equal(a, b) for s in singles[1:] for a, b in zip(singles[0], s, strict=True)])
        if len(singles) > 1
        else 1.0
    )

    if ctx.arg("deterministic", True):
        with deterministic(model) as switched:
            if switched:
                summary["deterministic"], _ = _pass(model, ids_list, same_length, gen_prompts, max_new_tokens)
            else:
                summary["deterministic"] = "experts implementation cannot be switched for this model"
    if ctx.arg("fp32", True):
        with float32(model.model):
            single32, margins32 = _top_and_margin(model, ids_list)
            summary["fp32_padded_batch_vs_single"] = _compare(single32, _batched_top(model, ids_list), margins32)
            summary["fp32_vs_bf16_single"] = _compare(single32, singles[0], margins32)
    after, _ = _top_and_margin(model, ids_list)
    summary["state_restored"] = all(np.array_equal(a, b) for a, b in zip(singles[0], after, strict=True))

    verdict = {}
    for key in ("bf16_padded_batch_vs_single", "bf16_same_length_batch_vs_single", "fp32_padded_batch_vs_single"):
        if key not in summary:
            continue
        large = summary[key]["flip_rate_by_margin"].get("[0.2,inf)", {}).get("flip_rate") or 0.0
        verdict[key] = "ok" if large < 0.01 else "FAULT: flips at large margins"
    if summary["bf16_single_repeat_identical"] < 1.0:
        verdict["bf16_single_repeat"] = "FAULT: nondeterministic solo forward"
    if (summary["same_edit_vs_clean_forward"]["topk_set_agreement"] or 0.0) < 0.99:
        verdict["same_edit"] = "FAULT: the no-op router edit selects different experts"
    if not summary["state_restored"]:
        verdict["state_restored"] = "FAULT: routing after the deterministic and fp32 branches differs from before"
    summary["verdicts"] = verdict
    return Result(summary=summary, rows=[])
