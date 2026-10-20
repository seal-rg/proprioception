import dataclasses
import math
import random

import numpy as np

from evals import judge
from evals.client import Generation, non_answer_summary
from evals.datasets import held_out_prompts
from evals.interventions import LayerEdit, RouterEdit
from evals.probes import Probe, ProbeBuilder
from evals.suites import Context, Result
from evals.torchmodel import TorchModel
from evals.truth import TorchLabeler
from proprioception.formats import SCALAR_TYPES
from proprioception.oracle import ends_inside_reasoning, visible_assistant_content
from proprioception.rendering import ASSISTANT, USER

NAME = "ablation"
PATH = "torch"
GROUP = 8
MODES = ("think", "nothink")
EXPERT_VALUED_TYPES = ("routing", "routing_rank")


@dataclasses.dataclass
class Condition:
    name: str
    router: list[RouterEdit]
    layer: list[LayerEdit]
    ablated: tuple[int, ...] = ()
    forced: int | None = None


def _generate(
    model: TorchModel, builder: ProbeBuilder, probes: list[Probe], max_new_tokens: int, thinking: str | None = None
) -> None:
    outs = model.generate(
        [p.prefix_ids for p in probes], max_new_tokens=max_new_tokens, bad_words_ids=builder.banned_ids(thinking)
    )
    for probe, ids in zip(probes, outs, strict=True):
        finish = "length" if len(ids) >= max_new_tokens else "stop"
        builder.read_answer(
            probe, Generation(token_ids=ids, finish_reason=finish, routed_experts=None, logprobs=[]), thinking
        )


def _label(labeler: TorchLabeler, probes: list[Probe]) -> list[int]:
    return [lab[0] for lab in labeler.label([p.prefix_ids for p in probes], [[p.request] for p in probes])]


def _top_experts(labeler: TorchLabeler, probes: list[Probe]) -> list[int]:
    requests = [[dataclasses.replace(p.request, type="routing", rank=1)] for p in probes]
    return [lab[0] for lab in labeler.label([p.prefix_ids for p in probes], requests)]


def _replies(model: TorchModel, builder: ProbeBuilder, prompts: list[str], max_new_tokens: int) -> list[list[dict]]:
    convs = []
    for start in range(0, len(prompts), GROUP):
        chunk = prompts[start : start + GROUP]
        ids = [
            builder.renderer.render([{"role": USER, "content": p}], add_generation_prompt=True).token_ids for p in chunk
        ]
        for prompt, out in zip(chunk, model.generate(ids, max_new_tokens=max_new_tokens), strict=True):
            reply = builder.assembler.carrier_reply(out)
            if reply:
                convs.append([{"role": USER, "content": prompt}, {"role": ASSISTANT, "content": reply}])
    return convs


def _routing_conditions(
    rng: random.Random, layer: int, clean_labels: list[int], family, noise: float
) -> list[Condition]:
    hit = sorted(set(clean_labels))
    others = [e for e in range(family.num_experts) if e not in hit]
    size = max(GROUP, len(hit))
    ablate_hit = tuple(sorted(hit + rng.sample(others, size - len(hit))))
    ablate_miss = tuple(sorted(rng.sample(others, min(size, len(others)))))
    forced = rng.choice(others)
    return [
        Condition("clean", [], []),
        Condition("same", [RouterEdit(layer=layer)], []),
        Condition("ablate_hit", [RouterEdit(layer=layer, ablate=ablate_hit)], [], ablated=ablate_hit),
        Condition("ablate_miss", [RouterEdit(layer=layer, ablate=ablate_miss)], [], ablated=ablate_miss),
        Condition("force", [RouterEdit(layer=layer, force_top=forced)], [], forced=forced),
        Condition("noise", [RouterEdit(layer=layer, noise=noise, seed=rng.randrange(1 << 30))], []),
    ]


def _generic_conditions(rng: random.Random, family, noise: float) -> list[Condition]:
    middle = family.moe_layers[len(family.moe_layers) // 2]
    skipped = rng.choice(family.moe_layers[2:-2] if len(family.moe_layers) > 4 else family.moe_layers)
    return [
        Condition("clean", [], []),
        Condition("same", [RouterEdit(layer=middle)], [LayerEdit(layer=middle)]),
        Condition("skip_layer", [], [LayerEdit(layer=skipped, skip=True)]),
        Condition("residual_noise", [], [LayerEdit(layer=middle, noise=0.5, seed=rng.randrange(1 << 30))]),
        Condition(
            "router_noise_all",
            [RouterEdit(layer=l, noise=noise, seed=rng.randrange(1 << 30)) for l in family.moe_layers],
            [],
        ),
    ]


def _which_experts(
    model: TorchModel,
    builder: ProbeBuilder,
    probes: list[Probe],
    condition: Condition,
    max_new: int,
    ctx: Context,
    thinking: str | None = None,
) -> list[dict]:
    rows = []
    prompts = []
    renderer = builder.mode(thinking)[0]
    answers = {id(p): builder.visible_answer(p, thinking) for p in probes}
    probes = [p for p in probes if answers[id(p)].strip()]
    for probe in probes:
        question = (
            f"One more question about that same layer {probe.layer}: some of your experts were disabled there while "
            "you read and answered. Which expert numbers were disabled? List every number you believe was disabled, "
            "separated by commas."
        )
        messages = [
            *probe.messages,
            {"role": ASSISTANT, "content": answers[id(probe)]},
            {"role": USER, "content": question},
        ]
        prompts.append(renderer.render(messages, add_generation_prompt=True).token_ids)
    outs = model.generate(prompts, max_new_tokens=max_new, bad_words_ids=builder.banned_ids(thinking))
    markers = builder.family.reasoning
    generations = [
        (markers.open if ends_inside_reasoning(renderer.decode(prompt), markers) else "") + renderer.decode(ids)
        for prompt, ids in zip(prompts, outs, strict=True)
    ]
    visibles = [visible_assistant_content(g, markers) for g in generations]
    extracted = judge.extract(
        ctx.require_judge(), [(question, v, "integers") for v in visibles], max_tokens=ctx.judge_max_tokens
    )
    for generation, visible, listed in zip(generations, visibles, extracted, strict=True):
        guessed = {
            int(e) for e in (listed or []) if str(e).lstrip("-").isdigit() and 0 <= int(e) < builder.family.num_experts
        }
        truth = set(condition.ablated)
        jaccard = len(guessed & truth) / len(guessed | truth) if guessed | truth else 0.0
        rows.append(
            {
                "guessed": sorted(guessed),
                "truth": sorted(truth),
                "jaccard": jaccard,
                "num_guessed": len(guessed),
                "text": visible,
                "generation": generation,
            }
        )
    return rows


def _upper_tail(successes: int, probabilities: list[float]) -> float:
    distribution = [1.0]
    for p in probabilities:
        shifted = [0.0, *distribution]
        distribution = [(1 - p) * a + p * b for a, b in zip([*distribution, 0.0], shifted, strict=True)]
    return min(1.0, math.fsum(distribution[successes:]))


def _chance_vs_new(row: dict, num_experts: int) -> float:
    return 1 / (num_experts - row["num_ablated"])


def condition_summary(subset: list[dict], name: str, num_experts: int) -> dict:
    moved = [r for r in subset if r["label_moved"]]
    entry = {
        "n": len(subset),
        "label_moved": float(np.mean([r["label_moved"] for r in subset])),
        "acc_vs_new": float(np.mean([r["correct"] for r in subset])),
        "acc_vs_old": float(np.mean([r["correct_vs_old"] for r in subset])),
        "parsed": float(np.mean([r["parsed"] for r in subset])),
        "truncated": float(np.mean([r["truncated"] for r in subset])),
        "answer_unchanged": float(np.mean([r["answer_unchanged"] for r in subset])),
        "non_answer": non_answer_summary([r.get("non_answer") for r in subset]),
    }
    new_only = sum(bool(r["correct"]) and not r["correct_vs_old"] for r in subset)
    old_only = sum(bool(r["correct_vs_old"]) and not r["correct"] for r in subset)
    entry["num_correct_vs_new_only"] = new_only
    entry["num_correct_vs_old_only"] = old_only
    entry["p_sign_test_two_sided_null_acc_vs_new_eq_acc_vs_old"] = (
        min(1.0, 2 * _upper_tail(max(new_only, old_only), [0.5] * (new_only + old_only)))
        if new_only + old_only
        else None
    )
    if moved:
        entry["n_moved"] = len(moved)
        entry["acc_vs_new_on_moved"] = float(np.mean([r["correct"] for r in moved]))
        entry["acc_vs_old_on_moved"] = float(np.mean([r["correct_vs_old"] for r in moved]))
        entry["answer_unchanged_on_moved"] = float(np.mean([r["answer_unchanged"] for r in moved]))
        closer = [r["closer_to_new"] for r in moved if "closer_to_new" in r]
        if closer:
            entry["closer_to_new_on_moved"] = float(np.mean(closer))
        expert_moved = [r for r in moved if r["type"] in EXPERT_VALUED_TYPES]
        if expert_moved:
            chances = [_chance_vs_new(r, num_experts) for r in expert_moved]
            num_correct = sum(bool(r["correct"]) for r in expert_moved)
            entry["expert_valued_on_moved"] = {
                "n": len(expert_moved),
                "acc_vs_new": num_correct / len(expert_moved),
                "chance_uniform_over_unablated_experts": float(np.mean(chances)),
                "p_one_sided_null_acc_vs_new_eq_uniform_chance": _upper_tail(num_correct, chances),
            }
    if name == "ablate_hit":
        entry["answered_ablated"] = float(np.mean([r["answered_ablated"] for r in subset]))
    if name == "force":
        entry["answered_forced"] = float(np.mean([r["answered_forced"] for r in subset]))
    return entry


def summarize(rows: list[dict], which_rows: list[dict], modes: list[str], num_experts: int) -> dict:
    summary: dict = {}
    conditions_seen = list(dict.fromkeys(r["condition"] for r in rows))
    items = {r["item"]: r["prompt_identical_across_modes"] for r in rows}
    if items:
        summary["prompt_identical_across_modes"] = float(np.mean(list(items.values())))
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue
        summary[mode] = {
            name: condition_summary([r for r in mode_rows if r["condition"] == name], name, num_experts)
            for name in conditions_seen
        }
        summary[mode]["per_type"] = {
            f"{type}/{name}": condition_summary(
                [r for r in mode_rows if r["type"] == type and r["condition"] == name], name, num_experts
            )
            for type in sorted({r["type"] for r in mode_rows})
            for name in conditions_seen
        }
        mode_which = [r for r in which_rows if r["mode"] == mode]
        if mode_which:
            summary[mode]["which_experts_disabled"] = {
                "n": len(mode_which),
                "jaccard": float(np.mean([r["jaccard"] for r in mode_which])),
                "mean_num_guessed": float(np.mean([r["num_guessed"] for r in mode_which])),
                "chance_jaccard": GROUP / (2 * num_experts - GROUP),
            }
    return summary


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    model, labeler, builder = ctx.torch_model, ctx.torch_labeler, ctx.probe_builder
    family = ctx.family
    domain = ctx.config.domain
    types = list(ctx.model.types or ctx.types)
    if domain == "routing":
        ctx.require_judge()
    num_groups = ctx.arg("num_groups", 8)
    max_new = ctx.max_tokens
    noise = ctx.arg("router_noise", 1.0)
    modes = [m for m in str(ctx.arg("answer_modes", ",".join(MODES))).split(",") if m]
    prompts = held_out_prompts(num_groups * GROUP, seed=ctx.seed + 2)
    convs = _replies(model, builder, prompts, ctx.max_tokens)
    ctx.log(f"{len(convs)} of {len(prompts)} carrier replies usable")

    rows: list[dict] = []
    which_rows: list[dict] = []
    follow_up_skipped = 0
    num_items = 0
    for group_index in range(num_groups):
        group_convs = convs[group_index * GROUP : (group_index + 1) * GROUP]
        if not group_convs:
            break
        layer = rng.choice(family.moe_layers)
        type = types[group_index % len(types)]
        draws = [
            builder.build(c, rng=rng, type=type, layer=layer if domain == "routing" else None) for c in group_convs
        ]
        base_questions = [p for p in draws if p]
        if not base_questions:
            continue
        if domain == "routing":
            conditions = _routing_conditions(rng, layer, _top_experts(labeler, base_questions), family, noise)
        else:
            conditions = _generic_conditions(rng, family, noise)
        rebuilt = {
            mode: [
                builder.build(
                    c,
                    rng=random.Random(0),
                    type=p.type,
                    turn=p.turn,
                    layer=p.layer,
                    pos=p.pos,
                    rank=p.rank,
                    thinking=mode,
                )
                for c, p in zip(group_convs, draws, strict=True)
                if p
            ]
            for mode in modes
        }
        probes_by_mode = {mode: [p for p in rebuilt[mode] if p is not None] for mode in modes}
        assert all(len(mode_questions) == len(base_questions) for mode_questions in probes_by_mode.values())
        items = list(range(num_items, num_items + len(base_questions)))
        num_items += len(base_questions)
        identical = [
            len({tuple(probes_by_mode[m][i].prefix_ids) for m in modes}) == 1 for i in range(len(base_questions))
        ]
        for mode in modes:
            mode_questions = probes_by_mode[mode]
            clean_labels: list[int | None] = []
            clean_answers: list[int | str | None] = []
            for condition in conditions:
                ctx.log(f"group {group_index} layer {layer} type {type}: {condition.name} ({mode})")
                probes = [
                    dataclasses.replace(p, meta={"condition": condition.name, "group": group_index, "mode": mode})
                    for p in mode_questions
                ]
                with model.edits(router=condition.router, layer=condition.layer):
                    new_labels = _label(labeler, probes)
                    for probe, label in zip(probes, new_labels, strict=True):
                        probe.label = label
                    _generate(model, builder, probes, max_new, mode)
                    if condition.ablated and domain == "routing":
                        follow_up_skipped += sum(not builder.visible_answer(p, mode).strip() for p in probes)
                        for extra in _which_experts(model, builder, probes, condition, max_new, ctx, mode):
                            which_rows.append(
                                {**extra, "condition": condition.name, "group": group_index, "mode": mode}
                            )
                if condition.name == "clean":
                    clean_labels = [p.label for p in probes]
                    clean_answers = [p.model_answer for p in probes]
                for probe, item, same_prompt, old, clean_answer in zip(
                    probes, items, identical, clean_labels, clean_answers, strict=True
                ):
                    row = probe.row()
                    row["item"] = item
                    row["prompt_identical_across_modes"] = same_prompt
                    row["num_ablated"] = len(condition.ablated)
                    row["old_label"] = old
                    row["label_moved"] = old != probe.label
                    row["correct_vs_old"] = builder.matches(probe.type, old, probe.model_answer)
                    row["answer_unchanged"] = probe.model_answer == clean_answer
                    answer, new = probe.model_answer, probe.label
                    if probe.type in SCALAR_TYPES and isinstance(answer, int) and new is not None and old is not None:
                        row["closer_to_new"] = abs(answer - new) < abs(answer - old)
                    expert_valued = probe.type in EXPERT_VALUED_TYPES
                    row["answered_ablated"] = expert_valued and probe.model_answer in condition.ablated
                    row["answered_forced"] = (
                        expert_valued and condition.forced is not None and probe.model_answer == condition.forced
                    )
                    rows.append(row)

    summary: dict = {
        "carriers_dropped": 1 - len(convs) / len(prompts),
        "judge": ctx.config.judge_model,
        **summarize(rows, which_rows, modes, family.num_experts),
    }
    summary["which_experts_skipped_for_empty_answer"] = follow_up_skipped
    return Result(summary=summary, rows=rows + [{"which_experts": r} for r in which_rows])
