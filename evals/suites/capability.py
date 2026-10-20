import random
import re

import numpy as np

from evals.client import non_answer_summary
from evals.datasets import gsm8k
from evals.interventions import RouterEdit
from evals.suites import Context, Result
from evals.suites.ablation import MODES
from proprioception.oracle import ReasoningMarkers, ends_inside_reasoning, visible_assistant_content
from proprioception.rendering import USER

NAME = "capability"
PATH = "torch"
BATCH = 8
INSTRUCTION = "\n\nSolve the problem and give the final numeric answer on its own last line as: #### <number>"
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    numbers = _NUMBER.findall(text or "")
    if not numbers:
        return None
    value = numbers[-1].replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def gold_number(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")


def _opened(prompt: str, markers: ReasoningMarkers) -> str:
    return markers.open if ends_inside_reasoning(prompt, markers) else ""


def conditions(rng: random.Random, moe_layers: list[int], num_experts: int) -> dict[str, list[RouterEdit]]:
    quarter = tuple(range(num_experts // 4))
    layer = rng.choice(moe_layers)
    octet = tuple(sorted(rng.sample(range(num_experts), min(8, num_experts))))
    return {
        "clean": [],
        "same": [RouterEdit(layer=layer) for layer in moe_layers],
        "quarter_global": [RouterEdit(layer=layer, ablate=quarter) for layer in moe_layers],
        "layer_octet": [RouterEdit(layer=layer, ablate=octet)],
    }


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    model, builder, family = ctx.torch_model, ctx.probe_builder, ctx.family
    problems = gsm8k(ctx.arg("num_problems", 100), seed=ctx.seed)
    modes = [m for m in str(ctx.arg("answer_modes", ",".join(MODES))).split(",") if m]
    budgets = {"think": ctx.max_tokens, "nothink": ctx.max_tokens}
    edits = conditions(rng, family.moe_layers, family.num_experts)
    rows: list[dict] = []
    summary: dict = {"num_problems": len(problems), "conditions": {k: [str(e) for e in v] for k, v in edits.items()}}
    for mode in modes:
        renderer, _ = builder.mode(mode)
        prompts = [
            renderer.render(
                [{"role": USER, "content": p["question"] + INSTRUCTION}], add_generation_prompt=True
            ).token_ids
            for p in problems
        ]
        clean_replies: list[str] = []
        summary[mode] = {}
        for name, router in edits.items():
            ctx.log(f"{name} ({mode})")
            replies: list[str] = []
            hit_budget: list[bool] = []
            with model.edits(router=router):
                for start in range(0, len(prompts), BATCH):
                    batch = prompts[start : start + BATCH]
                    outs = model.generate(batch, max_new_tokens=budgets[mode], bad_words_ids=builder.banned_ids(mode))
                    hit_budget += [len(ids) >= budgets[mode] for ids in outs]
                    replies += [
                        _opened(renderer.decode(prompt), family.reasoning) + builder.renderer.decode(ids)
                        for prompt, ids in zip(batch, outs, strict=True)
                    ]
            if name == "clean":
                clean_replies = replies
            entries = []
            for problem, reply, clean, at_budget in zip(problems, replies, clean_replies, hit_budget, strict=True):
                visible = visible_assistant_content(reply, family.reasoning)
                predicted = final_number(visible)
                gold = gold_number(problem["answer"])
                open_block = bool(family.reasoning.open) and ends_inside_reasoning(reply, family.reasoning)
                if predicted is not None:
                    category = None
                elif at_budget:
                    category = "truncated"
                elif open_block:
                    category = "unclosed_reasoning"
                elif not visible.strip():
                    category = "empty"
                else:
                    category = "unparsed"
                entries.append(
                    {
                        "mode": mode,
                        "condition": name,
                        "question": problem["question"],
                        "gold": gold,
                        "predicted": predicted,
                        "correct": predicted == gold,
                        "parsed": predicted is not None,
                        "truncated": at_budget,
                        "open_reasoning": open_block,
                        "non_answer": category,
                        "num_tokens": len(builder.renderer.encode_content(reply)),
                        "reply_unchanged": reply == clean,
                        "reply": reply,
                    }
                )
            rows += entries
            summary[mode][name] = {
                "n": len(entries),
                "accuracy": float(np.mean([e["correct"] for e in entries])),
                "parsed": float(np.mean([e["parsed"] for e in entries])),
                "truncated": float(np.mean([e["truncated"] for e in entries])),
                "non_answer": non_answer_summary([e["non_answer"] for e in entries]),
                "mean_tokens": float(np.mean([e["num_tokens"] for e in entries])),
                "reply_unchanged": float(np.mean([e["reply_unchanged"] for e in entries])),
            }
    return Result(summary=summary, rows=rows)
