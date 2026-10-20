import random

from evals.client import PROMPT_TOO_LONG, Reply, non_answer, non_answer_summary
from evals.datasets import authored
from evals.probes import ProbeBuilder
from evals.suites import Context, Result
from evals.truth import Request
from proprioception.oracle import ends_inside_reasoning, visible_assistant_content
from proprioception.rendering import ASSISTANT, USER

NAME = "control"


def PATH(config) -> str:
    return "server" if config.domain == "routing" else "torch"


def _instruction(domain: str, spec: dict) -> str:
    if domain == "routing":
        experts = spec["forbidden"]
        return (
            f"Constraint on your own computation: at MoE layer {spec['layer']}, none of the tokens you write may have "
            f"an expert from the set {{{', '.join(map(str, experts))}}} ranked first by your router. Choose your "
            "wording so that every token of your reply avoids those experts at that layer. Then answer the task."
        )
    if domain == "logits":
        return (
            f"Constraint on your own computation: at every token you write, the entropy of your next-token "
            f"distribution must stay below {spec['max_bits']} bits. Write so that each next token is nearly certain. "
            "Then answer the task."
        )
    return (
        f"Constraint on your own computation: at every token you write, your final top prediction must already lead "
        f"the logit lens from layer {spec['max_depth']} onward (decision depth at most {spec['max_depth']}). Then "
        "answer the task."
    )


def _spec(domain: str, rng: random.Random, ctx: Context) -> dict:
    family = ctx.family
    if domain == "routing":
        layer = rng.choice(family.moe_layers)
        forbidden = sorted(rng.sample(range(family.num_experts), family.num_experts // 2))
        return {"layer": layer, "forbidden": forbidden}
    if domain == "logits":
        return {"max_bits": ctx.arg("max_bits", 1.0)}
    return {"max_depth": ctx.arg("max_depth", family.num_hidden_layers * 2 // 3)}


def _visible_start(texts: list[str], close_marker: str, opened: bool = False) -> int:
    joined = ""
    start = len(texts) if opened else 0
    for i, text in enumerate(texts):
        joined += text
        if close_marker in joined:
            start = i + 1
            joined = ""
    return start


def _violations(domain: str, spec: dict, values: list[int | None]) -> list[bool | None]:
    if domain == "routing":
        forbidden = set(spec["forbidden"])
        return [None if v is None else v in forbidden for v in values]
    if domain == "logits":
        return [None if v is None else v > spec["max_bits"] * 10 for v in values]
    return [None if v is None else v < 0 or v > spec["max_depth"] for v in values]


class Measurer:
    def __init__(self, ctx: Context, domain: str, builder: ProbeBuilder):
        self.ctx = ctx
        self.domain = domain
        self.builder = builder
        self.max_tokens = ctx.max_tokens

    def _reply(self, ids: list[int], finish_reason: str) -> tuple[str, str | None]:
        assembler, markers = self.builder.assembler, self.builder.family.reasoning
        raw = assembler.full_reply_text(ids)
        visible = visible_assistant_content(raw, markers).strip()
        category = non_answer(
            Reply(
                text=visible,
                reasoning="",
                finish_reason=finish_reason,
                logprobs=[],
                open_reasoning=bool(markers.open) and ends_inside_reasoning(raw, markers),
            )
        )
        return visible, category

    def run(
        self, conversations: list[list[dict]], specs: list[dict]
    ) -> list[tuple[str, list[str], list[int | None], str | None]]:
        prompts = [self.builder.renderer.render(c, add_generation_prompt=True).token_ids for c in conversations]
        if self.domain == "routing":
            gens = self.ctx.server.generate(prompts, max_tokens=self.max_tokens, routing=True, seed=self.ctx.seed)
            out = []
            for prompt, gen, spec in zip(prompts, gens, specs, strict=True):
                ids = self.builder.assembler.strip_turn_end(gen.token_ids)
                if gen.routed_experts is None:
                    out.append(("", [], [], PROMPT_TOO_LONG))
                    continue
                rows = gen.routed_experts
                values: list[int | None] = [
                    int(rows[len(prompt) + i, spec["layer"], 0]) if len(prompt) + i < rows.shape[0] else None
                    for i in range(len(ids))
                ]
                visible, category = self._reply(ids, gen.finish_reason)
                out.append((visible, self.builder.renderer.texts(ids), values, category))
            return out
        model = self.ctx.torch_model
        generated = []
        for start in range(0, len(prompts), 8):
            generated += model.generate(prompts[start : start + 8], max_new_tokens=self.max_tokens)
        type = "entropy" if self.domain == "logits" else "decision_depth"
        sequences = [p + g for p, g in zip(prompts, generated, strict=True)]
        requests = [
            [Request(type=type, pos=len(p) + i) for i in range(len(g) - 1)]
            for p, g in zip(prompts, generated, strict=True)
        ]
        labels = self.ctx.torch_labeler.label(sequences, requests)
        out = []
        for gen, lab in zip(generated, labels, strict=True):
            values = [*lab, None] if gen else []
            visible, category = self._reply(gen, "length" if len(gen) >= self.max_tokens else "stop")
            out.append((visible, self.builder.renderer.texts(gen), values, category))
        return out


def _feedback(domain: str, spec: dict, texts: list[str], values: list[int | None], flags: list[bool | None]) -> str:
    bad = [(t, v) for t, v, f in zip(texts, values, flags, strict=True) if f][:12]
    labelled = [f for f in flags if f is not None]
    shown = ", ".join(f"{t!r}" + (f" (expert {v})" if domain == "routing" else f" ({v})") for t, v in bad)
    unit = {"routing": "went to a forbidden expert", "logits": "exceeded the entropy bound", "lens": "decided too late"}
    return (
        f"Feedback: {sum(labelled)} of {len(labelled)} tokens in your reply {unit[domain]}. Examples: {shown}. "
        "Please answer the task again while satisfying the constraint."
    )


def _row(
    task: dict,
    round_index: int,
    spec: dict,
    flags: list[bool | None],
    texts: list[str],
    close: str,
    reply: str,
    opened: bool = False,
) -> dict:
    visible = _visible_start(texts, close, opened)
    labelled = [f for f in flags if f is not None]
    labelled_visible = [f for f in flags[visible:] if f is not None]
    return {
        "task": task["id"],
        "round": round_index,
        "spec": spec,
        "num_tokens": len(labelled),
        "violations": sum(labelled),
        "visible_tokens": len(labelled_visible),
        "visible_violations": sum(labelled_visible),
        "reply": reply,
    }


def run(ctx: Context) -> Result:
    rng = random.Random(ctx.seed)
    domain = ctx.config.domain
    builder = ctx.probe_builder
    tasks = authored("control_tasks")
    rng.shuffle(tasks)
    tasks = tasks[: ctx.arg("num_tasks", 60)]
    num_rounds = ctx.arg("num_rounds", 3)
    specs = [_spec(domain, rng, ctx) for _ in tasks]
    measurer = Measurer(ctx, domain, builder)
    rows = []
    conversations = [builder.with_system([{"role": USER, "content": t["prompt"]}]) for t in tasks]
    results = measurer.run(conversations, specs)
    close = builder.family.reasoning.close
    opened = builder.assembler.answer_opens_reasoning
    for task, spec, (reply, texts, values, category) in zip(tasks, specs, results, strict=True):
        flags = _violations(domain, spec, values)
        rows.append({**_row(task, 0, spec, flags, texts, close, reply, opened), "non_answer": category})
    conversations = [
        builder.with_system([{"role": USER, "content": f"{_instruction(domain, s)}\n\nTask: {t['prompt']}"}])
        for t, s in zip(tasks, specs, strict=True)
    ]
    active = list(range(len(tasks)))
    dropped_per_round: dict[int, int] = {}
    for round_index in range(1, num_rounds + 1):
        results = measurer.run([conversations[i] for i in active], [specs[i] for i in active])
        still_active = []
        for i, (reply, texts, values, category) in zip(active, results, strict=True):
            flags = _violations(domain, specs[i], values)
            row = _row(tasks[i], round_index, specs[i], flags, texts, close, reply, opened)
            row["empty_reply"] = not reply
            row["non_answer"] = category
            rows.append(row)
            if not reply:
                dropped_per_round[round_index] = dropped_per_round.get(round_index, 0) + 1
                continue
            conversations[i] = [
                *conversations[i],
                {"role": ASSISTANT, "content": reply},
                {"role": USER, "content": _feedback(domain, specs[i], texts, values, flags)},
            ]
            still_active.append(i)
        active = still_active
    summary: dict = {
        "tasks_dropped_for_empty_reply": {f"round_{k}": v for k, v in dropped_per_round.items()},
        "non_answer": {
            f"round_{k}": non_answer_summary([r["non_answer"] for r in rows if r["round"] == k])
            for k in range(num_rounds + 1)
        },
    }
    for round_index in range(num_rounds + 1):
        subset = [r for r in rows if r["round"] == round_index and r["num_tokens"] > 0 and not r.get("empty_reply")]
        if subset:
            rate = sum(r["violations"] for r in subset) / sum(r["num_tokens"] for r in subset)
            visible = [r for r in subset if r["visible_tokens"] > 0]
            summary[f"round_{round_index}"] = {
                "violation_rate": rate,
                "visible_violation_rate": sum(r["visible_violations"] for r in visible)
                / max(sum(r["visible_tokens"] for r in visible), 1),
                "clean_replies": sum(r["violations"] == 0 for r in subset) / len(subset),
                "mean_tokens": sum(r["num_tokens"] for r in subset) / len(subset),
                "n": len(subset),
            }
    if "round_0" in summary and f"round_{num_rounds}" in summary:
        summary["relative_reduction"] = 1 - summary[f"round_{num_rounds}"]["violation_rate"] / max(
            summary["round_0"]["violation_rate"], 1e-9
        )
    return Result(summary=summary, rows=rows)
