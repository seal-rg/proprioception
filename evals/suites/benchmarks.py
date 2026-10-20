import os

from evals.suites import Context, Result
from proprioception.launch import loopback_no_proxy

NAME = "benchmarks"
PATH = "server"

DEFAULT_TASKS = ["gsm8k_cot_zeroshot", "ifeval", "minerva_math500", "aime25_answer"]
METRICS = {
    "gsm8k": ("gsm8k_cot_zeroshot", "exact_match,flexible-extract"),
    "ifeval_prompt": ("ifeval", "prompt_level_strict_acc,none"),
    "ifeval_inst": ("ifeval", "inst_level_strict_acc,none"),
    "math500": ("minerva_math500", "math_verify,none"),
    "aime25": ("aime25_answer", "exact_match,none"),
    "aime25_answer": ("aime25_answer", "answer_match,none"),
}


def run(ctx: Context) -> Result:
    from proprioception.lm_eval_answers import request_timeout, run_lm_eval

    os.environ.update(loopback_no_proxy(os.environ))
    tasks = ctx.arg("tasks", DEFAULT_TASKS)
    gen_kwargs = ctx.arg("gen_kwargs", "temperature=0,max_gen_toks=32768")
    limit = ctx.arg("limit", 0)
    payload = run_lm_eval(
        checkpoint=ctx.model.checkpoint,
        base_url=ctx.server.base_url,
        tasks=tasks,
        gen_kwargs=gen_kwargs,
        output=ctx.out_dir / "lm_eval",
        markers=ctx.family.reasoning,
        limit=limit,
    )
    results = payload["results"]
    summary: dict = {}
    for name, (task, metric) in METRICS.items():
        if task in results and metric in results[task]:
            summary[name] = float(results[task][metric])
    summary["invocation"] = {
        "tasks": list(tasks),
        "gen_kwargs": gen_kwargs,
        "limit": limit,
        "request_timeout_seconds": request_timeout(gen_kwargs),
        "request_timeouts": payload.get("request_timeouts"),
        "n_samples": payload.get("n-samples"),
        "answer_mode": ctx.config.answer_mode,
        "answer_mode_applied": False,
    }
    return Result(summary=summary, rows=[])
