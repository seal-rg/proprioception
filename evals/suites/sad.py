import json
import os

from evals.config import CACHE
from evals.sad import Facts, Runtime, facts_path, run_sad, table
from evals.fingerprint import artefact_key
from evals.sad.runtime import ARTEFACT_CODE
from evals.suites import Context, Result

NAME = "sad"
PATH = "server"


def facts_file(ctx: Context) -> str:
    override = ctx.arg("facts_file", "")
    return os.path.expandvars(override) if override else str(facts_path(ctx.config.family))


def run(ctx: Context) -> Result:
    path = facts_file(ctx)
    n = ctx.arg("n", 0)
    settings = {
        "family": ctx.config.family,
        "answer_mode": ctx.config.answer_mode,
        "max_tokens": ctx.max_tokens,
        "max_model_len": ctx.config.max_model_len,
        "seed": ctx.seed,
        "n": n,
    }
    runtime = Runtime(
        client=ctx.server,
        checkpoint=ctx.model.checkpoint,
        facts=Facts.load(path),
        cache=CACHE / "sad_model_data" / f"{ctx.config.name}-{ctx.model.name}",
        log=ctx.log,
        judge=ctx.judge,
        n=n,
        seed=ctx.seed,
        max_tokens=ctx.max_tokens,
        judge_max_tokens=ctx.judge_max_tokens,
        output_control_samples=ctx.arg("output_control_samples", 16),
        artefact_key=artefact_key(ctx.model.checkpoint, ARTEFACT_CODE, settings),
    )
    tasks = [name for name in ctx.arg("tasks", "").split(",") if name]
    variants = [name for name in ctx.arg("variants", "").split(",") if name]
    previous = None
    if tasks and (ctx.out_dir / "sad.json").exists():
        previous = json.loads((ctx.out_dir / "sad.json").read_text()).get("summary")
    summary = run_sad(runtime, ctx.out_dir / "sad", tasks=tasks, variants=variants, previous=previous)
    return Result(
        summary={"facts": path, "judge": ctx.config.judge_model, **summary},
        rows=table(summary),
    )
