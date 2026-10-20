from evals.ciaware import ARTEFACT_CODE, ESSAY_STEPS, TEMPERATURE, Setup, run_ciaware
from evals.config import CACHE
from evals.fingerprint import artefact_key
from evals.suites import Context, Result

NAME = "ciaware"
PATH = "server"


def run(ctx: Context) -> Result:
    trusted = ctx.require_judge()
    artefact_settings = {
        "family": ctx.config.family,
        "answer_mode": ctx.config.answer_mode,
        "max_tokens": ctx.max_tokens,
        "max_model_len": ctx.config.max_model_len,
        "seed": ctx.seed,
        "temperature": TEMPERATURE,
    }
    setup = Setup(
        client=ctx.server,
        trusted=trusted,
        cache=CACHE / "ciaware_model_data" / f"{ctx.config.name}-{ctx.model.name}",
        log=ctx.log,
        seed=ctx.seed,
        max_tokens=ctx.max_tokens,
        max_model_len=ctx.config.max_model_len,
        artefact_key=artefact_key(ctx.model.checkpoint, ARTEFACT_CODE, artefact_settings),
    )
    settings = [name for name in ctx.arg("settings", "essay,bcb").split(",") if name]
    results = run_ciaware(
        setup, ctx.out_dir / "ciaware", settings, num_problems=ctx.arg("limit", 50), steps=ctx.arg("steps", ESSAY_STEPS)
    )
    summary = {"intervening": ctx.config.judge_model, "metric": "balanced_accuracy = (TPR + TNR) / 2", **results}
    rows = [{"setting": name, **result} for name, result in results.items()]
    return Result(summary=summary, rows=rows)
