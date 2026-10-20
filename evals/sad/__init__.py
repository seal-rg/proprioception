import json
from pathlib import Path

from evals.client import ServerStalled
from evals.sad.runtime import Facts, Runtime, facts_path
from evals.sad.scoring import aggregate, score_task
from evals.sad.tasks import TASKS

ALL_TASKS = tuple(TASKS)
VARIANTS = ("plain", "sp")
AGGREGATES = ("sad_score/", "sad_score_fixed/", "category/", "category_fixed/")


def run_sad(
    rt: Runtime,
    out_dir: Path,
    tasks: list[str] | None = None,
    variants: list[str] | None = None,
    previous: dict | None = None,
) -> dict:
    tasks = list(tasks or ALL_TASKS)
    variants = list(variants or VARIANTS)
    unknown = [name for name in tasks if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown SAD tasks {unknown}; known: {ALL_TASKS}")
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"unknown SAD variants {variants}; known: {VARIANTS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str], dict] = {}
    failures: dict[str, str] = {}
    for key, value in (previous or {}).items():
        if isinstance(value, dict) and "num_trials" in value and not key.startswith(AGGREGATES):
            name, variant = key.split("/")
            if name not in tasks:
                rows[(name, variant)] = value
    for key, reason in (previous or {}).get("failed", {}).items():
        if key.split("/")[0] not in tasks:
            failures[key] = reason
    for name in tasks:
        task = TASKS[name]
        if task.needs_judge and rt.judge is None:
            failures[name] = "needs the judge model"
            rt.log(f"sad: skipping {name} (judge-graded, no judge configured)")
            continue
        for variant in variants:
            rt.log(f"sad: {name}/{variant}")
            rt.num_truncated = 0
            rt.notes = {}
            try:
                results = task.run(rt, variant)
            except ServerStalled:
                raise
            except Exception as exc:
                failures[f"{name}/{variant}"] = f"{type(exc).__name__}: {exc}"
                rt.log(f"sad: {name}/{variant} failed: {type(exc).__name__}: {exc}")
                continue
            row = score_task(results, task.random_chance, task.mode)
            row["random_chance"] = task.random_chance
            row["num_truncated"] = rt.num_truncated
            row |= rt.notes
            rows[(name, variant)] = row
            _write_samples(out_dir / "answers" / f"{name}--{variant}.jsonl", results)
    summary: dict = {f"{name}/{variant}": row for (name, variant), row in rows.items()}
    has_fixed = any("score_fixed" in row for row in rows.values())
    for fixed in (False, True) if has_fixed else (False,):
        suffix = "_fixed" if fixed else ""
        for normalised in (False, True):
            form = "normalised" if normalised else "raw"
            scores = aggregate(rows, normalised=normalised, fixed=fixed)
            for key, value in scores["sad_score"].items():
                subset, variant = key.split("/")
                summary[f"sad_score{suffix}/{subset}/{form}/{variant}"] = value
            if not normalised:
                for key, value in scores["categories"].items():
                    summary[f"category{suffix}/{key}"] = value
    if failures:
        summary["failed"] = failures
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def table(summary: dict) -> list[dict]:
    return [
        {"task": key.split("/")[0], "variant": key.split("/")[1], **value}
        for key, value in summary.items()
        if isinstance(value, dict) and "num_trials" in value and not key.startswith(AGGREGATES)
    ]


def _write_samples(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "score": result.score,
                        "parsed": result.parsed,
                        "correct": result.correct,
                        "output": result.output,
                        "splits": result.splits,
                        "non_answer": result.non_answer,
                        "excluded": result.excluded,
                        "lenient": result.lenient,
                    }
                )
                + "\n"
            )


__all__ = ["ALL_TASKS", "VARIANTS", "Facts", "Runtime", "facts_path", "run_sad", "table"]
