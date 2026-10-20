import json
from pathlib import Path

from proprioception.config import CapabilitiesConfig
from proprioception.families import FAMILIES, family_for_checkpoint


def run(args: CapabilitiesConfig) -> None:
    from proprioception.lm_eval_answers import run_lm_eval

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_lm_eval(
        checkpoint=args.checkpoint,
        base_url=args.server_url,
        tasks=args.tasks,
        gen_kwargs=args.gen_kwargs,
        output=out,
        markers=FAMILIES[family_for_checkpoint(args.checkpoint)].reasoning,
        num_concurrent=args.num_concurrent,
        limit=args.limit,
        timeout=args.request_timeout,
    )
    results = sorted(out.glob("*/results_*.json"))
    if not results:
        raise ValueError(f"lm_eval wrote no results under {out}")
    print(summary(results[-1]), flush=True)


def summary(results_path: Path) -> str:
    results = json.loads(results_path.read_text())["results"]
    lines = [f"capabilities results in {results_path.parent}"]
    for task, metrics in sorted(results.items()):
        values = ", ".join(
            f"{name}={value:.4f}"
            for name, value in sorted(metrics.items())
            if isinstance(value, float) and "stderr" not in name
        )
        lines.append(f"  {task}: {values}")
    return "\n".join(lines)
