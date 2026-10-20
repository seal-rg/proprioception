import json
import os
import subprocess
from pathlib import Path

from evals.config import REPO
from evals.server import free_port
from evals.suites import Context, Result

NAME = "template"
PATH = "torch"
NEEDS_TORCH_MODEL = False
PRIME_TORCHRUN = REPO / "external" / "prime-rl" / ".venv" / "bin" / "torchrun"


def run(ctx: Context) -> Result:
    data_dirs = [os.path.expandvars(d) for d in ctx.arg("data_dirs", [])]
    if not data_dirs:
        return Result(summary={"note": "no [suite.template] data_dirs configured"})
    if not PRIME_TORCHRUN.exists():
        return Result(summary={"note": f"prime-rl venv missing at {PRIME_TORCHRUN}"})
    gpu = ctx.config.resolved_gpu.split(",")[0]
    summary: dict = {}
    rows: list[dict] = []
    for data_dir in data_dirs:
        name = Path(data_dir).name
        out_dir = ctx.out_dir / "template" / name
        command = [
            str(PRIME_TORCHRUN),
            "--nproc_per_node=1",
            f"--master_port={free_port()}",
            "-m",
            "proprioception.cli",
            "eval",
            "--model",
            ctx.config.family,
            *([] if ctx.model.role == "base" else ["--checkpoint", ctx.model.checkpoint]),
            "--data_dir",
            data_dir,
            "--output",
            str(out_dir),
        ]
        env = {**os.environ}
        if not ctx.config.scheduler_allocated:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        ctx.log(" ".join(command))
        with (ctx.out_dir / f"template-{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, env=env, cwd=REPO)
        record = out_dir / f"eval-{name}.json"
        if not record.exists():
            summary[name] = {"note": f"no eval record; see {ctx.out_dir / f'template-{name}.log'}"}
            continue
        metrics = json.loads(record.read_text())["metrics"]
        summary[name] = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float)) or (isinstance(v, dict) and k == "types")
        }
        rows.append({"dataset": name, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
    return Result(summary=summary, rows=rows)
