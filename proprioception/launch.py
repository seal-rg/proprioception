import argparse
import copy
import dataclasses
import os
from collections.abc import Mapping
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from proprioception import config
from proprioception.bank import BANK
from proprioception.families import FAMILIES, family_for_checkpoint

REPO = Path(__file__).resolve().parents[1]
PRIME = REPO / "external" / "prime-rl"
READY_TIMEOUT_SECONDS = 1800
COMPILE_CACHE_DIRS = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "VLLM_CACHE_ROOT": "vllm",
    "TILELANG_CACHE_DIR": "tilelang",
    "CUDA_CACHE_PATH": "nv-compute",
}


def visible_gpus() -> list[str]:
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if listed is not None:
        return [gpu for gpu in listed.split(",") if gpu]
    if shutil.which("nvidia-smi") is None:
        return []
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True
    )
    return [line.strip() for line in output.stdout.splitlines() if line.strip()]


def split_gpus(gpus: list[str], num_server: int) -> tuple[list[str], list[str]]:
    if num_server and len(gpus) < num_server:
        raise ValueError(f"{len(gpus)} visible GPUs cannot hold {num_server} for the server")
    return gpus[:num_server], gpus[num_server:]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def run_command(model: str, num_gpus: int, run_argv: list[str], *, master_port: int) -> list[str]:
    if model and model not in FAMILIES:
        raise ValueError(f"unknown model family {model!r}; known: {sorted(FAMILIES)}")
    prime = bool(model) and FAMILIES[model].backend == "prime"
    venv = (PRIME if prime else REPO) / ".venv" / "bin"
    if num_gpus > 1 or (prime and num_gpus == 1):
        head = [str(venv / "torchrun"), f"--nproc_per_node={num_gpus}", "--master_port", str(master_port), "-m"]
    else:
        head = [str(venv / "python"), "-m"]
    return [*head, "proprioception.cli", *run_argv]


def server_command(
    spec: str,
    port: int | None,
    *,
    cwd: Path,
    model_override: str = "",
    vllm_args: list[str] | None = None,
    gpu_memory_utilization: float = 0.85,
) -> tuple[list[str], int, str, Path]:
    if spec.startswith("vllm:"):
        checkpoint = os.path.expandvars(model_override or spec[len("vllm:") :])
        chosen = port or free_port()
        command = [
            str(REPO / ".venv" / "bin" / "vllm"),
            "serve",
            checkpoint,
            *(vllm_args if vllm_args is not None else ["--enable-return-routed-experts"]),
            "--port",
            str(chosen),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--disable-log-stats",
            "--uvicorn-log-level",
            "warning",
        ]
        return command, chosen, checkpoint, cwd
    config_path = Path(os.path.expandvars(spec))
    config = tomllib.loads(config_path.read_text())
    model = os.path.expandvars(model_override or str(config.get("model", {}).get("name", "")))
    if not model:
        raise ValueError(f"{config_path} names no [model] name to serve")
    chosen = port or free_port()
    command = [str(PRIME / ".venv" / "bin" / "inference"), "@", str(config_path.resolve()), "--model.name", model]
    command += ["--server.port", str(chosen)]
    return command, chosen, model, PRIME


LOOPBACK = "127.0.0.1"


def loopback_no_proxy(environ: Mapping[str, str]) -> dict[str, str]:
    exempt = [LOOPBACK, "localhost"]
    out = {}
    for name in ("NO_PROXY", "no_proxy"):
        current = [entry.strip() for entry in environ.get(name, "").split(",") if entry.strip()]
        out[name] = ",".join(current + [entry for entry in exempt if entry not in current])
    return out


def compile_cache_env(environ: Mapping[str, str]) -> dict[str, str]:
    choice = environ.get("PROPRIOCEPTION_COMPILE_CACHE", "global")
    if choice == "global":
        return {}
    root = Path("/tmp/") if choice == "local" else Path(choice)
    return {name: str(root / "proprioception-compile-cache" / sub) for name, sub in COMPILE_CACHE_DIRS.items()}


def wait_ready(url: str, process: subprocess.Popen, *, timeout_seconds: float) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited with code {process.returncode} before it was ready")
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(3)
    raise TimeoutError(f"the server at {url} was not ready after {timeout_seconds:.0f}s")


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse(argv: list[str]) -> tuple[argparse.Namespace, str, list[str]]:
    if len(argv) < 2 or argv[0] != "@":
        raise ValueError("usage: proprio launch @ run.toml [--server ...] [--server_gpus N] [run overrides]")
    parser = argparse.ArgumentParser(prog="proprio launch", add_help=False)
    parser.add_argument("--server", default="", help="prime-rl inference TOML, or vllm:CHECKPOINT")
    parser.add_argument("--server_model", default="", help="serve this checkpoint instead of the server config's")
    parser.add_argument("--server_gpus", type=int, default=1, help="visible GPUs given to the server")
    parser.add_argument("--port", type=int, default=None, help="server port")
    parser.add_argument("--master_port", type=int, default=None, help="torchrun rendezvous port")
    parser.add_argument("--nccl_port", type=int, default=None, help="NCCL weight-broadcast port")
    parser.add_argument("--dry_run", action="store_true", help="print commands and exit")
    options, overrides = parser.parse_known_args(argv[2:])
    return options, argv[1], [os.path.expandvars(token) for token in overrides]


def toml_literal(raw: str):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def rl_values(
    values: dict,
    *,
    checkpoint: str,
    server_port: int,
    teacher_port: int,
    output: Path,
    sets: list[str],
    teacher_checkpoint: str = "",
) -> dict:
    resolved = copy.deepcopy(values)
    resolved.setdefault("model", {})["name"] = checkpoint
    resolved.setdefault("inference", {}).setdefault("server", {})["port"] = server_port
    orchestrator = resolved.setdefault("orchestrator", {})
    orchestrator.setdefault("model", {}).setdefault("client", {})["base_url"] = [f"http://{LOOPBACK}:{server_port}/v1"]
    for section in ("train", "eval"):
        for env in orchestrator.get(section, {}).get("env", []):
            taskset = env.setdefault("taskset", {})
            bank = taskset.get("prompt_bank_path")
            if not bank or not Path(bank).exists():
                taskset["prompt_bank_path"] = str(BANK)
            user = taskset.setdefault("task", {}).setdefault("user", {})
            user["oracle_model"] = checkpoint
            user["oracle_base_url"] = f"http://{LOOPBACK}:{server_port}"
    algos = [env.get("algo", {}) for env in orchestrator.get("train", {}).get("env", [])]
    algos.append(orchestrator.get("algo", {}))
    for algo in algos:
        for frozen in ("teacher", "reference"):
            model = algo.get(frozen)
            if model is not None:
                model["name"] = teacher_checkpoint or checkpoint
                model["base_url"] = [f"http://{LOOPBACK}:{teacher_port}/v1"]
    if any((output / "checkpoints").glob("step_*/STABLE")):
        resolved.setdefault("ckpt", {})["resume_step"] = -1
    for item in sets:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--set takes key.path=value, got {item!r}")
        *path, last = key.strip().split(".")
        target = resolved
        for part in path:
            target = target[_entry(target, part, key)] if isinstance(target, list) else target.setdefault(part, {})
            if not isinstance(target, (dict, list)):
                raise ValueError(f"--set {key}: {part!r} holds a value, so nothing can be set inside it")
        if isinstance(target, list):
            target[_entry(target, last, key)] = toml_literal(raw.strip())
        else:
            target[last] = toml_literal(raw.strip())
    return resolved


def _entry(entries: list, part: str, key: str) -> int:
    named = [i for i, entry in enumerate(entries) if isinstance(entry, dict) and entry.get("name") == part]
    if len(named) > 1:
        raise ValueError(f"--set {key}: {part!r} names {len(named)} entries of the list")
    if named:
        return named[0]
    if part.lstrip("-").isdigit() and -len(entries) <= int(part) < len(entries):
        return int(part)
    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    raise ValueError(f"--set {key}: {part!r} is neither a position in a list of {len(entries)} nor one of {names}")


def rl_options(overrides: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="proprio launch @ rl.toml", add_help=False)
    parser.add_argument("--checkpoint", required=True, help="the policy checkpoint to start from")
    parser.add_argument("--output", required=True, help="prime-rl's output directory")
    parser.add_argument("--teacher_server", default="", help="inference TOML for the frozen teacher")
    parser.add_argument(
        "--teacher_checkpoint", default="", help="the anchor's teacher; the policy checkpoint when unset"
    )
    parser.add_argument("--teacher_gpus", type=int, default=1, help="visible GPUs given to the teacher server")
    parser.add_argument("--set", action="append", default=[], help="key.path=TOML value applied to the config")
    try:
        return parser.parse_known_args(overrides)
    except SystemExit as error:
        raise ValueError("the rl stage needs --checkpoint and --output") from error


@dataclasses.dataclass
class Job:
    command: list[str]
    run_gpus: list[str]
    run_cwd: Path
    run_env: dict[str, str]
    shared_env: dict[str, str] = dataclasses.field(default_factory=dict)
    server: list[str] | None = None
    server_gpus: list[str] = dataclasses.field(default_factory=list)
    server_cwd: Path = REPO
    port: int = 0
    files: dict[Path, str] = dataclasses.field(default_factory=dict)


def execute(job: Job, *, dry_run: bool) -> int:
    if dry_run:
        if job.server is not None:
            print(f"server (GPUs {','.join(job.server_gpus) or '-'}): {shlex.join(job.server)}")
        print(f"run (GPUs {','.join(job.run_gpus) or '-'}): {shlex.join(job.command)}")
        for path, content in job.files.items():
            print(f"file {path}:\n{content}")
        return 0
    for path, content in job.files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    server_process = None
    run_process = None

    def forward(signum, _frame):
        stop(run_process)
        stop(server_process)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        if job.server is not None:
            server_env = {**os.environ, **job.shared_env, "CUDA_VISIBLE_DEVICES": ",".join(job.server_gpus)}
            server_env.pop("VIRTUAL_ENV", None)
            print(f"server: {shlex.join(job.server)}", flush=True)
            server_process = subprocess.Popen(job.server, cwd=job.server_cwd, env=server_env)
            wait_ready(f"http://{LOOPBACK}:{job.port}/v1/models", server_process, timeout_seconds=READY_TIMEOUT_SECONDS)
            print(f"server ready on port {job.port}", flush=True)
        print(f"run: {shlex.join(job.command)}", flush=True)
        run_process = subprocess.Popen(job.command, cwd=job.run_cwd, env=job.run_env)
        return run_process.wait()
    finally:
        stop(run_process)
        stop(server_process)


def main(argv: list[str]) -> int:
    options, run_config, overrides = parse(argv)
    stage, values = config.read_toml(run_config)
    model = next((overrides[i + 1] for i, token in enumerate(overrides[:-1]) if token == "--model"), None)
    model = model or (str(values.get("model", "")) if stage != "rl" else "")
    vllm_args: list[str] | None = None
    gpu_memory_utilization = 0.85
    rl = None
    if stage == "capabilities":
        settings = config.parse(stage, values, [*overrides, "--server_url", "pending"])
        assert isinstance(settings, config.CapabilitiesConfig)
        options.server = f"vllm:{settings.checkpoint}"
        options.server_gpus = len(visible_gpus()) or 1
        gpu_memory_utilization = settings.gpu_memory_utilization
        parser = settings.reasoning_parser
        if parser == "auto":
            parser = FAMILIES[family_for_checkpoint(settings.checkpoint)].reasoning_parser
        vllm_args = [
            *(["--reasoning-parser", parser] if parser else []),
            "--tensor-parallel-size",
            str(options.server_gpus),
        ]
        if settings.max_model_len:
            vllm_args += ["--max-model-len", str(settings.max_model_len)]
        if settings.max_num_seqs:
            vllm_args += ["--max-num-seqs", str(settings.max_num_seqs)]
    if stage == "rl":
        rl, overrides = rl_options(overrides)
        train_algos = [env.get("algo", {}) for env in values.get("orchestrator", {}).get("train", {}).get("env", [])]
        train_algos.append(values.get("orchestrator", {}).get("algo", {}))
        needs_teacher = any(algo.get(frozen) is not None for algo in train_algos for frozen in ("teacher", "reference"))
        if needs_teacher:
            options.server = os.path.expandvars(rl.teacher_server) or str(REPO / "configs/inference.toml")
            options.server_model = os.path.expandvars(rl.teacher_checkpoint) or os.path.expandvars(rl.checkpoint)
            options.server_gpus = rl.teacher_gpus
    server_gpus, run_gpus = split_gpus(visible_gpus(), options.server_gpus if options.server else 0)
    if stage in ("build", "capabilities", "convert"):
        run_gpus = []
    run_argv = ["@", run_config, *overrides]
    server = None
    port = 0
    server_cwd = REPO
    if options.server:
        if stage not in ("on-policy-sft", "capabilities", "rl"):
            raise ValueError(f"stage {stage!r} takes no server")
        if stage == "on-policy-sft" and "--nccl_port" not in overrides:
            run_argv += ["--nccl_port", str(options.nccl_port or free_port())]
        server, port, served_model, server_cwd = server_command(
            options.server,
            options.port,
            cwd=REPO,
            model_override=options.server_model,
            vllm_args=vllm_args,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if stage == "on-policy-sft":
            run_argv += ["--server_url", f"http://{LOOPBACK}:{port}", "--served_model", served_model]
            if "--server_gpus" not in overrides:
                run_argv += ["--server_gpus", str(len(server_gpus))]
            transport = "nccl" if FAMILIES[model].backend == "prime" else "filesystem"
            if "--weight_transport" not in overrides and not options.server.startswith("vllm:"):
                run_argv += ["--weight_transport", transport]
                server += ["--weight_broadcast.type", transport]
        elif stage == "capabilities":
            run_argv += ["--server_url", f"http://{LOOPBACK}:{port}"]
        elif stage == "rl":
            server += ["--weight_broadcast.type", "filesystem"]
    shared_env = compile_cache_env(os.environ)
    run_env = {**os.environ, **shared_env, "CUDA_VISIBLE_DEVICES": ",".join(run_gpus), **loopback_no_proxy(os.environ)}
    run_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    files: dict[Path, str] = {}
    if rl is not None:
        import tomli_w

        output = Path(os.path.expandvars(rl.output))
        resolved = rl_values(
            values,
            checkpoint=os.path.expandvars(rl.checkpoint),
            server_port=free_port(),
            teacher_port=port,
            output=output,
            sets=rl.set,
            teacher_checkpoint=options.server_model,
        )
        deployment = resolved.get("deployment", {})
        wanted = int(deployment.get("num_infer_gpus", 0)) + int(deployment.get("num_train_gpus", 0))
        if run_gpus and wanted != len(run_gpus):
            raise ValueError(
                f"the config deploys on {wanted} GPUs but {len(run_gpus)} remain after the teacher's {len(server_gpus)}"
            )
        resolved_path = output / "launch" / "rl.toml"
        files[resolved_path] = tomli_w.dumps(resolved)
        venv_bin = PRIME / ".venv" / "bin"
        command = [str(venv_bin / "rl"), "@", str(resolved_path), "--output-dir", str(output), *overrides]
        run_cwd = PRIME
        run_env["PATH"] = f"{venv_bin}{os.pathsep}{run_env.get('PATH', '')}"
        run_env.pop("VIRTUAL_ENV", None)
    else:
        command = run_command(model, max(len(run_gpus), 1), run_argv, master_port=options.master_port or free_port())
        run_cwd = REPO
    job = Job(command=command, run_gpus=run_gpus, run_cwd=run_cwd, run_env=run_env, shared_env=shared_env, files=files)
    if server is not None:
        job.server, job.server_gpus, job.server_cwd, job.port = server, server_gpus, server_cwd, port
    return execute(job, dry_run=options.dry_run)
