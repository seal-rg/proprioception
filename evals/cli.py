import contextlib
import dataclasses
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from evals import fingerprint
from evals.client import ChatClient
from evals.config import SYSTEM_PROMPTS, EvalConfig, ModelSpec, load_config
from evals.probes import ProbeBuilder
from evals.server import VllmServer, check_gpu_available
from evals.suites import Context, Result, load, resolve
from evals.torchmodel import TorchModel
from evals.truth import TorchLabeler
from proprioception.evaluation import STARTUP_COMMIT
from proprioception.families import resolve_family
from proprioception.launch import loopback_no_proxy


def _log(model: str, suite: str):
    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {model}/{suite}: {message}", flush=True)

    return log


DEPENDS = {"mask_self_report": ("mask",)}
OPT_IN = ("benchmarks", "diagnostics", "diagnostics_torch")


def selected_suites(selection: str) -> list[str]:
    named = {n.strip() for n in selection.split(",")}
    return [n for n in resolve(selection) if n not in OPT_IN or n in named]


def generation_settings(config: EvalConfig, checkpoint: str) -> dict:
    path = Path(checkpoint) / "generation_config.json"
    return {
        "generation_config": json.loads(path.read_text()) if path.is_file() else {},
        "max_tokens": config.max_tokens,
        "answer_mode": config.answer_mode,
        "system_prompt": config.system_prompt,
    }


def result_key(config: EvalConfig, spec: ModelSpec, suite: str) -> str:
    depends = {name: result_key(config, spec, name) for name in DEPENDS.get(suite, ())}
    extra = {"checkpoint": fingerprint.checkpoint(spec.checkpoint), "depends": depends}
    return fingerprint.suite_key(config, spec, suite, config.suite_args(suite), extra)


def result_keys(config: EvalConfig, spec: ModelSpec, suites: list[str]) -> dict[str, str]:
    return {name: result_key(config, spec, name) for name in suites}


def stored_key(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("key")
    except (json.JSONDecodeError, OSError):
        return None


def _write(
    path: Path,
    result: Result,
    spec: ModelSpec,
    suite: str,
    seconds: float,
    judge_blocked: int | None = None,
    budget_capped: int | None = None,
    *,
    key: str = "",
    settings: dict | None = None,
    prompt_too_long: int | None = None,
) -> None:
    payload = {
        "model": spec.name,
        "checkpoint": spec.checkpoint,
        "checkpoint_fingerprint": fingerprint.checkpoint(spec.checkpoint),
        "key": key,
        "role": spec.role,
        "domain": spec.domain,
        "suite": suite,
        "commit": STARTUP_COMMIT,
        "seconds": round(seconds, 1),
        "settings": settings or {},
        "judge_blocked": judge_blocked,
        "budget_capped": budget_capped,
        "prompt_too_long": prompt_too_long,
        "summary": result.summary,
        "rows": result.rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default))


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


@contextlib.contextmanager
def _guarded(out_dir: Path, model: str, suite: str, failures: list[str]):
    error = out_dir / f"{suite}.error.txt"
    try:
        yield
    except Exception:
        text = traceback.format_exc()
        error.write_text(text)
        failures.append(suite)
        _log(model, suite)(f"failed, continuing with the next suite: {text.strip().splitlines()[-1]} (see {error})")
    else:
        error.unlink(missing_ok=True)


def run_model(config: EvalConfig, spec: ModelSpec, suites: list[str], *, force: bool) -> list[str]:
    os.environ.update(loopback_no_proxy(os.environ))
    family = resolve_family(config.family)
    out_dir = config.output_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = result_keys(config, spec, suites)
    settings = generation_settings(config, spec.checkpoint)
    todo = [s for s in suites if force or stored_key(out_dir / f"{s}.json") != keys[s]]
    for name in todo:
        stored = out_dir / f"{name}.json"
        if stored.exists() and stored_key(stored) != keys[name]:
            stored.replace(out_dir / f"{name}.stale.json")
            print(f"{spec.name}/{name}: stored result has another checkpoint, code or settings key; moved aside")
    if not todo:
        print(f"{spec.name}: every suite has results for this checkpoint, code and settings; use --force 1 to rerun")
        return []
    failures: list[str] = []
    modules = {name: load(name) for name in todo}
    paths = {n: m.PATH if isinstance(m.PATH, str) else m.PATH(config) for n, m in modules.items()}
    server_suites = [n for n in todo if paths[n] == "server"]
    torch_suites = [n for n in todo if paths[n] == "torch"]
    has_judge = bool(config.judge_base_url and config.judge_model and "$" not in config.judge_api_key)
    judges = {
        name: ChatClient(config.judge_base_url, config.judge_model, concurrency=8, api_key=config.judge_api_key)
        if has_judge
        else None
        for name in todo
    }

    def blocked(name: str) -> int | None:
        judge = judges[name]
        return None if judge is None else judge.num_blocked

    pending: dict[str, tuple[Result, float, Context, int, int]] = {}

    def context(name: str) -> Context:
        return Context(
            config=config,
            model=spec,
            family=family,
            args=config.suite_args(name),
            out_dir=out_dir,
            seed=config.seed,
            log=_log(spec.name, name),
            judge=judges[name],
            key=keys[name],
        )

    if server_suites:
        with VllmServer(
            spec.checkpoint,
            gpu=config.resolved_gpu,
            allocated=config.scheduler_allocated,
            log_path=out_dir / f"vllm-{'+'.join(server_suites)[:60]}-{socket.gethostname()}-{os.getpid()}.log",
            port=config.port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            reasoning_parser=config.resolved_reasoning_parser,
            moe_backend=config.moe_backend,
            attention_backend=config.attention_backend,
        ) as server:
            builder = ProbeBuilder(
                spec.checkpoint, family, system_prompt=SYSTEM_PROMPTS[config.system_prompt], thinking=config.answer_mode
            )
            client = ChatClient(
                server.base_url,
                spec.checkpoint,
                concurrency=config.concurrency,
                renderer=builder.renderer,
                thinking=config.answer_mode,
                banned_ids=builder.banned_ids(config.answer_mode),
                template_kwargs=family.thinking.template_kwargs(config.answer_mode),
                markers=family.reasoning,
                max_model_len=config.max_model_len,
            )
            for name in server_suites:
                with _guarded(out_dir, spec.name, name, failures):
                    ctx = context(name)
                    ctx.client, ctx.builder = client, builder
                    capped_before = client.num_budget_capped
                    too_long_before = client.num_prompt_too_long
                    started = time.time()
                    ctx.log("start")
                    result = modules[name].run(ctx)
                    if result.pending or result.compare or result.finish is not None:
                        pending[name] = (
                            result,
                            started,
                            ctx,
                            client.num_budget_capped - capped_before,
                            client.num_prompt_too_long - too_long_before,
                        )
                        ctx.log(
                            f"{len(result.pending)} probes wait for torch labels, {len(result.compare)} for comparison"
                        )
                    else:
                        _write(
                            out_dir / f"{name}.json",
                            result,
                            spec,
                            name,
                            time.time() - started,
                            blocked(name),
                            client.num_budget_capped - capped_before,
                            key=keys[name],
                            settings=settings,
                            prompt_too_long=client.num_prompt_too_long - too_long_before,
                        )
                        ctx.log(f"done in {time.time() - started:.0f}s: {_brief(result.summary)}")
    standalone = [n for n in torch_suites if not getattr(modules[n], "NEEDS_TORCH_MODEL", True)]
    torch_suites = [n for n in torch_suites if n not in standalone]
    for name in standalone:
        with _guarded(out_dir, spec.name, name, failures):
            ctx = context(name)
            started = time.time()
            ctx.log("start")
            result = modules[name].run(ctx)
            _write(
                out_dir / f"{name}.json",
                result,
                spec,
                name,
                time.time() - started,
                blocked(name),
                key=keys[name],
                settings=settings,
            )
            ctx.log(f"done in {time.time() - started:.0f}s: {_brief(result.summary)}")
    if torch_suites or pending:
        gpu = config.resolved_gpu.split(",")[0]
        device = f"cuda:{gpu}"
        try:
            if not config.scheduler_allocated:
                check_gpu_available(gpu)
            torch_model = TorchModel(spec.checkpoint, family.router_scores, device=device)
        except Exception:
            text = traceback.format_exc()
            for name in [*pending, *torch_suites]:
                (out_dir / f"{name}.error.txt").write_text(text)
                failures.append(name)
            print(
                f"{spec.name}: torch model failed to load ({text.strip().splitlines()[-1]}); failed suites {failures}; "
                f"tracebacks in {out_dir}/<suite>.error.txt",
                flush=True,
            )
            return failures
        labeler = TorchLabeler(torch_model)
        builder = ProbeBuilder(
            spec.checkpoint, family, system_prompt=SYSTEM_PROMPTS[config.system_prompt], thinking=config.answer_mode
        )
        for name, (result, started, suite_ctx, capped, too_long) in pending.items():
            with _guarded(out_dir, spec.name, name, failures):
                assert result.finish is not None
                suite_ctx.torch, suite_ctx.labeler = torch_model, labeler
                labels = (
                    labeler.label([p.prefix_ids for p in result.pending], [[p.request] for p in result.pending])
                    if result.pending
                    else []
                )
                for probe, (label,) in zip(result.pending, labels, strict=True):
                    probe.label = label
                    builder.score(probe)
                if result.compare:
                    torch_labels = labeler.label(
                        [p.prefix_ids for p in result.compare], [[p.request] for p in result.compare]
                    )
                    for probe, (label,) in zip(result.compare, torch_labels, strict=True):
                        probe.meta["torch_label"] = label
                finished = result.finish(result.pending)
                _write(
                    out_dir / f"{name}.json",
                    finished,
                    spec,
                    name,
                    time.time() - started,
                    blocked(name),
                    capped,
                    key=keys[name],
                    settings=settings,
                    prompt_too_long=too_long,
                )
                _log(spec.name, name)(f"done: {_brief(finished.summary)}")
        for name in torch_suites:
            with _guarded(out_dir, spec.name, name, failures):
                ctx = context(name)
                ctx.torch, ctx.labeler, ctx.builder = torch_model, labeler, builder
                started = time.time()
                ctx.log("start")
                result = modules[name].run(ctx)
                _write(
                    out_dir / f"{name}.json",
                    result,
                    spec,
                    name,
                    time.time() - started,
                    blocked(name),
                    key=keys[name],
                    settings=settings,
                )
                ctx.log(f"done in {time.time() - started:.0f}s: {_brief(result.summary)}")
        del torch_model
        import torch

        torch.cuda.empty_cache()
    if failures:
        print(f"{spec.name}: failed suites {failures}; tracebacks in {out_dir}/<suite>.error.txt", flush=True)
    return failures


def _brief(summary: dict) -> str:
    items = []
    for key, value in summary.items():
        if isinstance(value, float):
            items.append(f"{key}={value:.3f}")
        elif isinstance(value, dict) and "acc" in value:
            acc = value["acc"]
            items.append(f"{key}={'–' if acc is None else f'{acc:.3f}'}(n={value.get('n', 0)})")
        if len(items) >= 8:
            break
    return " ".join(items)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "run":
        config, extra = load_config(rest)
        suites = selected_suites(",".join(config.suites))
        force = extra.get("force", "").lower() in ("1", "true", "yes")
        failed = {}
        for name in config.model_names(",".join(config.run_models)):
            failures = run_model(config, config.models[name], suites, force=force)
            if failures:
                failed[name] = failures
        if failed:
            raise SystemExit(f"failed suites per model: {failed}")
    elif command == "prefetch":
        from evals.datasets import prefetch

        prefetch()
    elif command == "report":
        from evals.report import report

        print(report(Path(rest[0])))
    else:
        raise SystemExit(f"unknown command {command!r}; use run, prefetch or report")
