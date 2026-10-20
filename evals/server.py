import contextlib
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from evals.config import REPO

VLLM = REPO / ".venv" / "bin" / "vllm"
FAILURE_MARKERS = ("Engine core initialization failed", "EngineDeadError", "torch.OutOfMemoryError")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def gpu_users(gpu: str) -> set[str]:
    users: set[str] = set()
    for index in gpu.split(","):
        out = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        for pid in out:
            owner = subprocess.run(["ps", "-o", "user=", "-p", pid.strip(",")], capture_output=True, text=True).stdout
            users.add(owner.strip() or "unknown")
    return users


def gpu_is_free(gpu: str) -> bool:
    return not gpu_users(gpu)


def check_gpu_available(gpu: str) -> None:
    users = gpu_users(gpu)
    mine = os.environ.get("USER", "")
    if users - {mine}:
        raise RuntimeError(f"GPU {gpu} is in use by {sorted(users - {mine})}; refusing to use it")
    if users:
        print(f"note: GPU {gpu} already carries processes of {mine}; sharing it", flush=True)


def _exit_on_term(signum, frame) -> None:
    raise SystemExit(128 + signum)


class VllmServer:
    def __init__(
        self,
        checkpoint: str,
        *,
        gpu: str,
        log_path: Path,
        port: int = 0,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 16384,
        max_num_seqs: int = 128,
        reasoning_parser: str = "",
        moe_backend: str = "triton",
        attention_backend: str = "",
        extra_args: list[str] | None = None,
        startup_timeout: float = 1800,
        allocated: bool = False,
    ):
        self.checkpoint = checkpoint
        self.gpu = gpu
        self.allocated = allocated
        self.port = port or free_port()
        self.log_path = log_path
        self.args = [
            "serve",
            checkpoint,
            "--enable-return-routed-experts",
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            str(max_num_seqs),
            "--tensor-parallel-size",
            str(len(gpu.split(","))),
            "--uvicorn-log-level",
            "warning",
            *(["--reasoning-parser", reasoning_parser] if reasoning_parser else []),
            *(["--moe-backend", moe_backend] if moe_backend and moe_backend != "auto" else []),
            *(["--attention-backend", attention_backend] if attention_backend else []),
            *(extra_args or []),
        ]
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "VllmServer":
        self._previous_handler = signal.signal(signal.SIGTERM, _exit_on_term)
        if not self.allocated:
            check_gpu_available(self.gpu)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost"}
        if not self.allocated:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        with self.log_path.open("w") as log:
            self.process = subprocess.Popen(
                [str(VLLM), *self.args], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.process.returncode}; see {self.log_path}{self._log_tail()}")
            try:
                response = httpx.get(f"{self.base_url}/v1/models", timeout=2, trust_env=False)
                if response.status_code == 200 and self.checkpoint in response.text:
                    return
            except httpx.HTTPError:
                pass
            text = self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
            if any(marker in text for marker in FAILURE_MARKERS):
                raise RuntimeError(f"vLLM failed to start; see {self.log_path}{self._log_tail()}")
            time.sleep(3)
        raise TimeoutError(f"vLLM not ready after {self.startup_timeout}s; see {self.log_path}{self._log_tail()}")

    def _log_tail(self, num_lines: int = 20) -> str:
        if not self.log_path.exists():
            return ""
        lines = self.log_path.read_text(errors="replace").splitlines()[-num_lines:]
        return "\n" + "\n".join(lines) if lines else ""

    def close(self) -> None:
        if getattr(self, "_previous_handler", None) is not None:
            signal.signal(signal.SIGTERM, self._previous_handler)
            self._previous_handler = None
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.process = None
        for _ in range(20):
            if self.allocated or gpu_is_free(self.gpu):
                return
            time.sleep(1)
