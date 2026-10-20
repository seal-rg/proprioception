import pickle
import shutil
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from proprioception import distributed

_pending: tuple[object, Path, float] | None = None
_write_group: torch.distributed.ProcessGroup | None = None


def _process_group() -> torch.distributed.ProcessGroup | None:
    global _write_group
    if _write_group is None and distributed.is_distributed():
        group = torch.distributed.new_group(backend="gloo", timeout=distributed.COLLECTIVE_TIMEOUT)
        assert isinstance(group, torch.distributed.ProcessGroup)
        _write_group = group
    return _write_group


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _checkpoint_number(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def _is_checkpoint(path: Path) -> bool:
    return path.is_dir() and "-" in path.name and path.name.rsplit("-", 1)[1].isdigit()


def latest_checkpoint(directory: Path) -> Path | None:
    complete = [p for p in directory.glob("*-*") if _is_checkpoint(p) and (p / ".metadata").exists()]
    return max(complete, key=_checkpoint_number, default=None)


def _prune_older(path: Path) -> None:
    if distributed.is_main():
        for sibling in path.parent.glob("*-*"):
            if _is_checkpoint(sibling) and _checkpoint_number(sibling) < _checkpoint_number(path):
                shutil.rmtree(sibling, ignore_errors=True)


def wait_for_checkpoint() -> float | None:
    global _pending
    if _pending is None:
        return None
    future, path, start = _pending
    future.result()
    _pending = None
    _prune_older(path)
    return (path / ".metadata").stat().st_mtime - start


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    state: dict,
    threads: int = 1,
    background: bool = True,
) -> float:
    previous = wait_for_checkpoint()
    if previous is not None and distributed.is_main():
        print(f"previous checkpoint written in {previous:.1f}s")
    start = time.time()
    model_state, optim_state = get_state_dict(model, optimizer)
    rank, _ = distributed.rank_and_world()
    payload = {
        "model": model_state,
        "optim": optim_state,
        "state": pickle.dumps(state),
        "rng": {f"rank{rank}": _rng_state()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = dcp.FileSystemWriter(str(path), thread_count=threads)
    if background:
        global _pending
        future = dcp.async_save(payload, storage_writer=writer, process_group=_process_group())
        _pending = (future, path, start)
    else:
        dcp.save(payload, storage_writer=writer, process_group=_process_group())
        _prune_older(path)
    return time.time() - start


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    model_state, optim_state = get_state_dict(model, optimizer)
    rank, _ = distributed.rank_and_world()
    payload = {"model": model_state, "optim": optim_state, "state": b"", "rng": {f"rank{rank}": _rng_state()}}
    saved = set(dcp.FileSystemReader(str(path)).read_metadata().state_dict_metadata)
    flattened, _ = flatten_state_dict({"model": payload["model"]})
    missing = sorted(key for key in flattened if key not in saved)
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} model tensors, first {missing[:3]}")
    dcp.load(
        payload,
        checkpoint_id=str(path),
        planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
        process_group=_process_group(),
    )
    set_state_dict(model, optimizer, model_state_dict=payload["model"], optim_state_dict=payload["optim"])
    rng = payload["rng"][f"rank{rank}"]
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"])
    return pickle.loads(payload["state"])
