import datetime
import os
import re

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

COLLECTIVE_TIMEOUT = datetime.timedelta(minutes=30)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup() -> tuple[int, int, str]:
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) == 1:
        return 0, 1, "cuda"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=COLLECTIVE_TIMEOUT)
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"


def teardown() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def rank_and_world() -> tuple[int, int]:
    if not is_distributed():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def is_main() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def report_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    if is_main():
        print(f"memory after {label}: {peak:.1f} GiB peak allocated, {reserved:.1f} GiB reserved", flush=True)
    torch.cuda.reset_peak_memory_stats()


def all_reduce_sum_tensor(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_max(value: int) -> int:
    if not is_distributed():
        return value
    tensor = torch.tensor([value], dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def uniform_batches(items: list, batch_size: int, filler, *, max_tokens: int = 0, layout: str = "padded"):
    if max_tokens:
        from proprioception.batching import token_chunks

        chunks = token_chunks(items, batch_size, max_tokens, 1, layout, keep_order=True)
    else:
        chunks = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    for index in range(all_reduce_max(len(chunks))):
        chunk = chunks[index] if index < len(chunks) else []
        yield (chunk or [filler]), len(chunk)


def all_gather_list(items: list) -> list:
    if not is_distributed():
        return items
    gathered: list[list] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, items)
    return [item for rank_items in gathered for item in rank_items]


def shard_slice(items: list, rank: int, world_size: int) -> list:
    if world_size == 1:
        return items
    per_rank = (len(items) + world_size - 1) // world_size
    return items[rank * per_rank : (rank + 1) * per_rank]


def shard_model(model: torch.nn.Module, layers: list[torch.nn.Module]) -> torch.nn.Module:
    if not is_distributed():
        return model
    if isinstance(model, FSDPModule) or any(isinstance(module, FSDPModule) for module in model.modules()):
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not getattr(model, "expert_parallel", False):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return
    from torchtitan.distributed.utils import clip_grad_norm_

    clip_grad_norm_(parameters, max_norm=max_norm, ep_enabled=True)


def full_state_dict(model: torch.nn.Module) -> dict:
    model = getattr(model, "publication_module", model)
    if not is_distributed():
        return model.state_dict()
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


_LAYER_UNIT = re.compile(r"^(.*?\.layers\.\d+)\.")


def export_owners(sizes: dict[str, int], world: int) -> dict[str, int]:
    groups: dict[str, list[str]] = {}
    for key in sizes:
        match = _LAYER_UNIT.match(key)
        groups.setdefault(match.group(1) if match else "", []).append(key)
    load = [0] * world
    owners: dict[str, int] = {}
    for _unit, keys in sorted(groups.items(), key=lambda item: (-sum(sizes[k] for k in item[1]), item[0])):
        rank = load.index(min(load))
        load[rank] += sum(sizes[k] for k in keys)
        for key in keys:
            owners[key] = rank
    return owners


def owned_full_state_dict(model: torch.nn.Module) -> dict:
    model = getattr(model, "publication_module", model)
    if not is_distributed():
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}
    rank, world = rank_and_world()
    sharded = {k: v for k, v in get_model_state_dict(model).items() if isinstance(v, torch.Tensor)}
    owners = export_owners({key: value.numel() * value.element_size() for key, value in sharded.items()}, world)
    owned = {}
    for key, value in sharded.items():
        full = value.full_tensor() if isinstance(value, DTensor) else value
        if owners[key] == rank:
            owned[key] = full.detach().cpu()
    return owned
