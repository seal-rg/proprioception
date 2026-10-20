import inspect
import json
import shutil
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors import safe_open

import torch
from torch.distributed.tensor import DTensor

from proprioception import distributed, modeling

if TYPE_CHECKING:
    from proprioception.families import ModelFamily


def decoder_layer_list(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.ModuleList)]
    if not candidates:
        raise ValueError("no decoder layer list found on this model")
    for name, module_list in candidates:
        if any(type(inner).__name__.endswith("Router") for block in module_list for inner in block.modules()):
            return name, module_list
    return max(candidates, key=lambda item: sum(p.numel() for p in item[1].parameters()))


def decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return list(decoder_layer_list(model)[1])


def language_model(model: torch.nn.Module) -> torch.nn.Module:
    name = decoder_layer_list(model)[0]
    parent, _, _ = name.rpartition(".")
    return model.get_submodule(parent) if parent else model


def is_router(module: torch.nn.Module) -> bool:
    return type(module).__name__.endswith("Router") or hasattr(module, "route_tokens_to_experts")


def router_modules(model: torch.nn.Module) -> dict[int, tuple[torch.nn.Module, torch.nn.Module]]:
    routers = {}
    for index, layer in enumerate(decoder_layers(model)):
        for name, module in layer.named_modules():
            if is_router(module):
                parent = name.rpartition(".")[0]
                routers[index] = (module, layer.get_submodule(parent) if parent else layer)
                break
    return routers


_capturing_classes: dict[tuple[type, bool, bool, bool], type] = {}

ROUTER_SCORES = ("softmax", "sigmoid", "probabilities")


def _detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detached(entry) for entry in value)
    return value


def _install_capture(module: torch.nn.Module, *, keep_input: bool = False, keep_output: bool = False) -> None:
    base = type(module)
    if getattr(base, "_is_routing_capture", False):
        raise ValueError(f"{base.__name__} already captures")
    routes = hasattr(module, "route_tokens_to_experts")
    key = (base, keep_input, keep_output, routes)
    if key not in _capturing_classes:

        class Capturing(base):
            _is_routing_capture = True

            def forward(self, *args, **kwargs):
                output = base.forward(self, *args, **kwargs)
                if keep_input:
                    self._captured_input = _detached(args[0] if args else None)
                if keep_output:
                    self._captured_output = _detached(output)
                return output

            def route_tokens_to_experts(self, router_logits, *args, **kwargs):
                original = base.route_tokens_to_experts
                selected, weights = original(self, router_logits, *args, **kwargs)
                self._captured_output = (router_logits.detach(), selected.detach())
                return selected, weights

        if not routes:
            del Capturing.route_tokens_to_experts
        Capturing.__name__ = f"Capturing{base.__name__}"
        Capturing.__qualname__ = Capturing.__name__
        _capturing_classes[key] = Capturing
    module.__class__ = _capturing_classes[key]
    _reset_capture(module)


def _reset_capture(module: torch.nn.Module) -> None:
    module.__dict__["_captured_input"] = None
    module.__dict__["_captured_output"] = None


def _captured(module: torch.nn.Module, which: str):
    return module.__dict__[f"_captured_{which}"]


class RoutingModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, family: "ModelFamily"):
        super().__init__()
        self.model = model
        if family.router_scores not in ROUTER_SCORES:
            raise ValueError(
                f"family {family.name} declares router_scores={family.router_scores!r}, not {ROUTER_SCORES}"
            )
        self.router_scores = family.router_scores
        self.num_experts = family.num_experts
        self.routers = router_modules(model)
        self._num_layers = len(decoder_layers(model))
        self._scores: list[torch.Tensor | None] = [None] * self._num_layers
        self._selection: list[torch.Tensor | None] = [None] * self._num_layers
        self._selected: list[torch.Tensor | None] = [None] * self._num_layers
        self._readers: dict[int, str] = {}
        for index, (router, owner) in self.routers.items():
            self._watch(index, router, owner)
        self.accepts_head_columns = "logits_to_keep" in inspect.signature(model.forward).parameters
        self._init_lens_state(model)

    def _watch(self, index: int, router: torch.nn.Module, owner: torch.nn.Module) -> None:
        if hasattr(router, "route_tokens_to_experts"):
            _install_capture(router)
            self._readers[index] = "route_tokens"
            return
        gate = getattr(router, "gate", None)
        if isinstance(gate, torch.nn.Module):
            _install_capture(gate, keep_output=True)
            _install_capture(router, keep_output=True)
            self._readers[index] = "logits"
        elif isinstance(gate, torch.Tensor):
            _install_capture(router, keep_input=True, keep_output=True)
            self._readers[index] = "weight_gate"
        else:
            _install_capture(router, keep_output=True)
            self._readers[index] = "output"

    def _watched_modules(self, index: int) -> list[torch.nn.Module]:
        router, _ = self.routers[index]
        gate = getattr(router, "gate", None)
        return [router, gate] if isinstance(gate, torch.nn.Module) else [router]

    def _read_captures(self, index: int) -> None:
        router, owner = self.routers[index]
        reader = self._readers[index]
        output = _captured(router, "output")
        if output is None:
            return
        if reader == "route_tokens":
            router_logits, selected = output
            self._scores[index] = self._ranked_scores(router, router_logits, owner)
            self._selection[index] = selected
            return
        if reader == "logits":
            gate = router.gate
            assert isinstance(gate, torch.nn.Module)
            self._scores[index] = self._ranked_scores(router, _captured(gate, "output"), owner)
        elif reader == "weight_gate":
            weight = router.gate
            assert isinstance(weight, torch.Tensor)
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            logits = torch.nn.functional.linear(_captured(router, "input").float(), weight.float())
            self._scores[index] = self._ranked_scores(router, logits, owner)
        else:
            self._scores[index] = self._ranked_scores(router, self._per_expert_output(router, output), owner)
        self._selection[index] = self._selection_from(output)

    @staticmethod
    def _selection_bias(router: torch.nn.Module, owner: torch.nn.Module | None) -> torch.Tensor | None:
        holders = [router, getattr(router, "gate", None), owner]
        found = []
        for holder in holders:
            if not isinstance(holder, torch.nn.Module):
                continue
            for name in ("expert_bias", "e_score_correction_bias"):
                bias = getattr(holder, name, None)
                if isinstance(bias, torch.Tensor) and not any(bias is other for other in found):
                    found.append(bias.full_tensor() if isinstance(bias, DTensor) else bias)
        if not found:
            return None
        total = found[0].float()
        for bias in found[1:]:
            total = total + bias.float()
        return total

    def _ranked_scores(
        self, router: torch.nn.Module, values: torch.Tensor, owner: torch.nn.Module | None = None
    ) -> torch.Tensor:
        if self.router_scores == "softmax":
            scores = torch.softmax(values.float(), dim=-1)
        elif self.router_scores == "sigmoid":
            scores = torch.sigmoid(values.float())
        else:
            scores = values.float()
        bias = self._selection_bias(router, owner)
        return scores if bias is None else scores + bias.to(scores.device)

    def _per_expert_output(self, router: torch.nn.Module, output) -> torch.Tensor:
        for entry in output if isinstance(output, tuple) else (output,):
            if isinstance(entry, torch.Tensor) and entry.is_floating_point() and entry.shape[-1] == self.num_experts:
                return entry.reshape(-1, self.num_experts)
        raise ValueError(f"{type(router).__name__} returned no [tokens, {self.num_experts}] values")

    @staticmethod
    def _selection_from(output) -> torch.Tensor | None:
        for entry in output if isinstance(output, tuple) else (output,):
            if isinstance(entry, torch.Tensor) and not entry.is_floating_point() and entry.dim() == 2:
                return entry
        return None

    @staticmethod
    def _output_scale(router: torch.nn.Module, owner: torch.nn.Module | None) -> torch.Tensor | None:
        for holder in (router, owner):
            scale = getattr(holder, "per_expert_scale", None) if isinstance(holder, torch.nn.Module) else None
            if isinstance(scale, torch.Tensor):
                return scale.full_tensor() if isinstance(scale, DTensor) else scale
        return None

    @staticmethod
    def _top_selected(scores: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        selection = selection.to(scores.device).long()
        ranked = scores.gather(1, selection)
        return selection.gather(1, ranked.argmax(dim=1, keepdim=True)).squeeze(1)

    def _init_lens_state(self, model: torch.nn.Module) -> None:
        self._lens_request: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._lens_hidden: torch.Tensor | None = None
        self._lens_normed: torch.Tensor | None = None
        self._lens_norm_rows = 0
        self._lens_head_rows = 0
        self._lens_pending: int | None = None
        self.lens_logits: torch.Tensor | None = None
        inner = language_model(model)
        norm = getattr(inner, "norm", None) or getattr(inner, "norm_f", None) or getattr(inner, "embedding_norm", None)
        head = getattr(model, "lm_head", None)
        self._lens_norm_and_head: tuple[torch.nn.Module, torch.nn.Module] | None = None
        if isinstance(norm, torch.nn.Module) and isinstance(head, torch.nn.Module):
            self._lens_norm_and_head = (norm, head)
        self.accepts_lens_request = self._lens_norm_and_head is not None
        self._lens_hooks_installed = False

    def _install_lens_hooks(self) -> None:
        assert self._lens_norm_and_head is not None
        norm, head = self._lens_norm_and_head
        for index, layer in enumerate(decoder_layers(self.model)):
            layer.register_forward_hook(self._layer_lens_hook(index))
        norm.register_forward_pre_hook(self._lens_norm_pre_hook)
        norm.register_forward_hook(self._lens_norm_hook)
        head.register_forward_pre_hook(self._lens_head_pre_hook, with_kwargs=True)
        head.register_forward_hook(self._lens_head_hook)
        self.model.register_forward_hook(self._root_lens_split)
        self._lens_hooks_installed = True

    def _layer_lens_hook(self, index: int):
        def hook(_module, _inputs, output):
            if self._lens_request is None:
                return None
            layers, rows, cols = self._lens_request
            hit = torch.eq(layers, index).nonzero(as_tuple=True)[0]
            if not hit.numel():
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            stash = self._lens_hidden
            if stash is None:
                stash = self._lens_hidden = hidden.new_zeros((layers.numel(), hidden.shape[-1]))
            stash[hit] = hidden.detach()[rows[hit], cols[hit]]
            return None

        return hook

    @staticmethod
    def _append_rows(x: torch.Tensor, vectors: torch.Tensor) -> tuple[torch.Tensor, int]:
        _, width, hidden = x.shape
        rows = -(-vectors.shape[0] // width)
        block = x.new_zeros((rows * width, hidden))
        block[: vectors.shape[0]] = vectors.to(x.dtype)
        return torch.cat([x, block.view(rows, width, hidden)], dim=0), rows

    @staticmethod
    def _take_rows(x: torch.Tensor, rows: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        keep = x.shape[0] - rows
        return x[:keep], x[keep:].reshape(-1, x.shape[-1])[:count]

    def _lens_norm_pre_hook(self, _module, args):
        if self._lens_hidden is None:
            return None
        extended, self._lens_norm_rows = self._append_rows(args[0], self._lens_hidden)
        return (extended, *args[1:])

    def _lens_norm_hook(self, _module, _args, output):
        if self._lens_hidden is None:
            return None
        kept, self._lens_normed = self._take_rows(output, self._lens_norm_rows, self._lens_hidden.shape[0])
        return kept

    def _lens_head_pre_hook(self, _module, args, kwargs):
        if self._lens_normed is None:
            return None
        companions = [*args[1:], *kwargs.values()]
        if any(argument is not None for argument in companions):
            raise ValueError("the lens rides on the head's own call, which cannot also carry labels or temperatures")
        extended, self._lens_head_rows = self._append_rows(args[0], self._lens_normed)
        return (extended, *args[1:]), kwargs

    def _lens_head_hook(self, _module, _args, output):
        if self._lens_normed is None:
            return None
        self._lens_pending = self._lens_normed.shape[0]
        return None

    def _root_lens_split(self, _module, _args, output):
        count = self._lens_pending
        if count is None:
            return None
        self._lens_pending = None
        logits = output["logits"] if isinstance(output, dict) else output
        kept, tail = self._take_rows(logits, self._lens_head_rows, count)
        self.lens_logits = tail.detach()
        if isinstance(output, dict):
            output["logits"] = kept
            return output
        return kept

    def check_capture_is_complete(self) -> None:
        missing = sorted(set(self.routers) - {index for index, value in enumerate(self._scores) if value is not None})
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(self.routers)} routed layers reported no routing (layers {missing[:8]}): "
                "the capture did not run on them"
            )

    def forward(self, input_ids, position_ids=None, head_columns=None, lens_request=None):
        for index in range(self._num_layers):
            self._scores[index] = None
            self._selection[index] = None
            self._selected[index] = None
        for index in self.routers:
            for module in self._watched_modules(index):
                _reset_capture(module)
        if lens_request is not None and not self.accepts_lens_request:
            raise ValueError(f"{type(self.model).__name__} exposes no final norm and head to read a lens through")
        if lens_request is not None and not self._lens_hooks_installed:
            self._install_lens_hooks()
        self._lens_request = lens_request
        self._lens_hidden = None
        self._lens_normed = None
        self.lens_logits = None
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(input_ids.shape[0], -1)
        arguments = {"input_ids": input_ids, "position_ids": position_ids}
        if head_columns is not None:
            arguments["logits_to_keep"] = head_columns
        try:
            output = self.model(**arguments)
        finally:
            self._lens_request = None
            self._lens_hidden = None
            self._lens_normed = None
        for index in self.routers:
            self._read_captures(index)
        if lens_request is not None and self.lens_logits is None:
            raise ValueError("the lens request produced no logits; the norm or head hook did not run")
        logits = output["logits"] if isinstance(output, dict) else output.logits
        if head_columns is not None and logits.shape[1] != head_columns.numel():
            raise ValueError(
                f"the model returned logits for {logits.shape[1]} positions where "
                f"{head_columns.numel()} head columns were requested"
            )
        batch, tokens = input_ids.shape
        scores = {index: value.view(batch, tokens, -1) for index, value in enumerate(self._scores) if value is not None}
        for index, selection in enumerate(self._selection):
            if selection is None:
                continue
            layer_scores = self._scores[index]
            if layer_scores is None:
                raise ValueError(f"layer {index} reported a selection but no scores to rank it by")
            self._selected[index] = self._top_selected(layer_scores, selection).view(batch, tokens)
        return logits, scores

    @property
    def publication_module(self) -> torch.nn.Module:
        return self.model

    def routing_weights(self, layers: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        bias = scores.new_zeros((self._num_layers, scores.shape[-1]), dtype=torch.float32)
        scale = torch.ones_like(bias)
        for index, (router, owner) in self.routers.items():
            selection_bias = self._selection_bias(router, owner)
            if selection_bias is not None:
                bias[index] = selection_bias.to(bias.device)
            output_scale = self._output_scale(router, owner)
            if output_scale is not None:
                scale[index] = output_scale.float().to(bias.device)
        rows = layers.to(bias.device)
        return (scores.float() - bias[rows]) * scale[rows]

    def selected_experts(self, layers, rows, positions) -> torch.Tensor:
        present = [index for index, entry in enumerate(self._selected) if entry is not None]
        stacked = torch.stack([self._selected[index] for index in present])
        slot = torch.full((self._num_layers,), len(present), dtype=torch.long, device=stacked.device)
        slot[torch.tensor(present, device=stacked.device)] = torch.arange(len(present), device=stacked.device)
        return stacked[slot[layers.to(stacked.device)], rows.to(stacked.device), positions.to(stacked.device)]

    def label_disagreements(self, scores: dict[int, torch.Tensor]) -> dict[int, tuple[int, int]]:
        if not any(entry is not None for entry in self._selected):
            raise ValueError("routers reported no selection to check the scores against")
        disagreements = {}
        for index, selected in enumerate(self._selected):
            if selected is None:
                continue
            layer_scores = scores[index]
            at_selected = layer_scores.gather(-1, selected.unsqueeze(-1).long()).squeeze(-1)
            outranked = at_selected < layer_scores.max(dim=-1).values
            count = int(outranked.sum())
            if count:
                disagreements[index] = (count, int(outranked.numel()))
        return disagreements


def unfinished_conversion(checkpoint: str | Path) -> Path | None:
    for name in ("prime", "hf"):
        cache = Path(checkpoint) / name
        if not cache.is_dir():
            continue
        if not (cache / "model.safetensors.index.json").exists() and not (cache / "model.safetensors").exists():
            return cache
    return None


def untrained_keys(checkpoint: str | Path) -> set[str]:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    depth = config.get("text_config", config).get("num_hidden_layers")
    untrained = set()
    for key in hub_weight_map(checkpoint):
        layer = re.search(r"\.layers\.(\d+)\.", key)
        if key.startswith("mtp.") or (depth is not None and layer is not None and int(layer.group(1)) >= depth):
            untrained.add(key)
    return untrained


def hub_weight_map(checkpoint: str | Path) -> dict[str, Path]:
    root = Path(checkpoint)
    index = root / "model.safetensors.index.json"
    if index.exists():
        return {key: root / shard for key, shard in json.loads(index.read_text())["weight_map"].items()}
    weight_map = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                weight_map[key] = shard
    return weight_map


def _drop_tied_head(checkpoint: str | Path, state_dict: dict) -> None:
    head = state_dict.get("lm_head.weight")
    if head is None:
        return
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    text_config = config.get("text_config", config)
    if not text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        return
    embedding = state_dict.get("model.embed_tokens.weight")
    if embedding is not None and not torch.equal(head, embedding):
        raise ValueError("lm_head.weight differs from the tied embedding; exporting it would discard training")
    del state_dict["lm_head.weight"]


def is_prime_class(model: torch.nn.Module) -> bool:
    try:
        from prime_rl.trainer.models import PreTrainedModelPrimeRL
    except ImportError:
        return False
    module: Any = getattr(model, "publication_module", model)
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module
    return isinstance(module, PreTrainedModelPrimeRL)


def convert_export(checkpoint: str | Path) -> Path | None:
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.model import get_model
    from prime_rl.trainer.models import PreTrainedModelPrimeRL
    from prime_rl.trainer.weights import load_state_dict, load_state_dict_keys, save_state_dict

    root = Path(checkpoint)
    stale = unfinished_conversion(root)
    if stale is not None:
        shutil.rmtree(stale)
    target = root / "prime"
    if target.exists():
        return None
    model = get_model(ModelConfig(name=str(root), attn="sdpa"), device=torch.device("meta"))
    if not isinstance(model, PreTrainedModelPrimeRL):
        return None
    prime_class: Any = type(model)
    keys = dict.fromkeys(load_state_dict_keys(root))
    if not (prime_class.is_hf_state_dict(keys) and prime_class.is_prime_state_dict(dict.fromkeys(model.state_dict()))):
        return None
    partial = root / "prime.partial"
    if partial.exists():
        shutil.rmtree(partial)
    state_dict = load_state_dict(root)
    prime_class.convert_to_prime(state_dict)
    save_state_dict(state_dict, partial)
    partial.rename(target)
    return target


def to_hub_layout(model: torch.nn.Module, checkpoint: str | Path, state_dict: dict) -> dict:
    from transformers import PreTrainedModel

    module: Any = getattr(model, "publication_module", model)
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module
    if isinstance(module, PreTrainedModel):
        from prime_rl.trainer.models import PreTrainedModelPrimeRL

        if isinstance(module, PreTrainedModelPrimeRL):
            prime_class: Any = type(module)
            if prime_class.is_prime_state_dict(state_dict):
                prime_class.convert_to_hf(state_dict)
            trained = set(hub_weight_map(checkpoint)) - untrained_keys(checkpoint)
            if not any(re.search(r"\.experts\.\d+\.", key) for key in trained):
                modeling.stack_experts(state_dict)
        else:
            from transformers.core_model_loading import revert_weight_conversion

            state_dict = revert_weight_conversion(module, state_dict)
    elif hasattr(module, "to_hub_layout"):
        state_dict = module.to_hub_layout(state_dict)
    _drop_tied_head(checkpoint, state_dict)
    return state_dict


_invariant_library: torch.library.Library | None = None


def use_invariant_kernels(enabled: bool) -> None:
    if not enabled:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from vllm.model_executor.layers import batch_invariant

    global _invariant_library
    batch_invariant.enable_batch_invariant_mode()
    batch_invariant._batch_invariant_LIB._destroy()
    _invariant_library = torch.library.Library("aten", "IMPL")
    if batch_invariant.current_platform.is_device_capability_family(80):
        _invariant_library.impl("aten::mm", batch_invariant.mm_batch_invariant, "CUDA")
        _invariant_library.impl("aten::addmm", batch_invariant.addmm_batch_invariant, "CUDA")
    _invariant_library.impl("aten::_log_softmax", batch_invariant._log_softmax_batch_invariant, "CUDA")
    _invariant_library.impl("aten::softmax", batch_invariant.softmax_batch_invariant, "CUDA")
    _invariant_library.impl("aten::_softmax", batch_invariant.softmax_batch_invariant, "CUDA")
    _invariant_library.impl("aten::mean.dim", batch_invariant.mean_batch_invariant, "CUDA")
    _invariant_library.impl("aten::bmm", batch_invariant.bmm_batch_invariant, "CUDA", allow_override=True)


def load_moe(
    checkpoint: str,
    device: str,
    family: "ModelFamily",
    *,
    expert_parallel: bool = False,
    replicate: bool = False,
    activation_checkpointing: bool = True,
    compile: bool = True,
    experts_implementation: str | None = None,
    attn: str | None = None,
    optimization_dtype: str = "bfloat16",
) -> torch.nn.Module:
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.parallel_dims import (
        get_parallel_dims,
    )

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", timeout=distributed.COLLECTIVE_TIMEOUT)
    config = ModelConfig(
        name=checkpoint,
        optimization_dtype=optimization_dtype,
        fused_lm_head_token_chunk_size="disabled",
    )
    if attn is not None:
        config.attn = attn
    if not activation_checkpointing:
        config.ac = None
    if not compile:
        config.compile = None

    if (stale := unfinished_conversion(checkpoint)) is not None:
        raise ValueError(f"{stale} is an unfinished weight conversion; remove it so this run converts again")

    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    world_size = torch.distributed.get_world_size()
    if replicate:
        config.dp_replicate = world_size
    config.ep = min(world_size, 8) if expert_parallel and world_size > 1 else 1
    model = setup_model(config, get_parallel_dims(config))
    if experts_implementation is not None:
        model.config._experts_implementation = experts_implementation
    wrapped = RoutingModel(model, family)
    setattr(wrapped, "gradient_checkpointing", config.ac is not None)
    setattr(wrapped, "expert_parallel", config.ep > 1)
    return wrapped
