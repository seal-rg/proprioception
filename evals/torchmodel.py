import contextlib
import dataclasses
from collections.abc import Iterator

import torch

from evals.interventions import (
    LayerEdit,
    RouterEdit,
    RouterOutput,
    RouterSemantics,
    apply_layer_edit,
    repack_router_output,
    unpack_router_output,
)
from proprioception.families import moe_structure, read_config
from proprioception.rendering import ChatRenderer


def _find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers") and len(module) > 1:
            child = module[0]
            if any(hasattr(child, attr) for attr in ("feed_forward", "mlp", "block_sparse_moe", "moe")):
                return module
    raise ValueError("no decoder layer list found on the model")


def _find_router(layer: torch.nn.Module) -> torch.nn.Module | None:
    return next((module for module in layer.modules() if type(module).__name__.endswith("Router")), None)


def _find_final_norm(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "model", model)
    base = getattr(base, "language_model", base)
    for attr in ("norm", "embedding_norm", "final_norm", "final_layernorm", "ln_f"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise ValueError("no final norm found on the model")


@dataclasses.dataclass
class Forward:
    logits: torch.Tensor
    hidden: list[torch.Tensor]
    routing: dict[int, RouterOutput]
    lengths: list[int]


class TorchModel:
    def __init__(self, checkpoint: str, router_scores: str, device: str = "cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.checkpoint = checkpoint
        self.device = torch.device(device)
        config = read_config(checkpoint)
        self.num_experts, self.top_k, self.moe_layers, self.num_layers = moe_structure(config)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.renderer = ChatRenderer(checkpoint)
        self.layers = _find_layers(self.model)
        if len(self.layers) != self.num_layers:
            raise ValueError(f"config names {self.num_layers} layers, the model has {len(self.layers)}")
        self.final_norm = _find_final_norm(self.model)
        head = self.model.get_output_embeddings()
        assert head is not None
        self.lm_head = head
        self.router_edits: dict[int, RouterEdit] = {}
        self.layer_edits: dict[int, LayerEdit] = {}
        self.record_routing = False
        self.record_hidden = False
        self._routing: dict[int, list[RouterOutput]] = {}
        self._hidden: dict[int, torch.Tensor] = {}
        self._positions: torch.Tensor | None = None
        self._left_padding: torch.Tensor | None = None
        self._num_seen = 0
        self._layer_input: torch.Tensor | None = None
        self.routers: dict[int, torch.nn.Module] = {}
        self.semantics: dict[int, RouterSemantics] = {}
        for index, layer in enumerate(self.layers):
            router = _find_router(layer)
            if router is None:
                continue
            self.routers[index] = router
            declared_norm = getattr(router, "norm_topk_prob", config.get("norm_topk_prob", True))
            expert_scale = getattr(router, "per_expert_scale", None)
            self.semantics[index] = RouterSemantics(
                rule=router_scores,
                top_k=int(getattr(router, "top_k", getattr(router, "num_experts_per_tok", self.top_k))),
                norm_topk_prob=bool(declared_norm),
                scaling=float(
                    getattr(
                        router,
                        "routed_scaling_factor",
                        config.get("routed_scaling_factor", config.get("route_scale", 1.0)),
                    )
                ),
                expert_scale=expert_scale.detach() if expert_scale is not None else None,
            )
            router.register_forward_hook(self._router_hook(index), with_kwargs=True)
            layer.register_forward_pre_hook(self._layer_pre_hook(index), with_kwargs=True)
            layer.register_forward_hook(self._layer_hook(index), with_kwargs=True)
        if sorted(self.routers) != sorted(self.moe_layers):
            raise ValueError(f"routers found on layers {sorted(self.routers)}, config says {self.moe_layers}")
        for index, layer in enumerate(self.layers):
            if index not in self.routers:
                layer.register_forward_pre_hook(self._layer_pre_hook(index), with_kwargs=True)
                layer.register_forward_hook(self._layer_hook(index), with_kwargs=True)
        self.layers[0].register_forward_pre_hook(self._position_hook, with_kwargs=True, prepend=True)

    def _router_hook(self, index: int):
        def hook(module, args, kwargs, output):
            edit = self.router_edits.get(index)
            if edit is None and not self.record_routing:
                return output
            semantics = self.semantics[index]
            values, weights, selected = unpack_router_output(module, args, output)
            logits = semantics.logits(values)
            bias = _selection_bias(module, args, kwargs)
            if edit is not None:
                reselected = semantics.select(logits, bias, edit, self._positions)
                weights, selected = reselected.weights.to(weights.dtype), reselected.selected
                output = repack_router_output(output, weights, selected)
            else:
                weights, selected = semantics.by_score(logits, bias, weights, selected)
            if self.record_routing:
                self._routing.setdefault(index, []).append(
                    RouterOutput(router_logits=logits.detach(), weights=weights.detach(), selected=selected.detach())
                )
            return output

        return hook

    def _position_hook(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        batch, width = hidden.shape[:2]
        padding = self._left_padding if self._left_padding is not None else torch.zeros(batch, dtype=torch.long)
        offsets = torch.arange(self._num_seen, self._num_seen + width, device=hidden.device)
        self._positions = offsets.unsqueeze(0) - padding.to(hidden.device).unsqueeze(1)
        self._num_seen += width
        return None

    def _start_call(self, left_padding: list[int]) -> None:
        self._left_padding = torch.tensor(left_padding, dtype=torch.long)
        self._num_seen = 0

    def _layer_pre_hook(self, index: int):
        def hook(module, args, kwargs):
            if index in self.layer_edits or index in self.router_edits:
                self._layer_input = args[0] if args else kwargs["hidden_states"]
            return None

        return hook

    def _layer_hook(self, index: int):
        def hook(module, args, kwargs, output):
            edit = self.layer_edits.get(index)
            hidden_out = output[0] if isinstance(output, tuple) else output
            if edit is not None:
                assert self._positions is not None and self._layer_input is not None
                hidden_out = apply_layer_edit(edit, self._layer_input, hidden_out, self._positions)
                output = (hidden_out, *output[1:]) if isinstance(output, tuple) else hidden_out
            if self.record_hidden:
                self._hidden[index] = hidden_out.detach()
            return output

        return hook

    def check_router_semantics(self) -> None:
        ids = self.tokenizer("The quick brown fox jumps over the lazy dog, twice.", return_tensors="pt").input_ids
        recorded: dict[int, list[tuple]] = {}

        def make_hook(index):
            def hook(module, args, kwargs, output):
                semantics = self.semantics[index]
                values, weights, selected = unpack_router_output(module, args, output)
                logits = semantics.logits(values)
                bias = _selection_bias(module, args, kwargs)
                ours = semantics.select(logits, bias, None)
                weights, selected = semantics.by_score(logits, bias, weights, selected)
                recorded[index] = [(selected, weights, ours.selected, ours.weights)]
                return output

            return hook

        handles = [router.register_forward_hook(make_hook(i), with_kwargs=True) for i, router in self.routers.items()]
        self._start_call([0])
        try:
            with torch.no_grad():
                self.model(input_ids=ids.to(self.device))
        finally:
            for handle in handles:
                handle.remove()
        for index, rows in recorded.items():
            selected, weights, our_selected, our_weights = rows[0]
            if not torch.equal(selected, our_selected) or not torch.allclose(
                weights.float(), our_weights.float(), atol=1e-3, rtol=1e-2
            ):
                raise RuntimeError(
                    f"router semantics on layer {index} do not reproduce the module's selection; "
                    f"module {selected[:2].tolist()} {weights[:2].tolist()} vs {our_selected[:2].tolist()} "
                    f"{our_weights[:2].tolist()}"
                )

    @contextlib.contextmanager
    def edits(self, router: list[RouterEdit] = (), layer: list[LayerEdit] = ()) -> Iterator[None]:
        previous = (dict(self.router_edits), dict(self.layer_edits))
        self.router_edits.update({e.layer: e for e in router})
        self.layer_edits.update({e.layer: e for e in layer})
        try:
            yield
        finally:
            self.router_edits, self.layer_edits = previous

    def pad_right(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def forward(
        self, sequences: list[list[int]], *, hidden: bool = False, routing: bool = True, logits: bool = True
    ) -> Forward:
        ids, mask = self.pad_right(sequences)
        self.record_routing = routing
        self.record_hidden = hidden
        self._routing = {}
        self._hidden = {}
        self._start_call([0] * len(sequences))
        try:
            out = self.model(
                input_ids=ids, attention_mask=mask, use_cache=False, **({} if logits else {"logits_to_keep": 1})
            )
        finally:
            self.record_routing = False
            self.record_hidden = False
        routing_out: dict[int, RouterOutput] = {}
        for index, records in self._routing.items():
            record = records[-1]
            routing_out[index] = RouterOutput(
                router_logits=record.router_logits.view(ids.shape[0], ids.shape[1], -1),
                weights=record.weights.view(ids.shape[0], ids.shape[1], -1),
                selected=record.selected.view(ids.shape[0], ids.shape[1], -1),
            )
        self._routing = {}
        hidden_states = [self._hidden[i] for i in range(self.num_layers)] if hidden else []
        self._hidden = {}
        return Forward(
            logits=out.logits, hidden=hidden_states, routing=routing_out, lengths=[len(s) for s in sequences]
        )

    @property
    def recommended_sampling(self) -> dict:
        return recommended_sampling(self.checkpoint)

    @torch.no_grad()
    def lens_logits(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(residual))

    @torch.no_grad()
    def generate(
        self,
        sequences: list[list[int]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        bad_words_ids: list[list[int]] | None = None,
        sampling: dict | None = None,
    ) -> list[list[int]]:
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, width - len(seq) :] = torch.tensor(seq)
            mask[row, width - len(seq) :] = 1
        stops = {self.tokenizer.eos_token_id, *self.renderer.assistant_turn_end}
        if sampling is not None:
            torch.manual_seed(seed)
            sampling = {"do_sample": True, **sampling}
        elif temperature > 0:
            torch.manual_seed(seed)
            sampling = {"do_sample": True, "temperature": temperature}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        self._start_call([width - len(s) for s in sequences])
        out = self.model.generate(
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=pad,
            eos_token_id=sorted(t for t in stops if t is not None),
            bad_words_ids=bad_words_ids or None,
            **sampling,
        )
        results = []
        for row in range(len(sequences)):
            generated = out[row, width:].tolist()
            cut = len(generated)
            for i, token in enumerate(generated):
                if token in stops:
                    cut = i
                    break
            results.append(generated[:cut])
        return results


def recommended_sampling(checkpoint: str) -> dict:
    from transformers import GenerationConfig

    try:
        declared = GenerationConfig.from_pretrained(checkpoint)
    except OSError:
        declared = GenerationConfig()
    return {
        "temperature": declared.temperature if declared.temperature is not None else 1.0,
        "top_k": declared.top_k if declared.top_k is not None else 0,
        "top_p": declared.top_p if declared.top_p is not None else 1.0,
        "repetition_penalty": declared.repetition_penalty if declared.repetition_penalty is not None else 1.0,
    }


def _selection_bias(router: torch.nn.Module, args: tuple, kwargs: dict) -> torch.Tensor | None:
    bias = args[1] if len(args) > 1 else kwargs.get("expert_bias")
    return getattr(router, "e_score_correction_bias", None) if bias is None else bias
