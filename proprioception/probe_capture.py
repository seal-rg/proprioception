import collections
import json
import logging
import math
import uuid

import torch

logger = logging.getLogger(__name__)

CONFIG_KEY = "probe_capture"
EXTRA_ARGS_KEY = "proprio_probe"
MAX_PROBE_ROWS = 64
TOP_K = 5
MAX_STORED_RESULTS = 4096
FINAL_NORM_NAMES = ("norm", "embedding_norm", "final_layernorm", "norm_f", "ln_f")

_state: "CaptureState | None" = None
_results: collections.OrderedDict[str, dict] = collections.OrderedDict()


def enabled(additional_config) -> bool:
    if isinstance(additional_config, str):
        additional_config = json.loads(additional_config) if additional_config else {}
    return bool((additional_config or {}).get(CONFIG_KEY, False))


def _capture_rows(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    rows: torch.Tensor,
    out_hidden: torch.Tensor,
    out_residual: torch.Tensor,
    layer: int,
    has_residual: bool,
) -> None:
    rows = rows.clamp(max=hidden.shape[0] - 1)
    out_hidden[layer].copy_(hidden.index_select(0, rows))
    if has_residual:
        out_residual[layer].copy_(residual.index_select(0, rows))
    else:
        out_residual[layer].zero_()


def _capture_rows_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    rows: torch.Tensor,
    out_hidden: torch.Tensor,
    out_residual: torch.Tensor,
    layer: int,
    has_residual: bool,
) -> None:
    return None


_op_registered = False


def _register_op() -> None:
    global _op_registered
    if _op_registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="proprio_capture_rows",
        op_func=_capture_rows,
        mutates_args=["out_hidden", "out_residual"],
        fake_impl=_capture_rows_fake,
    )
    _op_registered = True


def _find_decoder_layers(model: torch.nn.Module, num_layers: int) -> tuple[torch.nn.Module, torch.nn.ModuleList]:
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) == num_layers:
            return module, layers
    raise ValueError(f"no module holds a ModuleList of {num_layers} decoder layers")


def _find_final_norm(inner: torch.nn.Module) -> torch.nn.Module:
    for name in FINAL_NORM_NAMES:
        norm = getattr(inner, name, None)
        if isinstance(norm, torch.nn.Module):
            return norm
    raise ValueError(f"the decoder stack names no final norm among {FINAL_NORM_NAMES}")


class CaptureState:
    def __init__(self, runner) -> None:
        hf_config = runner.model_config.hf_text_config
        self.model = runner.get_model()
        self.num_layers = int(hf_config.num_hidden_layers)
        self.top_k_experts = int(getattr(hf_config, "num_experts_per_tok", None) or hf_config.top_k_experts)
        inner, self.layers = _find_decoder_layers(self.model, self.num_layers)
        self.final_norm = _find_final_norm(inner)
        hidden_size = int(hf_config.hidden_size)
        device = runner.device
        dtype = runner.model_config.dtype
        self.rows = torch.zeros(MAX_PROBE_ROWS, dtype=torch.long, device=device)
        self.rows_dirty = False
        self.out_hidden = torch.zeros(self.num_layers, MAX_PROBE_ROWS, hidden_size, dtype=dtype, device=device)
        self.out_residual = torch.zeros_like(self.out_hidden)
        self.router_ids = torch.zeros(
            self.num_layers, MAX_PROBE_ROWS, self.top_k_experts, dtype=torch.long, device=device
        )
        self.router_weights = torch.zeros(
            self.num_layers, MAX_PROBE_ROWS, self.top_k_experts, dtype=torch.float32, device=device
        )
        self.router_layers: list[int] = []
        self.dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
        self.tp_size = runner.vllm_config.parallel_config.tensor_parallel_size
        self.pending: list[tuple[str, int, int | None, int]] = []
        self.failed: dict[str, str] = {}
        self._wrap_layers()
        self._bind_routers(runner)

    def _wrap_layers(self) -> None:
        _register_op()
        state = self
        for index, layer in enumerate(self.layers):
            original = layer.forward

            def forward(*args, _original=original, _index=index, **kwargs):
                output = _original(*args, **kwargs)
                if isinstance(output, tuple):
                    hidden, residual = output[0], output[1]
                else:
                    hidden, residual = output, None
                torch.ops.vllm.proprio_capture_rows(
                    hidden,
                    hidden if residual is None else residual,
                    state.rows,
                    state.out_hidden,
                    state.out_residual,
                    _index,
                    residual is not None,
                )
                return output

            layer.forward = forward

    def _bind_routers(self, runner) -> None:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        for module in runner.compilation_config.static_forward_context.values():
            if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter):
                if module.router.eplb_state is not None:
                    raise ValueError("probe capture reads logical expert ids; EPLB remaps them to physical ones")
                module.router._proprio_layer = int(module.layer_id)
                self.router_layers.append(int(module.layer_id))
        self.router_layers.sort()

    def local_rows(self, num_routed_tokens: int) -> torch.Tensor | None:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        if context.dp_metadata is None:
            return self.rows
        tokens_per_rank = context.dp_metadata.num_tokens_across_dp_cpu
        own = int(tokens_per_rank[self.dp_rank].item())
        if num_routed_tokens == int(tokens_per_rank.sum().item()):
            return self.rows + int(tokens_per_rank[: self.dp_rank].sum().item())
        if num_routed_tokens == own:
            return self.rows
        return None

    def capture_router(self, layer: int, weights: torch.Tensor, ids: torch.Tensor) -> None:
        rows = self.local_rows(ids.shape[0])
        if rows is None:
            if self.tp_size > 1:
                from vllm.distributed.parallel_state import get_tp_group

                ids = get_tp_group().all_gather(ids, dim=0)
                weights = get_tp_group().all_gather(weights, dim=0)
                rows = self.rows
            else:
                raise AssertionError(f"probe capture: unexpected router batch of {ids.shape[0]} tokens")
        rows = rows.clamp(max=ids.shape[0] - 1)
        self.router_ids[layer].copy_(ids.index_select(0, rows))
        self.router_weights[layer].copy_(weights.index_select(0, rows).float())

    def schedule(self, runner, num_scheduled_tokens) -> None:
        rows: list[int] = []
        self.pending = []
        for index, req_id in enumerate(runner.input_batch.req_ids):
            request = runner.requests.get(req_id)
            params = None if request is None else request.sampling_params
            spec = None if params is None else (params.extra_args or {}).get(EXTRA_ARGS_KEY)
            if not spec:
                continue
            start = int(runner.input_batch.num_computed_tokens_cpu[index])
            count = int(num_scheduled_tokens[index])
            offset = int(runner.query_start_loc.np[index])
            prompt = request.prompt_token_ids or []
            for position in spec["positions"]:
                if not start <= position < start + count:
                    continue
                if len(rows) == MAX_PROBE_ROWS:
                    self.failed[spec["id"]] = f"more than {MAX_PROBE_ROWS} probe positions in one step"
                    continue
                rows.append(offset + position - start)
                following = prompt[position + 1] if position + 1 < len(prompt) else None
                self.pending.append((spec["id"], position, following, len(spec["positions"])))
        if rows or self.rows_dirty:
            padded = rows + [0] * (MAX_PROBE_ROWS - len(rows))
            self.rows.copy_(torch.tensor(padded, dtype=torch.long))
            self.rows_dirty = bool(rows)

    @torch.inference_mode()
    def collect(self, final_hidden: torch.Tensor) -> None:
        num = len(self.pending)
        hidden_size = self.out_hidden.shape[-1]
        hidden = self.out_hidden[:, :num].reshape(-1, hidden_size).clone()
        residual = self.out_residual[:, :num].reshape(-1, hidden_size).clone()
        normed = self.final_norm(hidden, residual)
        if isinstance(normed, tuple):
            normed = normed[0]
        normed[(self.num_layers - 1) * num :] = final_hidden.index_select(0, self.rows[:num])
        logits = self.model.compute_logits(normed)
        pending, self.pending = self.pending, []
        if logits is None:
            return
        log_probs = torch.log_softmax(logits.float(), dim=-1).view(self.num_layers, num, -1)
        top = log_probs.topk(TOP_K, dim=-1)
        entropy_bits = -(log_probs.exp() * log_probs).sum(-1) / math.log(2)
        following = torch.tensor([f if f is not None else 0 for _, _, f, _ in pending], device=log_probs.device)
        next_logprob = log_probs.gather(2, following.view(1, num, 1).expand(self.num_layers, num, 1)).squeeze(-1)
        top_ids, top_logprobs = top.indices.tolist(), top.values.tolist()
        entropy, next_lp = entropy_bits.tolist(), next_logprob.tolist()
        router_ids = self.router_ids[:, :num].tolist()
        router_weights = self.router_weights[:, :num].tolist()
        for slot, (probe_id, position, token_following, num_positions) in enumerate(pending):
            readout = {
                "layers": [
                    {
                        "top_ids": top_ids[layer][slot],
                        "top_logprobs": top_logprobs[layer][slot],
                        "argmax": top_ids[layer][slot][0],
                        "entropy_bits": entropy[layer][slot],
                        "next_logprob": None if token_following is None else next_lp[layer][slot],
                    }
                    for layer in range(self.num_layers)
                ],
                "router": {
                    str(layer): {"ids": router_ids[layer][slot], "weights": router_weights[layer][slot]}
                    for layer in self.router_layers
                },
            }
            entry = _results.setdefault(probe_id, {"num_positions": num_positions, "readouts": {}})
            entry["readouts"][str(position)] = readout
            _results.move_to_end(probe_id)
        while len(_results) > MAX_STORED_RESULTS:
            _results.popitem(last=False)


def fetch(worker, probe_id: str) -> dict | None:
    if _state is None:
        return {"error": "probe capture is not installed in this worker"}
    failure = _state.failed.pop(probe_id, None)
    if failure is not None:
        return {"error": failure}
    return _results.pop(probe_id, None)


def _install_worker_patches() -> None:
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(GPUModelRunner, "_proprio_patched", False):
        return
    load_model = GPUModelRunner.load_model
    prepare_inputs = GPUModelRunner._prepare_inputs
    model_forward = GPUModelRunner._model_forward
    select_experts = BaseRouter._select_experts

    def patched_load_model(self, *args, **kwargs):
        global _state
        result = load_model(self, *args, **kwargs)
        if not enabled(self.vllm_config.additional_config):
            GPUModelRunner.load_model = load_model
            GPUModelRunner._prepare_inputs = prepare_inputs
            GPUModelRunner._model_forward = model_forward
            BaseRouter._select_experts = select_experts
            del Worker.proprio_fetch_probe
            return result
        _state = CaptureState(self)
        logger.info(
            "proprioception probe capture installed: %d decoder layers, %d MoE layers",
            _state.num_layers,
            len(_state.router_layers),
        )
        return result

    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens_np, *args, **kwargs):
        result = prepare_inputs(self, scheduler_output, num_scheduled_tokens_np, *args, **kwargs)
        if _state is not None:
            _state.schedule(self, num_scheduled_tokens_np)
        return result

    def patched_model_forward(self, *args, **kwargs):
        output = model_forward(self, *args, **kwargs)
        if _state is not None and _state.pending:
            if not isinstance(output, torch.Tensor):
                raise ValueError("probe capture reads the forward's hidden states; this model returns more than that")
            _state.collect(output)
        return output

    def patched_select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, **kwargs):
        weights, ids = select_experts(self, hidden_states, router_logits, topk_indices_dtype, **kwargs)
        layer = getattr(self, "_proprio_layer", None)
        if layer is not None and _state is not None:
            _state.capture_router(layer, weights, ids)
        return weights, ids

    GPUModelRunner.load_model = patched_load_model
    GPUModelRunner._prepare_inputs = patched_prepare_inputs
    GPUModelRunner._model_forward = patched_model_forward
    BaseRouter._select_experts = patched_select_experts
    Worker.proprio_fetch_probe = fetch
    GPUModelRunner._proprio_patched = True


def _install_api_route() -> None:
    import vllm.entrypoints.openai.api_server as api_server

    build_app = api_server.build_app
    if getattr(build_app, "_proprio_route", False):
        return

    def patched_build_app(args, *rest, **kwargs):
        from fastapi import Request
        from fastapi.responses import JSONResponse
        from vllm.sampling_params import SamplingParams

        app = build_app(args, *rest, **kwargs)
        if not enabled(getattr(args, "additional_config", None)):
            return app

        @app.post("/proprio/probe")
        async def probe(raw_request: Request):
            body = await raw_request.json()
            token_ids = [int(t) for t in body["token_ids"]]
            positions = sorted({int(p) for p in body["positions"]})
            if not positions or positions[0] < 0 or positions[-1] >= len(token_ids):
                return JSONResponse({"error": "positions must lie inside token_ids"}, status_code=400)
            probe_id = uuid.uuid4().hex
            params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                skip_reading_prefix_cache=True,
                extra_args={EXTRA_ARGS_KEY: {"id": probe_id, "positions": positions}},
            )
            engine = raw_request.app.state.engine_client
            async for _ in engine.generate({"prompt_token_ids": token_ids}, params, probe_id, data_parallel_rank=0):
                pass
            replies = await engine.collective_rpc("proprio_fetch_probe", args=(probe_id,))
            errors = [r["error"] for r in replies if r and "error" in r]
            found = next((r for r in replies if r and "readouts" in r), None)
            if found is None or len(found["readouts"]) != len(positions):
                detail = errors[0] if errors else "the workers returned no readout for every position"
                return JSONResponse({"error": detail}, status_code=500)
            return {"readouts": [found["readouts"][str(p)] for p in positions], "positions": positions}

        return app

    patched_build_app._proprio_route = True
    api_server.build_app = patched_build_app


def install() -> None:
    _install_worker_patches()
    _install_api_route()
