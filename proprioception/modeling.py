import dataclasses
import itertools
import json
import re
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open


def load_safetensors(directory: str | Path) -> dict[str, torch.Tensor]:
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(directory / shard, framework="pt") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
    return state


def stack_experts(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = sorted({match.group(1) for key in state_dict if (match := re.match(r"(.*\.experts)\.\d+\.", key))})
    for prefix in prefixes:
        gates, ups, downs = [], [], []
        for expert in itertools.count():
            if f"{prefix}.{expert}.gate_proj.weight" not in state_dict:
                break
            gates.append(state_dict.pop(f"{prefix}.{expert}.gate_proj.weight"))
            ups.append(state_dict.pop(f"{prefix}.{expert}.up_proj.weight"))
            downs.append(state_dict.pop(f"{prefix}.{expert}.down_proj.weight"))
        if not gates:
            raise ValueError(f"{prefix} stores its experts under names other than gate_proj/up_proj/down_proj")
        state_dict[f"{prefix}.gate_up_proj"] = torch.cat([torch.stack(gates), torch.stack(ups)], dim=1)
        state_dict[f"{prefix}.down_proj"] = torch.stack(downs)
    return state_dict


def split_experts(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = sorted(key.removesuffix(".gate_up_proj") for key in state_dict if key.endswith(".experts.gate_up_proj"))
    for prefix in prefixes:
        gate_up = state_dict.pop(f"{prefix}.gate_up_proj")
        down = state_dict.pop(f"{prefix}.down_proj")
        half = gate_up.shape[1] // 2
        for expert in range(gate_up.shape[0]):
            state_dict[f"{prefix}.{expert}.gate_proj.weight"] = gate_up[expert, :half]
            state_dict[f"{prefix}.{expert}.up_proj.weight"] = gate_up[expert, half:]
            state_dict[f"{prefix}.{expert}.down_proj.weight"] = down[expert]
    return state_dict


@dataclasses.dataclass
class OlmoeConfig:
    hidden_size: int = 2048
    intermediate_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_experts: int = 64
    num_experts_per_tok: int = 8
    vocab_size: int = 50304
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "OlmoeConfig":
        raw = json.loads((Path(path) / "config.json").read_text())
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in fields and v is not None}
        return cls(**kwargs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rotary_tables(config: OlmoeConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(config.max_position_embeddings, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos[None, None, : q.shape[2], :].to(q.dtype)
    sin = sin[None, None, : q.shape[2], :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class OlmoeAttention(torch.nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        self.q_proj = torch.nn.Linear(h, h, bias=False)
        self.k_proj = torch.nn.Linear(h, h, bias=False)
        self.v_proj = torch.nn.Linear(h, h, bias=False)
        self.o_proj = torch.nn.Linear(h, h, bias=False)
        self.q_norm = RMSNorm(h, config.rms_norm_eps)
        self.k_norm = RMSNorm(h, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        q = self.q_norm(self.q_proj(x)).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(x)).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, hidden))


class OlmoeSparseMlp(torch.nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        e, h, i = config.num_experts, config.hidden_size, config.intermediate_size
        self.gate = torch.nn.Linear(h, e, bias=False)
        self.experts = torch.nn.ParameterDict(
            {
                "gate_up_proj": torch.nn.Parameter(torch.empty(e, 2 * i, h)),
                "down_proj": torch.nn.Parameter(torch.empty(e, h, i)),
            }
        )
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = i

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = x.shape
        flat = x.reshape(-1, hidden)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        weights = topk_probs.to(x.dtype)

        out = torch.zeros_like(flat)
        expert_mask = torch.nn.functional.one_hot(topk_ids, self.gate.out_features).permute(2, 1, 0)
        hit_experts = torch.nonzero(expert_mask.sum(dim=(1, 2)), as_tuple=False).flatten()
        for expert in hit_experts.tolist():
            slot, token = torch.where(expert_mask[expert])
            h_in = flat[token]
            gate_up = h_in @ self.experts["gate_up_proj"][expert].T
            gate_part, up_part = gate_up.split(self.intermediate_size, dim=-1)
            h_out = (torch.nn.functional.silu(gate_part) * up_part) @ self.experts["down_proj"][expert].T
            out.index_add_(0, token, h_out * weights[token, slot, None])
        return out.reshape(batch, seq, hidden), router_logits.reshape(batch, seq, -1)


class OlmoeLayer(torch.nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = OlmoeAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = OlmoeSparseMlp(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        mlp_out, router_logits = self.mlp(self.post_attention_layernorm(x))
        return x + mlp_out, router_logits


class OlmoeForCausalLM(torch.nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict(
            {
                "embed_tokens": torch.nn.Embedding(config.vocab_size, config.hidden_size),
                "layers": torch.nn.ModuleList(OlmoeLayer(config) for _ in range(config.num_hidden_layers)),
                "norm": RMSNorm(config.hidden_size, config.rms_norm_eps),
            }
        )
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        cos, sin = rotary_tables(config, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def layers(self) -> torch.nn.ModuleList:
        return cast(torch.nn.ModuleList, self.model["layers"])

    accepts_lens_request = True

    def routing_weights(self, layers: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores.float(), dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        lens_request: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.model["embed_tokens"](input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        all_router_logits = []
        lens_hidden = None
        if lens_request is not None:
            lens_layers, lens_rows, lens_cols = lens_request
            lens_hidden = x.new_zeros((lens_layers.numel(), x.shape[-1]))
        for index, layer in enumerate(self.layers):
            x, router_logits = layer(x, cos, sin)
            all_router_logits.append(router_logits)
            if lens_hidden is not None:
                hit = torch.eq(lens_layers, index).nonzero(as_tuple=True)[0]
                if hit.numel():
                    lens_hidden[hit] = x.detach()[lens_rows[hit], lens_cols[hit]]
        if lens_hidden is not None:
            with torch.no_grad():
                self.lens_logits = self.lm_head(self.model["norm"](lens_hidden))
        x = self.model["norm"](x)
        return self.lm_head(x), all_router_logits

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cuda", dtype=torch.bfloat16) -> "OlmoeForCausalLM":
        config = OlmoeConfig.from_checkpoint(path)
        model = cls(config)
        state = stack_experts(load_safetensors(path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith("rope_")]
        if missing or unexpected:
            raise ValueError(f"State dict mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
        model = model.to(device=device, dtype=dtype)
        cos, sin = rotary_tables(config, torch.device(device))
        model.rope_cos, model.rope_sin = cos, sin
        return model

    def to_hub_layout(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return split_experts(state_dict)
