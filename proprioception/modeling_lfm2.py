import dataclasses
import json
from pathlib import Path
from typing import cast

import torch
import torch.utils.checkpoint

from proprioception.modeling import load_safetensors


@dataclasses.dataclass
class Lfm2Config:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_dense_layers: int
    num_experts: int
    num_experts_per_tok: int
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    rope_theta: float
    vocab_size: int
    conv_l_cache: int
    conv_bias: bool
    norm_topk_prob: bool
    use_expert_bias: bool
    routed_scaling_factor: float
    layer_types: list[str]
    max_seq_len: int = 4096

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "Lfm2Config":
        raw = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            moe_intermediate_size=raw["moe_intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_dense_layers=raw["num_dense_layers"],
            num_experts=raw["num_experts"],
            num_experts_per_tok=raw["num_experts_per_tok"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            norm_eps=raw["norm_eps"],
            rope_theta=raw["rope_parameters"]["rope_theta"],
            vocab_size=raw["vocab_size"],
            conv_l_cache=raw["conv_L_cache"],
            conv_bias=raw["conv_bias"],
            norm_topk_prob=raw["norm_topk_prob"],
            use_expert_bias=raw["use_expert_bias"],
            routed_scaling_factor=raw["routed_scaling_factor"],
            layer_types=raw["layer_types"],
        )


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


def rotary_tables(config: Lfm2Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim)
    )
    positions = torch.arange(config.max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Lfm2Attention(torch.nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.config = config
        head_dim = config.head_dim
        self.q_proj = torch.nn.Linear(config.hidden_size, config.num_attention_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        self.out_proj = torch.nn.Linear(config.num_attention_heads * head_dim, config.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(head_dim, config.norm_eps)
        self.k_layernorm = RMSNorm(head_dim, config.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        head_dim = self.config.head_dim
        q = self.q_layernorm(self.q_proj(x).view(batch, seq, -1, head_dim)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).view(batch, seq, -1, head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, -1, head_dim).transpose(1, 2)
        cos_s, sin_s = cos[:seq].to(x.dtype), sin[:seq].to(x.dtype)
        q = q * cos_s + rotate_half(q) * sin_s
        k = k * cos_s + rotate_half(k) * sin_s
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).reshape(batch, seq, -1)
        return self.out_proj(y)


class Lfm2ShortConv(torch.nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_l_cache,
            groups=config.hidden_size,
            bias=config.conv_bias,
            padding=config.conv_l_cache - 1,
        )
        self.in_proj = torch.nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=config.conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        bcx = self.in_proj(x).transpose(-1, -2)
        b, c, x_inner = bcx.chunk(3, dim=-2)
        conv_out = self.conv(b * x_inner)[..., :seq]
        return self.out_proj((c * conv_out).transpose(-1, -2).contiguous())


class Lfm2ExpertMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Lfm2SparseMoe(torch.nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.config = config
        self.gate = torch.nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            Lfm2ExpertMlp(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)
        )
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = x.shape
        flat = x.view(-1, hidden)
        scores = self.gate(flat).sigmoid()
        selection_scores = scores.float() + self.expert_bias if self.config.use_expert_bias else scores.float()
        _, selected = torch.topk(selection_scores, self.config.num_experts_per_tok, dim=-1)
        weights = torch.gather(scores, 1, selected)
        if self.config.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights * self.config.routed_scaling_factor

        out = torch.zeros_like(flat)
        expert_mask = torch.nn.functional.one_hot(selected, self.config.num_experts).permute(2, 1, 0)
        for expert_idx in expert_mask.sum(dim=(-1, -2)).nonzero().flatten().tolist():
            slot, token_idx = torch.where(expert_mask[expert_idx])
            expert_out = self.experts[expert_idx](flat[token_idx]) * weights[token_idx, slot, None]
            out.index_add_(0, token_idx, expert_out.to(out.dtype))
        return out.view(batch, seq, hidden), selection_scores.view(batch, seq, -1)


class Lfm2DecoderLayer(torch.nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config)
        self.is_moe_layer = layer_idx >= config.num_dense_layers
        if self.is_moe_layer:
            self.feed_forward: torch.nn.Module = Lfm2SparseMoe(config)
        else:
            self.feed_forward = Lfm2ExpertMlp(config.hidden_size, config.intermediate_size)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.is_attention_layer:
            x = x + self.self_attn(self.operator_norm(x), cos, sin)
        else:
            x = x + self.conv(self.operator_norm(x))
        if self.is_moe_layer:
            ffn_out, routing_scores = self.feed_forward(self.ffn_norm(x))
        else:
            ffn_out, routing_scores = self.feed_forward(self.ffn_norm(x)), None
        return x + ffn_out, routing_scores


class Lfm2MoeForCausalLM(torch.nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict(
            {
                "embed_tokens": torch.nn.Embedding(config.vocab_size, config.hidden_size),
                "layers": torch.nn.ModuleList(
                    Lfm2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
                ),
                "embedding_norm": RMSNorm(config.hidden_size, config.norm_eps),
            }
        )
        cos, sin = rotary_tables(config, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = False

    def routing_weights(self, layers: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self.config.use_expert_bias:
            return scores.float()
        table = scores.new_zeros((len(self.layers), scores.shape[-1]), dtype=torch.float32)
        for index, layer in enumerate(self.layers):
            moe = layer.feed_forward
            if isinstance(moe, Lfm2SparseMoe):
                bias = moe.expert_bias
                assert isinstance(bias, torch.Tensor)
                table[index] = bias.to(table.device)
        return scores.float() - table[layers.to(table.device)]

    @property
    def layers(self) -> torch.nn.ModuleList:
        return cast(torch.nn.ModuleList, self.model["layers"])

    accepts_lens_request = True

    def forward(
        self,
        input_ids: torch.Tensor,
        lens_request: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        x = self.model["embed_tokens"](input_ids)
        all_routing_scores: list[torch.Tensor | None] = []
        lens_hidden = None
        if lens_request is not None:
            lens_layers, lens_rows, lens_cols = lens_request
            lens_hidden = x.new_zeros((lens_layers.numel(), x.shape[-1]))
        for index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                result = torch.utils.checkpoint.checkpoint(layer, x, self.rope_cos, self.rope_sin, use_reentrant=False)
                assert result is not None
                x, routing_scores = result
            else:
                x, routing_scores = layer(x, self.rope_cos, self.rope_sin)
            all_routing_scores.append(routing_scores)
            if lens_hidden is not None:
                hit = torch.eq(lens_layers, index).nonzero(as_tuple=True)[0]
                if hit.numel():
                    lens_hidden[hit] = x.detach()[lens_rows[hit], lens_cols[hit]]
        embedding = cast(torch.nn.Embedding, self.model["embed_tokens"])
        if lens_hidden is not None:
            with torch.no_grad():
                self.lens_logits = torch.nn.functional.linear(
                    self.model["embedding_norm"](lens_hidden), embedding.weight
                )
        x = self.model["embedding_norm"](x)
        logits = torch.nn.functional.linear(x, embedding.weight)
        return logits, all_routing_scores

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: str = "cpu") -> "Lfm2MoeForCausalLM":
        config = Lfm2Config.from_checkpoint(checkpoint_dir)
        model = cls(config).to(torch.bfloat16)
        state_dict = load_safetensors(checkpoint_dir)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        rope_buffers = {"rope_cos", "rope_sin"}
        real_missing = [k for k in missing if k not in rope_buffers]
        if real_missing or unexpected:
            raise ValueError(f"state dict mismatch: missing {real_missing[:5]}, unexpected {list(unexpected)[:5]}")
        model.rope_cos = model.rope_cos.float()
        model.rope_sin = model.rope_sin.float()
        return model.to(device)
