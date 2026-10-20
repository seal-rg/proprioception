import dataclasses
import hashlib

import torch

from proprioception.primerl_backend import ROUTER_SCORES

NOISE_BLOCK_POSITIONS = 64


def _block_seed(seed: int, block: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{block}".encode(), digest_size=4).digest(), "little")


def positional_noise(seed: int, positions: torch.Tensor, width: int) -> torch.Tensor:
    flat = positions.reshape(-1).long()
    out = torch.empty(flat.shape[0], width, device=positions.device)
    blocks = flat // NOISE_BLOCK_POSITIONS
    for block in torch.unique(blocks).tolist():
        generator = torch.Generator(device=positions.device).manual_seed(_block_seed(seed, block))
        table = torch.randn(NOISE_BLOCK_POSITIONS, width, generator=generator, device=positions.device)
        rows = blocks == block
        out[rows] = table[flat[rows] % NOISE_BLOCK_POSITIONS]
    return out.view(*positions.shape, width)


@dataclasses.dataclass
class RouterEdit:
    layer: int
    ablate: tuple[int, ...] = ()
    force_top: int | None = None
    noise: float = 0.0
    seed: int = 0


@dataclasses.dataclass
class LayerEdit:
    layer: int
    skip: bool = False
    noise: float = 0.0
    add: torch.Tensor | None = None
    strength: float = 1.0
    from_position: int = 0
    seed: int = 0


@dataclasses.dataclass
class RouterOutput:
    router_logits: torch.Tensor
    weights: torch.Tensor
    selected: torch.Tensor


@dataclasses.dataclass
class RouterSemantics:
    rule: str
    top_k: int
    norm_topk_prob: bool
    scaling: float
    expert_scale: torch.Tensor | None = None

    def __post_init__(self):
        if self.rule not in ROUTER_SCORES:
            raise ValueError(f"router scoring rule {self.rule!r} is not one of {ROUTER_SCORES}")

    def logits(self, router_values: torch.Tensor) -> torch.Tensor:
        if self.rule == "probabilities":
            return router_values.float().log()
        return router_values

    def probabilities(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.rule == "sigmoid":
            return router_logits.sigmoid()
        return torch.softmax(router_logits.float(), dim=-1)

    def by_score(
        self, router_logits: torch.Tensor, bias: torch.Tensor | None, weights: torch.Tensor, selected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.probabilities(router_logits).float()
        if bias is not None:
            key = key + bias.float()
        order = key.gather(-1, selected.long()).argsort(dim=-1, descending=True, stable=True)
        return weights.gather(-1, order), selected.gather(-1, order)

    def select(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor | None,
        edit: RouterEdit | None = None,
        positions: torch.Tensor | None = None,
    ) -> RouterOutput:
        logits = router_logits
        if edit is not None and edit.noise > 0:
            if positions is None or positions.numel() != logits.shape[0]:
                raise ValueError("router noise needs the absolute position of every router row")
            noise = positional_noise(edit.seed, positions.to(logits.device), logits.shape[-1])
            logits = logits + (edit.noise * noise.view_as(logits)).to(logits.dtype)
        probs = self.probabilities(logits)
        scores = probs if bias is None else probs + bias
        if edit is not None and edit.ablate:
            scores = scores.clone()
            scores[:, list(edit.ablate)] = float("-inf")
        if edit is not None and edit.force_top is not None:
            scores = scores.clone()
            scores[:, edit.force_top] = float("inf")
        _, selected = torch.topk(scores, k=self.top_k, dim=-1)
        weights = torch.gather(probs, dim=1, index=selected)
        if self.rule == "sigmoid":
            weights = weights.type_as(router_logits)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + (1e-6 if self.rule == "sigmoid" else 0.0))
        weights = weights * self.scaling
        if self.expert_scale is not None:
            weights = weights * self.expert_scale.to(weights.device)[selected]
        if self.rule == "softmax":
            weights = weights.to(router_logits.dtype)
        return RouterOutput(router_logits=router_logits, weights=weights, selected=selected)


def unpack_router_output(
    module: torch.nn.Module, args: tuple, output
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(output, tuple) and len(output) == 3:
        return output[0], output[1], output[2]
    if isinstance(output, tuple) and len(output) == 2:
        hidden = args[0]
        logits = torch.nn.functional.linear(hidden, module.weight)
        return logits, output[0], output[1]
    raise TypeError(f"unrecognised router output of type {type(output)} with {len(output)} entries")


def repack_router_output(output, weights: torch.Tensor, selected: torch.Tensor):
    if len(output) == 3:
        return (output[0], weights, selected)
    return (weights, selected)


def apply_layer_edit(
    edit: LayerEdit, hidden_in: torch.Tensor, hidden_out: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if edit.skip:
        hidden_out = hidden_in
    if edit.noise > 0:
        rms = hidden_out.float().pow(2).mean(-1, keepdim=True).sqrt()
        noise = positional_noise(edit.seed, positions.to(hidden_out.device), hidden_out.shape[-1])
        hidden_out = hidden_out + (edit.noise * rms * noise).to(hidden_out.dtype)
    if edit.add is not None and edit.strength != 0:
        addend = edit.strength * edit.add.to(hidden_out.device, torch.float32)
        addend = addend.view(1, 1, -1) if addend.dim() == 1 else addend.unsqueeze(1)
        if addend.shape[0] not in (1, hidden_out.shape[0]):
            raise ValueError(f"{addend.shape[0]} per-row addends for a batch of {hidden_out.shape[0]}")
        mask = (positions.to(hidden_out.device) >= edit.from_position).unsqueeze(-1)
        hidden_out = (hidden_out.float() + torch.where(mask, addend, 0.0)).to(hidden_out.dtype)
    return hidden_out
