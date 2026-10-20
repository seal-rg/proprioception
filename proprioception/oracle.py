import base64
import dataclasses
import functools
import io

import httpx
import numpy as np

from proprioception.rendering import ChatRenderer, Rendered


@dataclasses.dataclass(frozen=True)
class PolicySample:
    prompt_token_ids: list[int]
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    text: str


@dataclasses.dataclass
class ConversationRouting:
    rendered: Rendered
    token_texts: list[str]
    routed_experts: np.ndarray

    @property
    def token_ids(self) -> list[int]:
        return self.rendered.token_ids

    @property
    def num_decoder_layers(self) -> int:
        return self.routed_experts.shape[1]

    @property
    def top_k(self) -> int:
        return self.routed_experts.shape[2]

    def experts_at(self, token_index: int, layer: int) -> list[int]:
        if token_index >= self.routed_experts.shape[0]:
            raise ValueError(
                f"Token index {token_index} has no routing row (only {self.routed_experts.shape[0]} forwarded tokens)"
            )
        return [int(e) for e in self.routed_experts[token_index, layer]]


@dataclasses.dataclass(frozen=True)
class LayerReadout:
    top_ids: list[int]
    top_logprobs: list[float]
    argmax: int
    entropy_bits: float
    next_logprob: float | None


@dataclasses.dataclass(frozen=True)
class PositionReadout:
    layers: list[LayerReadout]
    router: dict[int, tuple[list[int], list[float]]]


@dataclasses.dataclass
class ProbeCapture:
    rendered: Rendered
    token_texts: list[str]
    readouts: dict[int, PositionReadout]

    @property
    def token_ids(self) -> list[int]:
        return self.rendered.token_ids


def parse_readout(payload: dict) -> PositionReadout:
    return PositionReadout(
        layers=[
            LayerReadout(
                top_ids=[int(t) for t in layer["top_ids"]],
                top_logprobs=[float(v) for v in layer["top_logprobs"]],
                argmax=int(layer["argmax"]),
                entropy_bits=float(layer["entropy_bits"]),
                next_logprob=None if layer["next_logprob"] is None else float(layer["next_logprob"]),
            )
            for layer in payload["layers"]
        ],
        router={
            int(layer): ([int(e) for e in entry["ids"]], [float(w) for w in entry["weights"]])
            for layer, entry in payload["router"].items()
        },
    )


class WordNotFound(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ReasoningMarkers:
    open: str = "<think>"
    close: str = "</think>"


DEFAULT_REASONING = ReasoningMarkers()


def visible_assistant_content(reply: str, markers: ReasoningMarkers = DEFAULT_REASONING) -> str:
    tail = reply.rsplit(markers.close, maxsplit=1)[-1]
    if markers.open in tail:
        return ""
    return tail.strip()


def ends_inside_reasoning(text: str, markers: ReasoningMarkers = DEFAULT_REASONING) -> bool:
    return text.rfind(markers.open) > text.rfind(markers.close)


def reopens_reasoning(reply: str, markers: ReasoningMarkers = DEFAULT_REASONING) -> bool:
    return markers.close in reply and markers.open in reply.split(markers.close, maxsplit=1)[1]


def token_occurrence(token_texts: list[str], index: int, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    same = [i for i in range(start, end) if token_texts[i] == token_texts[index]]
    return same.index(index) + 1, len(same)


def find_word_token_index(token_texts: list[str], word: str, *, occurrence: int = 0) -> int:
    target = " " + word
    seen = 0
    for start in range(len(token_texts)):
        if not target.startswith(token_texts[start]):
            continue
        acc = token_texts[start]
        end = start
        while len(acc) < len(target) and end + 1 < len(token_texts):
            end += 1
            acc += token_texts[end]
        if acc == target:
            boundary_ok = end + 1 >= len(token_texts) or not token_texts[end + 1][:1].isalnum()
            if boundary_ok:
                if seen == occurrence:
                    return end
                seen += 1
    raise WordNotFound(f"Word {word!r} (occurrence {occurrence}) not found in token texts")


def decode_routed_experts(payload: str | dict) -> np.ndarray:
    if isinstance(payload, str):
        return np.load(io.BytesIO(base64.b64decode(payload)))
    if payload.get("start", 0) != 0:
        raise ValueError(f"Unexpected routed_experts start offset {payload['start']}")
    data = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    return data.reshape(payload["shape"])


class RoutingOracle:
    def __init__(
        self, base_url: str, model: str, *, renderer: ChatRenderer | None = None, timeout_seconds: float = 120.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.renderer = renderer or ChatRenderer(model)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    @functools.cached_property
    def text_config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(self.model).get_text_config()

    @property
    def num_hidden_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def num_experts_per_tok(self) -> int:
        config = self.text_config
        return int(getattr(config, "num_experts_per_tok", None) or config.top_k_experts)

    async def sample(self, messages: list[dict], *, max_tokens: int, temperature: float) -> PolicySample:
        rendered = self.renderer.render_as_template(messages, add_generation_prompt=True)
        response = await self._client.post(
            "/inference/v1/generate",
            json={
                "model": self.model,
                "token_ids": rendered.token_ids,
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop_token_ids": self.renderer.assistant_turn_end,
                    "skip_special_tokens": False,
                    "logprobs": 1,
                },
            },
            timeout=httpx.Timeout(600.0),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        token_ids = list(choice.get("token_ids") or [])
        content = (choice.get("logprobs") or {}).get("content") or []
        if len(content) != len(token_ids) or any(entry.get("logprob") is None for entry in content):
            raise RuntimeError(f"the server returned {len(content)} logprobs for {len(token_ids)} sampled tokens")
        logprobs = [float(entry["logprob"]) for entry in content]
        return PolicySample(
            prompt_token_ids=list(rendered.token_ids),
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=choice.get("finish_reason") or "stop",
            text=self.renderer.decode(self.renderer.strip_turn_end(token_ids)),
        )

    async def capture(self, messages: list[dict]) -> ConversationRouting:
        return await self.capture_rendered(self.renderer.render_as_template(messages))

    async def capture_rendered(self, rendered: Rendered) -> ConversationRouting:
        response = await self._client.post(
            "/inference/v1/generate",
            json={
                "model": self.model,
                "token_ids": rendered.token_ids,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "skip_reading_prefix_cache": True,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        if choice.get("routed_experts") is None:
            raise RuntimeError("Server returned no routed_experts; launch vLLM with --enable-return-routed-experts")
        routed = decode_routed_experts(choice["routed_experts"])
        if routed.ndim != 3:
            raise RuntimeError(f"Unexpected routed_experts shape {routed.shape}, expected 3 dims")
        if routed.shape[0] < len(rendered.token_ids):
            raise RuntimeError(
                f"Routing rows ({routed.shape[0]}) cover fewer than all prompt tokens "
                f"({len(rendered.token_ids)}); prefix caching may not have been bypassed"
            )
        return ConversationRouting(
            rendered=rendered,
            token_texts=self.renderer.texts(rendered.token_ids),
            routed_experts=routed[: len(rendered.token_ids)],
        )

    async def probe(self, messages: list[dict], positions: list[int]) -> ProbeCapture:
        return await self.probe_rendered(self.renderer.render_as_template(messages), positions)

    async def probe_rendered(self, rendered: Rendered, positions: list[int]) -> ProbeCapture:
        response = await self._client.post(
            "/proprio/probe", json={"token_ids": rendered.token_ids, "positions": positions}
        )
        if response.status_code == 404:
            raise RuntimeError(
                "Server has no /proprio/probe route; enable the probe capture ([inference] enable_probe_capture)"
            )
        response.raise_for_status()
        payload = response.json()
        return ProbeCapture(
            rendered=rendered,
            token_texts=self.renderer.texts(rendered.token_ids),
            readouts={int(p): parse_readout(r) for p, r in zip(payload["positions"], payload["readouts"], strict=True)},
        )
