import asyncio
import dataclasses
import ssl
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
import numpy as np

from proprioception.oracle import (
    DEFAULT_REASONING,
    ReasoningMarkers,
    decode_routed_experts,
    ends_inside_reasoning,
    visible_assistant_content,
)


PROMPT_TOO_LONG = "prompt_too_long"
NON_ANSWERS = (PROMPT_TOO_LONG, "blocked", "truncated", "unclosed_reasoning", "empty", "unparsed")


@dataclasses.dataclass
class Reply:
    text: str
    reasoning: str
    finish_reason: str
    logprobs: list[tuple[str, float, dict[str, float]]]
    open_reasoning: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def blocked(self) -> bool:
        return self.finish_reason == "content_filter"


def non_answer(reply: Reply, parsed: bool | None = None) -> str | None:
    if reply.finish_reason == PROMPT_TOO_LONG:
        return PROMPT_TOO_LONG
    if reply.blocked:
        return "blocked"
    if reply.truncated and not reply.text.strip():
        return "truncated"
    if reply.open_reasoning and not reply.text.strip():
        return "unclosed_reasoning"
    if not reply.text.strip():
        return "empty"
    if parsed is False:
        return "unparsed"
    return None


def non_answer_summary(categories: Sequence[str | None]) -> dict:
    n = len(categories)
    out: dict = {"n": n}
    for name in NON_ANSWERS:
        out[name] = sum(c == name for c in categories) / n if n else None
    return out


def moderation_block(payload: dict) -> bool:
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    return error.get("code") in (403, "403") or "flagged" in str(error.get("message", "")).lower()


@dataclasses.dataclass
class Completion:
    text: str
    finish_reason: str
    top_logprobs: list[dict[str, float]]
    logprobs: list[tuple[str, float, dict[str, float]]] = dataclasses.field(default_factory=list)
    token_ids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Generation:
    token_ids: list[int]
    finish_reason: str
    routed_experts: np.ndarray | None
    logprobs: list[float]
    row_start: int = 0


def run_sync(coro):
    return asyncio.run(coro)


def _not_sent(result):
    async def call(_client):
        return result

    return call


class ServerStalled(RuntimeError):
    pass


STALL_CHECK_TIMEOUT = 120


def split_reply(text: str, markers: ReasoningMarkers = DEFAULT_REASONING) -> tuple[str, str]:
    visible = visible_assistant_content(text, markers)
    if markers.close in text:
        reasoning = text.rsplit(markers.close, maxsplit=1)[0]
    elif markers.open in text:
        reasoning = text.split(markers.open, maxsplit=1)[1]
    else:
        return visible, ""
    return visible, reasoning.split(markers.open, maxsplit=1)[-1].strip()


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        concurrency: int = 32,
        timeout: float = 1800,
        api_key: str = "",
        renderer=None,
        thinking: str | None = None,
        banned_ids: Sequence[Sequence[int]] = (),
        template_kwargs: dict | None = None,
        markers: ReasoningMarkers = DEFAULT_REASONING,
        max_model_len: int = 0,
    ):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.trust_env = not self.base_url.startswith(("http://127.0.0.1", "http://localhost"))
        self.renderer = renderer
        self.thinking = thinking
        self.logit_bias = {str(seq[0]): -100.0 for seq in banned_ids if seq}
        self.template_kwargs = dict(template_kwargs or {})
        self.markers = markers
        self.num_blocked = 0
        self.max_model_len = max_model_len
        self.num_budget_capped = 0
        self.num_prompt_too_long = 0
        self.stalled: str | None = None
        if thinking == "nothink" and renderer is None:
            raise ValueError("no-think chat needs the checkpoint's renderer")

    def _budget(self, prompt: Sequence[int], max_tokens: int) -> int:
        if not self.max_model_len:
            return max_tokens
        room = self.max_model_len - len(prompt)
        if room < 1:
            self.num_prompt_too_long += 1
            return 0
        if room < max_tokens:
            self.num_budget_capped += 1
            return room
        return max_tokens

    async def _responsive(self, client: httpx.AsyncClient) -> bool:
        try:
            health = await client.get("/health", timeout=STALL_CHECK_TIMEOUT)
            health.raise_for_status()
            response = await client.post(
                "/v1/completions",
                json={"model": self.model, "prompt": "Hello", "max_tokens": 1, "temperature": 0.0},
                timeout=STALL_CHECK_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def _gather(self, calls: Sequence[Callable[[httpx.AsyncClient], Awaitable[Any]]]) -> list:
        if self.stalled is not None:
            raise ServerStalled(self.stalled)
        semaphore = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency + 8, max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, limits=limits, headers=self.headers, trust_env=self.trust_env
        ) as client:

            async def one(call):
                async with semaphore:
                    if self.stalled is not None:
                        raise ServerStalled(self.stalled)
                    try:
                        return await call(client)
                    except httpx.TimeoutException as error:
                        if self.trust_env or self.stalled is not None or await self._responsive(client):
                            raise
                        self.stalled = f"{self.base_url} timed out after {self.timeout}s and failed a liveness check"
                        raise ServerStalled(self.stalled) from error

            return list(await asyncio.gather(*(one(c) for c in calls)))

    async def _chat(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        top_logprobs: int,
        extra: dict | None,
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        if extra:
            body.update(extra)
        payload: dict = {}
        for attempt in range(5):
            try:
                response = await client.post("/v1/chat/completions", json=body)
                if response.is_client_error:
                    raise RuntimeError(f"{response.status_code} from {self.model}: {response.text[:500]}")
                response.raise_for_status()
                payload = response.json()
                if "choices" not in payload and moderation_block(payload):
                    self.num_blocked += 1
                    return Reply(text="", reasoning="", finish_reason="content_filter", logprobs=[])
                if "choices" not in payload:
                    raise httpx.HTTPError(f"reply without choices from {self.model}: {response.text[:300]}")
                break
            except (httpx.HTTPError, ssl.SSLError, OSError):
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)
        choice = payload["choices"][0]
        message = choice["message"]
        logprobs = []
        for entry in (choice.get("logprobs") or {}).get("content") or []:
            top = {alt["token"]: alt["logprob"] for alt in entry.get("top_logprobs") or []}
            logprobs.append((entry["token"], entry["logprob"], top))
        return Reply(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
            finish_reason=choice.get("finish_reason") or "",
            logprobs=logprobs,
        )

    def chat(
        self,
        conversations: Sequence[list[dict]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
        top_logprobs: int = 0,
        extra: dict | None = None,
    ) -> list[Reply]:
        if self.renderer is not None:
            return self._chat_local(conversations, temperature, max_tokens, seed, top_logprobs)
        calls = [
            (
                lambda c, m=m, i=i: self._chat(
                    c,
                    m,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed + i,
                    top_logprobs=top_logprobs,
                    extra=extra,
                )
            )
            for i, m in enumerate(conversations)
        ]
        return run_sync(self._gather(calls))

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        assert self.renderer is not None
        if messages and messages[-1]["role"] == "assistant":
            raise ValueError("chat requests end with a user turn; assistant prefills are not supported")
        return self.renderer.render(messages, add_generation_prompt=True).token_ids

    def _reply_text(self, completion: Completion) -> str:
        assert self.renderer is not None
        ids = list(completion.token_ids)
        ends = {*self.renderer.assistant_turn_end, self.renderer.tokenizer.eos_token_id}
        while ids and ids[-1] in ends:
            ids.pop()
        return self.renderer.decode(ids)

    def _chat_local(self, conversations, temperature, max_tokens, seed, top_logprobs) -> list[Reply]:
        prompts = [self._prompt_ids(m) for m in conversations]
        calls = [
            (
                lambda c, p=p, i=i, budget=budget: self._complete(
                    c,
                    p,
                    temperature=temperature,
                    max_tokens=budget,
                    seed=seed + i,
                    top_logprobs=top_logprobs,
                    logit_bias=self.logit_bias or None,
                    return_token_ids=True,
                )
            )
            if budget
            else _not_sent(Completion(text="", finish_reason=PROMPT_TOO_LONG, top_logprobs=[]))
            for i, (p, budget) in enumerate(zip(prompts, [self._budget(p, max_tokens) for p in prompts], strict=True))
        ]
        completions = run_sync(self._gather(calls))
        assert self.renderer is not None
        replies = []
        for prompt, completion in zip(prompts, completions, strict=True):
            opened = self.markers.open if ends_inside_reasoning(self.renderer.decode(prompt), self.markers) else ""
            raw = opened + self._reply_text(completion)
            text, reasoning = split_reply(raw, self.markers)
            replies.append(
                Reply(
                    text=text,
                    reasoning=reasoning,
                    finish_reason=completion.finish_reason,
                    logprobs=completion.logprobs,
                    open_reasoning=bool(self.markers.open) and ends_inside_reasoning(raw, self.markers),
                )
            )
        return replies

    async def _complete(
        self,
        client: httpx.AsyncClient,
        prompt: str | list[int],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        top_logprobs: int,
        logit_bias: dict[str, float] | None = None,
        return_token_ids: bool = False,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if top_logprobs:
            body["logprobs"] = top_logprobs
        if logit_bias:
            body["logit_bias"] = logit_bias
        if return_token_ids:
            body["return_token_ids"] = True
        response = await client.post("/v1/completions", json=body)
        if response.is_client_error:
            raise RuntimeError(f"{response.status_code} from {self.model}: {response.text[:500]}")
        response.raise_for_status()
        choice = response.json()["choices"][0]
        logprobs = choice.get("logprobs") or {}
        tops = logprobs.get("top_logprobs") or []
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        return Completion(
            text=choice.get("text") or "",
            finish_reason=choice.get("finish_reason") or "",
            top_logprobs=[dict(entry) for entry in tops],
            logprobs=[
                (token, float(lp), dict(top or {}))
                for token, lp, top in zip(tokens, token_logprobs, tops or [{}] * len(tokens), strict=False)
                if lp is not None
            ],
            token_ids=list(choice.get("token_ids") or []),
        )

    def complete(
        self,
        prompts: Sequence[str],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        seed: int = 0,
        top_logprobs: int = 0,
    ) -> list[Completion]:
        calls = [
            (
                lambda c, p=p, i=i: self._complete(
                    c, p, temperature=temperature, max_tokens=max_tokens, seed=seed + i, top_logprobs=top_logprobs
                )
            )
            for i, p in enumerate(prompts)
        ]
        return run_sync(self._gather(calls))

    async def _generate(
        self,
        client: httpx.AsyncClient,
        token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        routing: bool,
        logprobs: bool,
        reuse_cache: bool,
        sampling: dict | None = None,
    ) -> Generation:
        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            **(sampling or {}),
        }
        if routing and not reuse_cache:
            params["skip_reading_prefix_cache"] = True
        if logprobs:
            params["logprobs"] = 0
        response = await client.post(
            "/inference/v1/generate", json={"model": self.model, "token_ids": token_ids, "sampling_params": params}
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        routed = None
        row_start = 0
        generated = list(choice.get("token_ids") or [])
        if routing:
            if choice.get("routed_experts") is None:
                raise RuntimeError("server returned no routed_experts; launch vLLM with --enable-return-routed-experts")
            routed = decode_routed_experts(choice["routed_experts"])
            forwarded = len(token_ids) + max(len(generated) - 1, 0)
            if routed.shape[0] < forwarded and not reuse_cache:
                raise RuntimeError(f"routing rows {routed.shape[0]} cover fewer than {forwarded} forwarded tokens")
            row_start = max(forwarded - routed.shape[0], 0)
        lps = []
        for entry in choice.get("logprobs") or []:
            if isinstance(entry, dict):
                lps.append(float(next(iter(entry.values()))["logprob"]) if entry else float("nan"))
        return Generation(
            token_ids=generated,
            finish_reason=choice.get("finish_reason") or "",
            routed_experts=routed,
            logprobs=lps,
            row_start=row_start,
        )

    def generate(
        self,
        prompts: Sequence[list[int]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        routing: bool = False,
        logprobs: bool = False,
        reuse_cache: bool = False,
        sampling: dict | None = None,
    ) -> list[Generation]:
        calls = [
            (
                lambda c, p=p, i=i, budget=budget: self._generate(
                    c,
                    p,
                    max_tokens=budget,
                    temperature=temperature,
                    seed=seed + i,
                    routing=routing,
                    logprobs=logprobs,
                    reuse_cache=reuse_cache,
                    sampling=sampling,
                )
            )
            if budget
            else _not_sent(Generation(token_ids=[], finish_reason=PROMPT_TOO_LONG, routed_experts=None, logprobs=[]))
            for i, (p, budget) in enumerate(zip(prompts, [self._budget(p, max_tokens) for p in prompts], strict=True))
        ]
        return run_sync(self._gather(calls))

    def routing(self, prompts: Sequence[list[int]]) -> list[np.ndarray]:
        captures = []
        for g, p in zip(self.generate(prompts, max_tokens=1, routing=True), prompts, strict=True):
            if g.routed_experts is None:
                raise ValueError(
                    f"a prompt of {len(p)} tokens leaves no room in the {self.max_model_len}-token context"
                )
            captures.append(g.routed_experts[: len(p)])
        return captures

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        if self.renderer is not None:
            return self._prompt_ids(messages)
        response = httpx.post(
            f"{self.base_url}/tokenize",
            json={"model": self.model, "messages": messages, "add_generation_prompt": True},
            timeout=60,
            trust_env=self.trust_env,
        )
        response.raise_for_status()
        return response.json()["tokens"]
