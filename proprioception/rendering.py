import dataclasses
import re

import httpx

SYSTEM, USER, ASSISTANT = "system", "user", "assistant"
FINAL_ASSISTANT, GENERATION, END, START = "assistant_final", "generation", "end", "start"

_SENTINELS = {SYSTEM: "quorvex", USER: "veloxq", ASSISTANT: "brimtal"}
_CHECK_SENTINELS = {SYSTEM: "zalmor", USER: "trivask", ASSISTANT: "wendrup"}


@dataclasses.dataclass
class Rendered:
    token_ids: list[int]
    message_indices: list[int]
    is_content: list[bool]
    message_roles: list[str]

    def content_span(self, message_index: int) -> tuple[int, int]:
        if message_index < 0:
            message_index += len(self.message_roles)
        positions = [i for i, k in enumerate(self.message_indices) if k == message_index]
        if positions:
            return positions[0], positions[-1] + 1
        previous = [i for i, k in enumerate(self.message_indices) if 0 <= k < message_index]
        start = previous[-1] + 1 if previous else 0
        while start < len(self.token_ids) and self.message_indices[start] == -1:
            start += 1
        return start, start

    def content_spans(self) -> list[tuple[int, int, str]]:
        return [(*self.content_span(k), role) for k, role in enumerate(self.message_roles)]

    def until_content_end(self, message_index: int = -1) -> list[int]:
        return self.token_ids[: self.content_span(message_index)[1]]


def tokens_within(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [i for i, (s, e) in enumerate(offsets) if s >= start and e <= end and e > s]


def segment_of_token(offsets: list[tuple[int, int]], boundaries: list[int]) -> list[int]:
    return [sum(s >= b for b in boundaries) for s, _ in offsets]


def _find(ids: list[int], needle: list[int], after: int = 0) -> int:
    for i in range(after, len(ids) - len(needle) + 1):
        if ids[i : i + len(needle)] == needle:
            return i
    raise ValueError(f"sentinel {needle} not found in template rendering {ids}")


class ChatRenderer:
    def __init__(self, checkpoint: str, *, template_kwargs: dict | None = None, generation_suffix: str = ""):
        from transformers import AutoTokenizer

        self.checkpoint = checkpoint
        self.template_kwargs = dict(template_kwargs or {})
        self.generation_suffix = generation_suffix
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._text_cache: dict[int, str] = {}
        added = sorted(self.tokenizer.get_added_vocab(), key=len, reverse=True)
        self._control_strings = re.compile("|".join(map(re.escape, added))) if added else None
        self.stop_token_ids = self._declared_stop_token_ids()
        self.scaffold = self._discover_scaffold(_SENTINELS)
        self.trimming = self._discover_trimming(_SENTINELS)
        self._check_against_template(_CHECK_SENTINELS)
        if generation_suffix:
            self.scaffold[(USER, GENERATION)] = self.scaffold[(USER, GENERATION)] + self.encode_content_as_template(
                generation_suffix
            )

    @classmethod
    def for_thinking(cls, checkpoint: str, thinking: str | None, control) -> "ChatRenderer":
        return cls(
            checkpoint,
            template_kwargs=control.template_kwargs(thinking),
            generation_suffix=control.generation_suffix(thinking),
        )

    def is_plain(self, text: str) -> bool:
        return self._control_strings is None or self._control_strings.search(text) is None

    def encode_content(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)

    def encode_content_as_template(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_content_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoding = self.tokenizer(
            text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True
        )
        return list(encoding["input_ids"]), [tuple(span) for span in encoding["offset_mapping"]]

    def text(self, token_id: int) -> str:
        if token_id not in self._text_cache:
            self._text_cache[token_id] = self.tokenizer.decode([token_id])
        return self._text_cache[token_id]

    def texts(self, ids: list[int]) -> list[str]:
        return [self.text(i) for i in ids]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def render(self, messages: list[dict], *, add_generation_prompt: bool = False) -> Rendered:
        return self._render(messages, self.encode_content, add_generation_prompt)

    def render_as_template(self, messages: list[dict], *, add_generation_prompt: bool = False) -> Rendered:
        spliced = self._render(messages, self.encode_content_as_template, add_generation_prompt)
        token_ids = self._template_ids(messages, add_generation_prompt)
        if spliced.token_ids != token_ids:
            raise ValueError(
                f"content of {messages} merges into the template's scaffold: spliced {self.decode(spliced.token_ids)!r}"
                f" against the template's {self.decode(token_ids)!r}"
            )
        return Rendered(token_ids, spliced.message_indices, spliced.is_content, spliced.message_roles)

    def content_as_template(self, role: str, content: str) -> str:
        strip_leading, strip_trailing = self.trimming.get(role, (False, False))
        if strip_leading:
            content = content.lstrip()
        if strip_trailing:
            content = content.rstrip()
        return content

    def _render(self, messages: list[dict], encode, add_generation_prompt: bool) -> Rendered:
        token_ids: list[int] = []
        message_indices: list[int] = []
        is_content: list[bool] = []

        def emit(ids: list[int], message_index: int) -> None:
            token_ids.extend(ids)
            message_indices.extend([message_index] * len(ids))
            is_content.extend([message_index >= 0] * len(ids))

        previous = START
        for k, message in enumerate(messages):
            role = message["role"]
            if role == ASSISTANT and k == len(messages) - 1 and not add_generation_prompt:
                role = FINAL_ASSISTANT
            emit(self.segment(previous, role), -1)
            emit(encode(self.content_as_template(message["role"], message["content"])), k)
            previous = role
        emit(self.segment(previous, GENERATION if add_generation_prompt else END), -1)
        return Rendered(token_ids, message_indices, is_content, [m["role"] for m in messages])

    def segment(self, previous: str, role: str) -> list[int]:
        if (previous, role) not in self.scaffold:
            raise ValueError(f"the chat template offers no transition {previous} -> {role} the renderer has read")
        return self.scaffold[(previous, role)]

    @property
    def generation_prompt(self) -> list[int]:
        return self.segment(USER, GENERATION)

    @property
    def assistant_turn_end(self) -> list[int]:
        return self.scaffold[(ASSISTANT, USER)][:1]

    @property
    def turn_end_token_ids(self) -> set[int]:
        return set(self.assistant_turn_end) | set(self.stop_token_ids)

    def strip_turn_end(self, ids: list[int]) -> list[int]:
        end_ids = self.turn_end_token_ids
        end = len(ids)
        while end > 0 and ids[end - 1] in end_ids:
            end -= 1
        return list(ids[:end])

    def _template_ids(self, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
                **self.template_kwargs,
            )
        )

    def _declared_stop_token_ids(self) -> list[int]:
        from transformers import GenerationConfig

        declared = [self.tokenizer.eos_token_id]
        try:
            eos = GenerationConfig.from_pretrained(self.checkpoint).eos_token_id
        except OSError:
            eos = None
        declared += eos if isinstance(eos, list) else [eos]
        return sorted({int(i) for i in declared if i is not None})

    def _discover_scaffold(self, sentinels: dict[str, str]) -> dict[tuple[str, str], list[int]]:
        word = {role: self.tokenizer.encode(text, add_special_tokens=False) for role, text in sentinels.items()}
        for role, ids in word.items():
            if not self.is_plain(sentinels[role]) or not ids:
                raise ValueError(f"sentinel {sentinels[role]!r} does not tokenize as plain content")

        def message(role: str) -> dict:
            return {"role": role, "content": sentinels[role]}

        def cuts(ids: list[int], roles: list[str]) -> list[tuple[int, int]]:
            spans, after = [], 0
            for role in roles:
                start = _find(ids, word[role], after)
                spans.append((start, start + len(word[role])))
                after = start + len(word[role])
            return spans

        scaffold: dict[tuple[str, str], list[int]] = {}
        roles = [SYSTEM, USER, ASSISTANT, USER, ASSISTANT]
        ids = self._template_ids([message(r) for r in roles])
        spans = cuts(ids, roles)
        transitions = [
            (START, SYSTEM),
            (SYSTEM, USER),
            (USER, ASSISTANT),
            (ASSISTANT, USER),
            (USER, FINAL_ASSISTANT),
            (FINAL_ASSISTANT, END),
        ]
        bounds = [0, *(position for span in spans for position in span), len(ids)]
        for (previous, role), start, end in zip(transitions, bounds[::2], bounds[1::2]):
            scaffold[(previous, role)] = ids[start:end]

        ids = self._template_ids([message(USER)])
        ((start, end),) = cuts(ids, [USER])
        scaffold[(START, USER)] = ids[:start]
        scaffold[(USER, END)] = ids[end:]
        ids = self._template_ids([message(USER)], add_generation_prompt=True)
        ((start, end),) = cuts(ids, [USER])
        scaffold[(USER, GENERATION)] = ids[end:]

        terminator = scaffold[(ASSISTANT, USER)][:1]
        final_closing = scaffold[(FINAL_ASSISTANT, END)]
        if not terminator:
            raise ValueError("the chat template writes nothing between an assistant turn and the next user turn")
        if final_closing:
            if final_closing[:1] != terminator:
                raise ValueError("the chat template does not terminate every assistant turn with the same token")
        elif terminator[0] not in self.stop_token_ids:
            raise ValueError(
                "the chat template leaves the final assistant turn open, and the token opening the next turn "
                f"({self.decode(terminator)!r}) is not one generation stops on"
            )
        return scaffold

    def _discover_trimming(self, sentinels: dict[str, str]) -> dict[str, tuple[bool, bool]]:
        roles = [SYSTEM, USER, ASSISTANT, USER, ASSISTANT]
        base = [{"role": r, "content": sentinels[r]} for r in roles]
        reference = self._template_ids(base)
        trimming = {}
        for role in (SYSTEM, USER, ASSISTANT):
            sides = []
            for pad in (" \n{}", "{} \n"):
                verdicts = set()
                for k, r in enumerate(roles):
                    if r != role:
                        continue
                    padded = [dict(m) for m in base]
                    padded[k]["content"] = pad.format(sentinels[role])
                    verdicts.add(self._template_ids(padded) == reference)
                if len(verdicts) != 1:
                    raise ValueError(f"the chat template strips whitespace around {role} content differently by turn")
                sides.append(verdicts.pop())
            trimming[role] = (sides[0], sides[1])
        return trimming

    def _check_against_template(self, sentinels: dict[str, str]) -> None:
        def message(role: str, padded: bool) -> dict:
            return {"role": role, "content": f" \n{sentinels[role]}\n " if padded else sentinels[role]}

        conversations = [
            ([USER], False, False),
            ([USER], True, False),
            ([SYSTEM, USER], True, False),
            ([USER, ASSISTANT, USER], True, False),
            ([SYSTEM, USER, ASSISTANT], False, False),
            ([USER, ASSISTANT, USER, ASSISTANT, USER, ASSISTANT], False, False),
            ([SYSTEM, USER, ASSISTANT, USER, ASSISTANT, USER], True, False),
            ([SYSTEM, USER, ASSISTANT, USER, ASSISTANT], False, True),
            ([USER, ASSISTANT, USER], True, True),
        ]
        for roles, add_generation_prompt, padded in conversations:
            messages = [message(r, padded) for r in roles]
            rendered = self.render(messages, add_generation_prompt=add_generation_prompt)
            expected = self._template_ids(messages, add_generation_prompt)
            same = (
                self.decode(rendered.token_ids) == self.decode(expected) if padded else rendered.token_ids == expected
            )
            if not same:
                raise ValueError(
                    f"rendering {roles} (generation prompt {add_generation_prompt}) by construction gives "
                    f"{self.decode(rendered.token_ids)!r} but the chat template gives {self.decode(expected)!r}"
                )
            for k, role in enumerate(roles):
                start, end = rendered.content_span(k)
                content = self.content_as_template(role, messages[k]["content"])
                if self.decode(rendered.token_ids[start:end]) != content:
                    raise ValueError(f"content span of {role} turn {k} does not hold its content")


def check_server_parity(renderer: ChatRenderer, base_url: str, served_model: str, timeout: float = 300) -> None:
    sentinel = _CHECK_SENTINELS
    conversations = [
        ([{"role": USER, "content": sentinel[USER]}, {"role": ASSISTANT, "content": sentinel[ASSISTANT]}], False),
        (
            [
                {"role": USER, "content": sentinel[USER]},
                {"role": ASSISTANT, "content": sentinel[ASSISTANT]},
                {"role": USER, "content": sentinel[USER]},
            ],
            True,
        ),
        ([{"role": USER, "content": sentinel[USER]}], True),
        ([{"role": SYSTEM, "content": sentinel[SYSTEM]}, {"role": USER, "content": sentinel[USER]}], True),
    ]
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        for messages, add_generation_prompt in conversations:
            response = client.post(
                "/tokenize",
                json={"model": served_model, "messages": messages, "add_generation_prompt": add_generation_prompt},
            )
            response.raise_for_status()
            served = response.json()["tokens"]
            local = renderer.render(messages, add_generation_prompt=add_generation_prompt).token_ids
            if served != local:
                raise ValueError(
                    f"local chat template renders {messages} (generation prompt {add_generation_prompt}) to "
                    f"{local} but the server's /tokenize to {served}"
                )
