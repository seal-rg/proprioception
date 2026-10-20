import asyncio
import itertools
import random
import re

import httpx

from proprioception.answer_scan import READERS
from proprioception.batching import ANSWER_PATH_SEGMENT
from proprioception.oracle import DEFAULT_REASONING, ReasoningMarkers, visible_assistant_content
from proprioception.formats import (
    EXPERT_ID_TYPES,
    ANSWER_SCAN_TOKEN,
    ANSWER_SCAN_DECIMAL,
    LENS_LAYER_TYPES,
    ROUTING_TYPES,
    SINGLE_TOKEN_TYPES,
    TOP_TOKEN_MAX_RANK,
    conv_answer_mode,
    draw_lens_layer,
    scanned_answer_bound,
    scanned_answer_names_a_layer,
)
from proprioception.questions import (
    ANSWER_PREFIX_BANKS,
    CONV_ANSWER_PREFIXES,
    compose_conv_question,
    final_quoted_span,
    format_token_ref,
    parse_bounded_token,
    quoted_span,
    token_reply_matches,
)
from proprioception.rendering import ASSISTANT, SYSTEM, USER, ChatRenderer, check_server_parity
from proprioception.thinking import ThinkBlock, compose_think

TOKEN_ANSWER_MAX_CHARS = 60
TOKEN_ANSWER_QUOTE_MAX_CHARS = 40

_FOLLOWING_TURN = {"role": USER, "content": "?"}


def is_number(text: str) -> bool:
    return bool(text) and text.isascii() and text.isdigit()


ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def committed_answer_span(
    token_texts: list[str],
    *,
    markers: ReasoningMarkers = DEFAULT_REASONING,
    layer_word_is_answer: bool = False,
) -> tuple[int, int] | None:
    joined = ""
    scan_start = 0
    for i, s in enumerate(token_texts):
        joined += s
        if joined.endswith(markers.close):
            scan_start = i + 1
    if markers.open in joined.split(markers.close)[-1]:
        return None

    def digits(index: int) -> str:
        return token_texts[index].strip()

    layer_words = () if layer_word_is_answer else ("layer", "layers", "depth", "level")

    def follows_layer_word(index: int) -> bool:
        words = 0
        for j in range(index - 1, -1, -1):
            text = token_texts[j].strip().strip("-–—:,(").lower()
            if not text:
                continue
            if text in layer_words or (words == 0 and text == "top"):
                return True
            words += 1
            if words >= 2:
                return False
        return False

    i = scan_start
    while i < len(token_texts):
        text = digits(i)
        if not is_number(text):
            i += 1
            continue
        run_end = i
        while (
            run_end + 1 < len(token_texts)
            and is_number(token_texts[run_end + 1])
            and token_texts[run_end + 1] == digits(run_end + 1)
        ):
            run_end += 1
        ordinal = run_end + 1 < len(token_texts) and token_texts[run_end + 1].lower() in ORDINAL_SUFFIXES
        if not follows_layer_word(i) and not ordinal:
            return i, run_end - i + 1
        i = run_end + 1
    return None


def committed_decimal_span(
    token_texts: list[str], whole_bound: int, *, markers: ReasoningMarkers = DEFAULT_REASONING
) -> tuple[int, int] | None:
    joined = ""
    scan_start = 0
    for i, piece in enumerate(token_texts):
        joined += piece
        if joined.endswith(markers.close):
            scan_start = i + 1
    if markers.open in joined.split(markers.close)[-1]:
        return None

    def shown(index: int) -> str:
        return token_texts[index].strip()

    def continues(index: int) -> bool:
        return not token_texts[index][:1].isspace()

    i = scan_start
    while i < len(token_texts):
        if not is_number(shown(i)):
            i += 1
            continue
        end = i
        while end + 1 < len(token_texts) and is_number(shown(end + 1)) and continues(end + 1):
            end += 1
        opens_fraction = end + 2 < len(token_texts) and shown(end + 1) == "." and continues(end + 1)
        if (
            opens_fraction
            and continues(end + 2)
            and token_texts[end + 2][:1].isdigit()
            and int("".join(shown(k) for k in range(i, end + 1))) < whole_bound
        ):
            return i, end - i + 3
        i = end + 1
    return None


CLOSE_THINK_PREFIX = "close_think_prefix"


def applies_fallback(fallback: str, *, dangling: bool) -> bool:
    if fallback == CLOSE_THINK_PREFIX:
        return dangling
    return fallback != "none"


def unclosed_answer(text: str, bound: int) -> int | None:
    match = re.search(r"(\d+)[\s`*\"'’”.)\]]*$", text)
    if match is None:
        return None
    digits = match.group(1).lstrip("0") or "0"
    return int(digits) if len(digits) <= len(str(bound)) and int(digits) < bound else None


def committed_answer_position(token_texts: list[str]) -> int | None:
    span = committed_answer_span(token_texts)
    return None if span is None else span[0]


def reasoning_close_text(opening: str, markers: ReasoningMarkers) -> str:
    at = opening.rfind(markers.close)
    return "" if at < 0 else opening[at:]


def template_close_text(opening: str, nothink_opening: str, markers: ReasoningMarkers) -> str:
    if nothink_opening.startswith(opening):
        return nothink_opening[len(opening) :]
    return reasoning_close_text(nothink_opening, markers)


class ConversationAssembler:
    def __init__(
        self,
        base_url: str,
        served_model: str,
        prefix_suffix: str,
        *,
        multi_digit: bool = False,
        markers: ReasoningMarkers = DEFAULT_REASONING,
        check_server: bool = True,
        renderer: ChatRenderer | None = None,
        answer_instruction: bool = True,
        reasoning_close: str = "",
        template_close: str = "",
        template_think: tuple[str, ...] | list[str] = (),
        template_think_style: str = "prose",
        answer_parser: str = "first",
        token_answer_bounded: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.renderer = renderer or ChatRenderer(served_model)
        self.markers = markers
        self.answer_instruction = answer_instruction
        if answer_parser != "first" and answer_parser not in READERS:
            raise ValueError(f"unknown answer parser {answer_parser!r}")
        self.answer_parser = answer_parser
        self.token_answer_bounded = token_answer_bounded
        self.multi_digit = multi_digit
        self.seg_assistant_to_user = self.renderer.segment(ASSISTANT, USER)
        self.seg_user_to_assistant = self.renderer.generation_prompt
        opening = self.renderer.decode(self.seg_user_to_assistant)
        self.answer_opens_reasoning = self.markers.open in opening and self.markers.close not in opening
        self.turn_end_ids = self.renderer.assistant_turn_end
        self.end_token_ids = self.renderer.turn_end_token_ids
        if check_server:
            check_server_parity(self.renderer, self.base_url, self.served_model)
        self.prefix_suffix = prefix_suffix
        self.template_think = tuple(template_think)
        self.template_think_style = template_think_style
        self.closing_ids = self.renderer.encode_content_as_template(reasoning_close or f"{self.markers.close}\n\n")
        self.template_closing_ids = (
            (self.renderer.encode_content_as_template(template_close) if template_close else self.closing_ids)
            if self.answer_opens_reasoning
            else []
        )

    def detokenize(self, token_id: int) -> str:
        return self.renderer.text(token_id)

    def content_ids(self, content: str) -> list[int]:
        return self.renderer.encode_content(content)

    def generation_prompt_ids(self, user_prompt: str) -> list[int]:
        return self.renderer.render([{"role": USER, "content": user_prompt}], add_generation_prompt=True).token_ids

    def strip_turn_end(self, ids: list[int]) -> list[int]:
        return self.renderer.strip_turn_end(ids)

    def completion_text(self, ids: list[int]) -> str:
        return self.renderer.decode(self.strip_turn_end(ids)).strip()

    def full_reply_text(self, ids: list[int]) -> str:
        opened = self.markers.open if self.answer_opens_reasoning else ""
        return opened + self.completion_text(ids)

    def carrier_reply(self, ids: list[int]) -> str:
        return visible_assistant_content(self.full_reply_text(ids), self.markers)

    def build_conversation(self, user_prompt: str, assistant_reply: str, *, system_prompt: str = "") -> dict:
        rendered = self.renderer.render(self.capture_messages(user_prompt, assistant_reply, system_prompt))
        spans = [(s, e, r) for s, e, r in rendered.content_spans() if r in (USER, ASSISTANT)][:2]
        return {"ids": rendered.token_ids[: spans[-1][1]], "spans": spans}

    def capture_messages(self, user_prompt: str, assistant_reply: str, system_prompt: str | None = None) -> list[dict]:
        messages = [
            {"role": USER, "content": user_prompt},
            {"role": ASSISTANT, "content": assistant_reply},
            _FOLLOWING_TURN,
        ]
        if system_prompt:
            messages.insert(0, {"role": SYSTEM, "content": system_prompt})
        return messages

    def build_probe(
        self,
        conversation: dict,
        *,
        rng: random.Random,
        want_role: str,
        pc_type: str,
        moe_layers: list[int],
        num_hidden_layers: int,
        num_experts_per_tok: int,
        lens_token_prob_depth: float | None = None,
    ) -> dict | None:
        positions = [
            i
            for start, end, role in conversation["spans"]
            if role == want_role
            for i in range(start, end)
            if self.detokenize(conversation["ids"][i]).strip()
        ]
        if pc_type not in ROUTING_TYPES:
            ends = {end - 1 for _, end, role in conversation["spans"] if role == want_role}
            positions = [i for i in positions if i not in ends]
        if not positions:
            return None
        pos = rng.choice(positions)
        token_id = conversation["ids"][pos]
        span_start, span_end = next((s, e) for s, e, r in conversation["spans"] if s <= pos < e)
        same = [i for i in range(span_start, span_end) if conversation["ids"][i] == token_id]
        if pc_type in ROUTING_TYPES:
            layer = rng.choice(moe_layers)
        elif pc_type in LENS_LAYER_TYPES:
            layer = draw_lens_layer(rng, pc_type, num_hidden_layers - 1, lens_token_prob_depth)
        else:
            layer = -1
        if pc_type == "routing_rank":
            rank = rng.randint(2, num_experts_per_tok)
        elif pc_type in ("top_token", "lens_top_token"):
            rank = rng.randint(1, TOP_TOKEN_MAX_RANK)
        else:
            rank = 1
        location = "in my first message" if want_role == "user" else "in your reply"
        token_text = self.detokenize(token_id)
        reference = format_token_ref(token_text, same.index(pos) + 1, len(same))
        token_ref = f"{reference} {location}"
        think_token_ref = f"{reference} {"in the user's first message" if want_role == 'user' else 'in my reply'}"
        window = self.renderer.decode(conversation["ids"][max(span_start, pos - 2) : min(span_end, pos + 3)])
        context_ref = f"'{token_text}' as in '{window.strip()}'"
        question = compose_conv_question(
            rng,
            pc_type,
            token_ref=token_ref,
            layer=layer,
            rank=rank,
            num_layers=num_hidden_layers,
            instruction=self.answer_instruction,
        )
        question_ids = self.content_ids(question)
        question_start = len(conversation["ids"]) + len(self.seg_assistant_to_user)
        input_ids = conversation["ids"] + self.seg_assistant_to_user + question_ids + self.seg_user_to_assistant
        return {
            "input_ids": input_ids,
            "probe_pos": pos,
            "type": pc_type,
            "layer": layer,
            "rank": rank,
            "segments": {"user_pc_question": [[question_start, question_start + len(question_ids)]]},
            "partner_span": [span_start, span_end],
            "probe_role": want_role,
            "question": question,
            "token_ref": token_ref,
            "think_token_ref": think_token_ref,
            "context_ref": context_ref,
        }

    def append_answer(
        self,
        probe: dict,
        answer_ids: list[int],
        *,
        num_experts: int,
        num_hidden_layers: int,
        terminated: bool,
        fallback: str = "none",
        rng: random.Random | None = None,
    ) -> dict:
        pc_type = probe.get("type", "routing")
        mode = conv_answer_mode(pc_type)
        ids = self.strip_turn_end(answer_ids)
        texts = self.renderer.texts(ids)
        text = self.renderer.decode(ids).strip()
        opened = self.markers.open if self.answer_opens_reasoning else ""
        if not text and applies_fallback(fallback, dangling=bool(opened)):
            fields = {"think_ok": not opened, "think_block": False, "answer_tokens": 0, "answer_text": ""}
            return self.unparsed_readout(probe, ids, texts, dangling=bool(opened), fallback=fallback, rng=rng, **fields)
        if not text:
            return {
                **probe,
                "answer_pos": None,
                "model_answer": None,
                "think_ok": False,
                "think_block": False,
                "answer_tokens": 0,
                "answer_text": "",
            }
        joined = "".join(texts)
        think_ok = self.markers.open not in (opened + joined).split(self.markers.close)[-1]
        think_block = think_ok and self.markers.open in (opened + joined)
        answer_start = len(probe["input_ids"])
        input_ids = probe["input_ids"] + ids + (self.turn_end_ids if terminated else [])
        analysed = ([self.markers.open] if self.answer_opens_reasoning else []) + texts
        offset = 1 if self.answer_opens_reasoning else 0
        model_answer = None
        if mode == ANSWER_SCAN_TOKEN:
            read = self.token_answer_slot(pc_type, ids, texts) if think_ok else None
            span = None
            if read is not None:
                ids, texts, slot, model_answer = read
                input_ids = probe["input_ids"] + ids + (self.turn_end_ids if terminated else [])
                span = (slot, 1)
        else:
            bound = scanned_answer_bound(pc_type, num_experts, num_hidden_layers)
            if mode == ANSWER_SCAN_DECIMAL:
                span = committed_decimal_span(analysed, bound, markers=self.markers)
                span = None if span is None else (span[0] - offset, span[1])
            elif self.answer_parser != "first" and pc_type in EXPERT_ID_TYPES:
                span = self.read_answer_span(texts, probe.get("question", ""))
            else:
                span = committed_answer_span(
                    analysed, markers=self.markers, layer_word_is_answer=scanned_answer_names_a_layer(pc_type)
                )
                span = None if span is None else (span[0] - offset, span[1])
            if span is not None:
                start, length = span
                shown = [s.strip() for s in texts[start : start + length]]
                if mode == ANSWER_SCAN_DECIMAL:
                    model_answer = int("".join(shown[:-2])) * 10 + int(shown[-1][0])
                else:
                    digits = "".join(shown).lstrip("0") or "0"
                    if len(digits) <= len(str(bound)) and int(digits) < bound:
                        model_answer = int(digits)
                if mode != ANSWER_SCAN_DECIMAL and not self.multi_digit:
                    span = (start, 1)
        if span is None and applies_fallback(fallback, dangling=not think_ok):
            fields = {"think_ok": think_ok, "think_block": think_block, "answer_tokens": len(ids), "answer_text": text}
            if fallback == CLOSE_THINK_PREFIX and terminated and mode not in (ANSWER_SCAN_TOKEN, ANSWER_SCAN_DECIMAL):
                fields["unclosed_answer"] = unclosed_answer(
                    text, scanned_answer_bound(pc_type, num_experts, num_hidden_layers)
                )
            return self.unparsed_readout(probe, ids, texts, dangling=not think_ok, fallback=fallback, rng=rng, **fields)
        segments = probe.get("segments", {})
        if span is not None and span[0] > 0:
            segments = {**segments, ANSWER_PATH_SEGMENT: [[answer_start, answer_start + span[0]]]}
        return {
            **probe,
            "segments": segments,
            "input_ids": input_ids,
            "answer_pos": None if span is None else answer_start + span[0],
            "answer_start": None if span is None else answer_start + span[0],
            "emitted_digits": 0 if span is None else span[1],
            "model_answer": model_answer,
            "think_ok": think_ok,
            "think_block": think_block,
            "answer_tokens": len(ids),
            "answer_text": text,
        }

    def token_answer_slot(
        self, pc_type: str, ids: list[int], texts: list[str]
    ) -> tuple[list[int], list[str], int, str] | None:
        start = self.visible_start(texts)
        visible = "".join(texts[start:])
        prefixes = CONV_ANSWER_PREFIXES[pc_type]
        if (
            parse_bounded_token(visible, max_chars=TOKEN_ANSWER_MAX_CHARS, prefixes=prefixes, markers=self.markers)
            is not None
        ):
            answer = visible.strip()
            lead = len(visible) - len(visible.lstrip())
            quoted = quoted_span(answer)
            if quoted is not None:
                offset = quoted[0]
            else:
                offset = 0
                for prefix in sorted(prefixes, key=len, reverse=True):
                    rest = answer[len(prefix) :]
                    if answer.startswith(prefix) and rest.strip():
                        offset = len(prefix) + len(rest) - len(rest.lstrip())
                        break
                while offset < len(answer) - 1 and answer[offset] in "'`":
                    offset += 1
            char = lead + offset
        elif self.token_answer_bounded or self.markers.open in visible or self.markers.close in visible:
            return None
        else:
            quoted = final_quoted_span(visible, TOKEN_ANSWER_QUOTE_MAX_CHARS)
            if quoted is None:
                return None
            char = quoted[0]
            answer = visible[quoted[0] - 1 : quoted[1] + 1]
        ends = list(itertools.accumulate(len(piece) for piece in texts[start:]))
        k = next((i for i, end in enumerate(ends) if end > char), None)
        if k is None:
            return None
        cut = char - (ends[k] - len(texts[start + k]))
        head, tail = texts[start + k][:cut], texts[start + k][cut:]
        slot = start + k
        if head.strip():
            pieces = self.renderer.encode_content(head), self.renderer.encode_content(tail)
            if "".join(self.renderer.texts(pieces[0] + pieces[1])) != texts[start + k]:
                return None
            ids = ids[:slot] + pieces[0] + pieces[1] + ids[slot + 1 :]
            texts = self.renderer.texts(ids)
            slot += len(pieces[0])
        return ids, texts, slot, answer

    def names_label(self, pc_type: str, answer: str | None, label: int, none_token: int | None) -> bool:
        if answer is None or label < 0:
            return False
        names_none = label == none_token
        return token_reply_matches(
            answer,
            self.renderer.text(label),
            names_none=names_none,
            max_chars=TOKEN_ANSWER_MAX_CHARS,
            prefixes=CONV_ANSWER_PREFIXES[pc_type],
            markers=self.markers,
        )

    def visible_start(self, texts: list[str]) -> int:
        joined = self.markers.open if self.answer_opens_reasoning else ""
        start = 0
        for i, piece in enumerate(texts):
            joined += piece
            if joined.endswith(self.markers.close):
                start = i + 1
        while start < len(texts) and not texts[start].strip():
            start += 1
        return start

    def read_answer_span(self, texts: list[str], question: str) -> tuple[int, int] | None:
        opened = self.markers.open if self.answer_opens_reasoning else ""
        if self.markers.open in (opened + "".join(texts)).split(self.markers.close)[-1]:
            return None
        start = self.visible_start(texts)
        found = READERS[self.answer_parser]("".join(texts[start:]), question)
        if found is None:
            return None
        position, digits = found
        ends = list(itertools.accumulate(len(piece) for piece in texts[start:]))
        first = next(k for k, end in enumerate(ends) if end > position)
        if texts[start + first][: position - (ends[first] - len(texts[start + first]))].strip():
            return None
        last = next(k for k, end in enumerate(ends) if end >= position + len(digits))
        if "".join(piece.strip() for piece in texts[start + first : start + last + 1]) != digits:
            return None
        return start + first, last - first + 1

    def answer_prefix_ids(self, probe: dict, rng: random.Random | None) -> list[int]:
        if rng is None:
            raise ValueError("RNG not forwarded.")
        pc_type = probe.get("type", "routing")
        sentence = rng.choice(ANSWER_PREFIX_BANKS[pc_type])
        sentence += ' "' if pc_type in SINGLE_TOKEN_TYPES else self.prefix_suffix
        return self.content_ids(sentence)

    def template_answer(self, probe: dict, rng: random.Random | None) -> dict:
        if rng is None:
            raise ValueError("RNG not forwarded.")
        prefix = self.answer_prefix_ids(probe, rng)
        opening_end = len(probe["input_ids"])
        ((_, question_end),) = probe["segments"]["user_pc_question"]
        block = ThinkBlock([])
        if self.template_think and self.template_closing_ids:
            block = compose_think(
                rng,
                probe.get("type", "routing"),
                self.template_think,
                question=probe["question"],
                token_ref=probe["think_token_ref"],
                layer=probe.get("layer", -1),
                rank=probe.get("rank", 1),
                context_ref=probe.get("context_ref"),
                prefix_suffix=self.prefix_suffix,
                style=self.template_think_style,
            )
        block_ids = self.content_ids(block.head) if block.head else []
        segments = {**probe.get("segments", {}), "assistant_pc_formatting": [[question_end, opening_end]]}
        row = {
            **probe,
            "model_answer": None,
            "completed": True,
            "think_ok": True,
            "think_block": bool(self.template_closing_ids)
            or self.markers.close in self.renderer.decode(self.seg_user_to_assistant),
            "answer_tokens": 0,
            "answer_text": "",
        }
        if block.sense is not None:
            sense_ids = self.content_ids(block.sense)
            slot = opening_end + len(block_ids) + len(sense_ids)
            segments["assistant_pc_reasoning"] = [[opening_end, slot]]
            tail_segments = {
                "assistant_pc_reasoning": [[0, len(self.template_closing_ids)]],
                "assistant_pc_response": [
                    [len(self.template_closing_ids), len(self.template_closing_ids) + len(prefix)]
                ],
            }
            return {
                **row,
                "input_ids": probe["input_ids"] + block_ids + sense_ids,
                "segments": segments,
                "tail": {"input_ids": self.template_closing_ids + prefix, "segments": tail_segments},
            }
        start = opening_end + len(block_ids) + len(self.template_closing_ids)
        if start > opening_end:
            segments["assistant_pc_reasoning"] = [[opening_end, start]]
        segments["assistant_pc_response"] = [[start, start + len(prefix)]]
        return {
            **row,
            "input_ids": probe["input_ids"] + block_ids + self.template_closing_ids + prefix,
            "segments": segments,
        }

    def unparsed_readout(
        self,
        probe: dict,
        ids: list[int],
        texts: list[str],
        *,
        dangling: bool,
        fallback: str,
        rng: random.Random | None,
        **fields,
    ) -> dict:
        prompt = probe["input_ids"]
        if fallback == "close":
            if dangling:
                kept = ids + self.closing_ids + self.turn_end_ids
                slot = len(prompt) + len(ids) + len(self.closing_ids)
            else:
                kept = ids + self.turn_end_ids
                slot = len(prompt) + self.visible_start(texts)
        elif fallback == "close_prefix":
            reasoning = ids + self.closing_ids if dangling else ids[: self.visible_start(texts)]
            answer = self.answer_prefix_ids(probe, rng)
            kept = reasoning + answer + self.turn_end_ids
            slot = len(prompt) + len(reasoning) + len(answer)
        elif fallback == CLOSE_THINK_PREFIX:
            if not dangling:
                raise ValueError(f"{CLOSE_THINK_PREFIX} completes only a turn whose reasoning is still open")
            return {
                **probe,
                **fields,
                "input_ids": prompt + ids + self.closing_ids + self.answer_prefix_ids(probe, rng),
                "model_answer": None,
                "fallback": fallback,
                "completed": True,
            }
        else:
            raise ValueError(f"unknown unparsed-answer fallback {fallback!r}")
        return {
            **probe,
            **fields,
            "input_ids": prompt + kept,
            "answer_pos": slot,
            "answer_start": slot,
            "emitted_digits": 1,
            "model_answer": None,
            "fallback": fallback,
        }


def generate_completions(
    base_url: str,
    served_model: str,
    prompt_token_ids: list[list[int]],
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    timeout: float = 600,
) -> list[tuple[list[int], bool]]:

    async def _one(client: httpx.AsyncClient, i: int, prompt: list[int]) -> tuple[list[int], bool]:
        response = await client.post(
            "/inference/v1/generate",
            json={
                "model": served_model,
                "token_ids": prompt,
                "sampling_params": {"max_tokens": max_tokens, "temperature": temperature, "seed": seed + i},
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return list(choice.get("token_ids") or []), choice.get("finish_reason") == "stop"

    async def _all() -> list[tuple[list[int], bool]]:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=64)
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, limits=limits) as client:
            return list(await asyncio.gather(*(_one(client, i, p) for i, p in enumerate(prompt_token_ids))))

    return asyncio.run(_all())
