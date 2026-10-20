import asyncio
import collections
import contextvars
import dataclasses
import functools
import itertools
import logging
import random
from pathlib import Path
from typing import Literal, cast

import verifiers.v1 as vf
from pydantic import model_validator
from verifiers.v1.types import content_text

from proprioception.bank import BANK, BANK_EVAL_PROMPTS, load_prompts
from proprioception.families import FAMILIES, family_for_checkpoint, moe_structure, read_config
from proprioception.formats import (
    ENTROPY_TYPES,
    draw_lens_layer,
    EXPERT_ID_TYPES,
    LENS_LAYER_TYPES,
    ROUTING_TYPES,
    SCALAR_TYPES,
    SINGLE_TOKEN_TYPES,
    TOP_TOKEN_MAX_RANK,
    resolve_types,
    scanned_answer_bound,
)
from proprioception.oracle import (
    ConversationRouting,
    PolicySample,
    ReasoningMarkers,
    RoutingOracle,
    token_occurrence,
    visible_assistant_content,
)
from proprioception.probe_labels import NONE_PARTNER, needs_readout, readout_label, routing_capture_label
from proprioception.questions import (
    CONV_ANSWER_PREFIXES,
    compose_conv_question,
    compose_open_expert_question,
    compose_token_question,
    format_token_ref,
    parse_bare_expert_id,
    parse_bounded_decimal,
    parse_bounded_expert_id,
    parse_bounded_integer,
    parse_bounded_token,
    parse_expert_id,
    parse_letter_answer,
    render_token_context,
    token_reply_matches,
)
from proprioception.rendering import Rendered, tokens_within

logger = logging.getLogger(__name__)


class ProprioceptionState(vf.State):
    question_asked: bool = False
    question: str | None = None
    conversation_over: bool = False
    correct_letter: str | None = None
    true_expert: int | None = None
    distractor_expert: int | None = None
    probed_token_index: int | None = None
    probed_layer: int | None = None
    probed_role: str | None = None
    open_format: bool = False
    oracle_failure: str | None = None
    probe_token_ids: list[int] | None = None
    probe_type: str = "routing"
    probed_rank: int = 1
    label: int | None = None
    label_text: str | None = None
    answer_bound: int | None = None
    lexical_label: int | None = None


class ProprioceptionUserConfig(vf.UserConfig):
    oracle_base_url: str = "http://localhost:8000"
    oracle_model: str = ""
    num_experts: int | None = None
    probed_layer: int | None = None
    probed_layers: list[int] = []
    assistant_token_fraction: float = 0.5
    local_carrier_render: bool = True
    on_failure: Literal["zero_reward", "error"] = "zero_reward"
    context_mismatch: Literal["error", "warn"] = "error"
    answer_format: Literal["loose", "strict", "bounded"] = "loose"
    answer_instruction: bool = True
    answer_max_chars: int = 60
    answer_length_penalty: float = 0.0
    answer_expected_tokens: int = 8
    answer_max_tokens: int = 64
    shared: bool = True
    probe_domain: Literal["routing", "logits", "lens", "mix"] | None = None
    probe_types: list[str] = []
    scalar_tolerance: dict[str, float] = {}
    lens_token_prob_depth: float | None = None
    lexical_contexts: int = 0
    lexical_match_weight: float = 1.0
    lexical_context_bank: str = ""

    @model_validator(mode="after")
    def validate_probe_types(self):
        resolved_probe_types(self)
        if not 0.0 <= self.lexical_match_weight <= 1.0:
            raise ValueError(f"lexical_match_weight {self.lexical_match_weight} must lie in [0, 1]")
        if self.lexical_match_weight < 1.0 and self.lexical_contexts <= 0:
            raise ValueError("lexical_match_weight below 1 needs lexical_contexts to find the lexical probes")
        unknown = set(self.scalar_tolerance) - set(SCALAR_TYPES)
        if unknown:
            raise ValueError(
                f"scalar_tolerance names types {sorted(unknown)} that are not scalar: {list(SCALAR_TYPES)}"
            )
        return self


def resolved_probe_types(config: "ProprioceptionUserConfig") -> list[str]:
    if config.probe_domain is None and not config.probe_types:
        return ["routing"]
    return resolve_types(config.probe_domain or "mix", config.probe_types)


@dataclasses.dataclass(frozen=True)
class ServedModel:
    num_experts: int
    num_experts_per_tok: int
    moe_layers: tuple[int, ...]
    num_hidden_layers: int
    reasoning: ReasoningMarkers


@functools.cache
def served_model(model: str) -> ServedModel:
    if not model:
        raise ValueError("oracle_model checkpoint not set")
    from transformers.utils import cached_file

    resolved = cached_file(model, "config.json")
    if resolved is None:
        raise ValueError(f"{model} has no config.json")
    checkpoint = Path(resolved).parent
    num_experts, num_experts_per_tok, moe_layers, num_hidden_layers = moe_structure(read_config(checkpoint))
    return ServedModel(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_layers=tuple(moe_layers),
        num_hidden_layers=num_hidden_layers,
        reasoning=FAMILIES[family_for_checkpoint(checkpoint)].reasoning,
    )


def routing_layers(config: "ProprioceptionUserConfig", served: ServedModel) -> list[int]:
    if config.num_experts is not None and config.num_experts != served.num_experts:
        raise ValueError(
            f"num_experts = {config.num_experts} but {config.oracle_model} has {served.num_experts} experts"
        )
    named = [] if config.probed_layer is None else [config.probed_layer]
    layers = list(config.probed_layers) or named or list(served.moe_layers)
    dense = sorted(set(layers + named) - set(served.moe_layers))
    if dense:
        raise ValueError(
            f"probed layers {dense} have no router in {config.oracle_model}; its MoE layers are {list(served.moe_layers)}"
        )
    return layers


def check_routing_shape(capture: ConversationRouting, served: ServedModel) -> None:
    if (capture.num_decoder_layers, capture.top_k) != (served.num_hidden_layers, served.num_experts_per_tok):
        raise RuntimeError(
            f"the server routes {capture.num_decoder_layers} layers x top-{capture.top_k}, the checkpoint config "
            f"{served.num_hidden_layers} layers x top-{served.num_experts_per_tok}"
        )


LEXICAL_CONTEXT_TOKENS = 128


@functools.cache
def lexical_context_prompts(path: str) -> list[str]:
    return load_prompts(path or BANK)[0]


async def lexical_label(
    oracle: RoutingOracle,
    config: ProprioceptionUserConfig,
    rng: random.Random,
    *,
    pc_type: str,
    probed_ids: list[int],
    layer: int,
    rank: int,
    served: ServedModel,
) -> int | None:
    if config.lexical_contexts <= 0 or pc_type == "routing_partner":
        return None
    prompts = rng.sample(lexical_context_prompts(config.lexical_context_bank), config.lexical_contexts)

    async def label_in(prompt: str) -> int:
        context = oracle.renderer.render_as_template([{"role": "user", "content": prompt}])
        start, end = context.content_span(0)
        position = min(end, start + LEXICAL_CONTEXT_TOKENS)
        rendered = Rendered(
            token_ids=context.token_ids[:position] + probed_ids,
            message_indices=context.message_indices[:position] + [0] * len(probed_ids),
            is_content=context.is_content[:position] + [True] * len(probed_ids),
            message_roles=context.message_roles,
        )
        if needs_readout(pc_type):
            readout = (await oracle.probe_rendered(rendered, [position])).readouts[position]
            return readout_label(pc_type, readout, layer=layer, rank=rank, following=probed_ids[1])
        capture = await oracle.capture_rendered(rendered)
        check_routing_shape(capture, served)
        return routing_capture_label(pc_type, capture, position=position, layer=layer, rank=rank, span=(0, 0))

    counts = collections.Counter(await asyncio.gather(*(label_in(prompt) for prompt in prompts)))
    return max(counts, key=counts.__getitem__)


_current_task: contextvars.ContextVar = contextvars.ContextVar("proprioception_task")


@dataclasses.dataclass
class ConversationProbe:
    question: str
    true_expert: int | None
    probed_token_index: int
    probed_layer: int
    probed_role: str
    probe_token_ids: list[int]
    type: str = "routing"
    rank: int = 1
    label: int | None = None
    label_text: str | None = None
    answer_bound: int | None = None
    lexical_label: int | None = None


def record_probe(state: ProprioceptionState, probe: ConversationProbe) -> None:
    state.true_expert = probe.true_expert
    state.probed_token_index = probe.probed_token_index
    state.probed_layer = probe.probed_layer
    state.probed_role = probe.probed_role
    state.probe_token_ids = list(probe.probe_token_ids)
    state.question = probe.question
    state.open_format = True
    state.probe_type = probe.type
    state.probed_rank = probe.rank
    state.label = probe.label
    state.label_text = probe.label_text
    state.answer_bound = probe.answer_bound
    state.lexical_label = probe.lexical_label


async def compose_conversation_probe(
    oracle: RoutingOracle,
    config: ProprioceptionUserConfig,
    *,
    seed: str,
    system_prompt: str | None,
    turn1_prompt: str,
    visible_reply: str,
) -> ConversationProbe:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": turn1_prompt})
    messages.append({"role": "assistant", "content": visible_reply})

    carrier = [*messages, {"role": "user", "content": "?"}]
    if config.local_carrier_render:
        rendered = oracle.renderer.render_as_template(carrier)
        carrier_token_texts = oracle.renderer.texts(rendered.token_ids)
    else:
        capture = await oracle.capture(carrier)
        rendered, carrier_token_texts = capture.rendered, capture.token_texts
    spans = [(s, e, r) for s, e, r in rendered.content_spans() if r in ("user", "assistant")][:2]
    served = served_model(oracle.model)
    rng = random.Random(seed)
    types = resolved_probe_types(config)
    pc_type = types[0] if len(types) == 1 else rng.choice(types)
    want_role = "assistant" if rng.random() < config.assistant_token_fraction else "user"
    turn_ends = set() if pc_type in ROUTING_TYPES else {end - 1 for _, end, _ in spans}
    candidates = [
        i
        for start, end, role in spans
        if role == want_role
        for i in range(start, end)
        if carrier_token_texts[i].strip() and i not in turn_ends
    ]
    if not candidates:
        raise ValueError(f"No probeable {want_role} tokens in conversation")
    token_index = rng.choice(candidates)
    span_start, span_end = next((s, e) for s, e, r in spans if s <= token_index < e)
    same = [i for i in range(span_start, span_end) if carrier_token_texts[i] == carrier_token_texts[token_index]]
    location = "in my first message" if want_role == "user" else "in your reply"
    token_ref = (
        format_token_ref(carrier_token_texts[token_index], same.index(token_index) + 1, len(same)) + f" {location}"
    )
    if pc_type == "routing":
        layer = rng.choice(routing_layers(config, served))
        question = compose_open_expert_question(
            rng, token_ref=token_ref, layer=layer, instruction=config.answer_instruction
        )

        probe_messages = [*messages, {"role": "user", "content": question}]
        probe_capture = await oracle.capture(probe_messages)
        if probe_capture.token_ids[: token_index + 1] != rendered.token_ids[: token_index + 1]:
            raise ValueError("Token alignment drifted between conversation capture and probe message")
        check_routing_shape(probe_capture, served)
        true_expert = probe_capture.experts_at(token_index, layer)[0]
        return ConversationProbe(
            question=question,
            true_expert=true_expert,
            probed_token_index=token_index,
            probed_layer=layer,
            probed_role=want_role,
            probe_token_ids=list(probe_capture.token_ids),
            label=true_expert,
            answer_bound=served.num_experts,
            lexical_label=await lexical_label(
                oracle,
                config,
                rng,
                pc_type=pc_type,
                probed_ids=rendered.token_ids[token_index : token_index + 1],
                layer=layer,
                rank=1,
                served=served,
            ),
        )

    num_layers = served.num_hidden_layers
    if pc_type in ROUTING_TYPES:
        layer = rng.choice(routing_layers(config, served))
    elif pc_type in LENS_LAYER_TYPES:
        layer = draw_lens_layer(rng, pc_type, num_layers - 1, config.lens_token_prob_depth)
    else:
        layer = -1
    if pc_type == "routing_rank":
        rank = rng.randint(2, served.num_experts_per_tok)
    elif pc_type in ("top_token", "lens_top_token"):
        rank = rng.randint(1, TOP_TOKEN_MAX_RANK)
    else:
        rank = 1
    question = compose_conv_question(
        rng,
        pc_type,
        token_ref=token_ref,
        layer=layer,
        rank=rank,
        num_layers=num_layers,
        instruction=config.answer_instruction,
    )
    probe_messages = [*messages, {"role": "user", "content": question}]
    aligned = span_end if pc_type == "routing_partner" else token_index + 2
    if needs_readout(pc_type):
        probe_capture = await oracle.probe(probe_messages, [token_index])
        probe_ids = probe_capture.token_ids
        if probe_ids[:aligned] != rendered.token_ids[:aligned]:
            raise ValueError("Token alignment drifted between conversation capture and probe message")
        label = readout_label(
            pc_type,
            probe_capture.readouts[token_index],
            layer=layer,
            rank=rank,
            following=probe_ids[token_index + 1],
        )
    else:
        routing_capture = await oracle.capture(probe_messages)
        probe_ids = routing_capture.token_ids
        if probe_ids[:aligned] != rendered.token_ids[:aligned]:
            raise ValueError("Token alignment drifted between conversation capture and probe message")
        check_routing_shape(routing_capture, served)
        label = routing_capture_label(
            pc_type, routing_capture, position=token_index, layer=layer, rank=rank, span=(span_start, span_end)
        )
    label_text = None
    if pc_type in SINGLE_TOKEN_TYPES:
        label_text = "none" if label == NONE_PARTNER else oracle.renderer.text(label)
        if not label_text:
            raise ValueError(f"the {pc_type} label is a token with no text a reply could write")
    answer_bound = None
    if pc_type not in SINGLE_TOKEN_TYPES:
        answer_bound = scanned_answer_bound(pc_type, served.num_experts, num_layers)
    return ConversationProbe(
        question=question,
        true_expert=label if pc_type in EXPERT_ID_TYPES else None,
        probed_token_index=token_index,
        probed_layer=layer,
        probed_role=want_role,
        probe_token_ids=list(probe_ids),
        type=pc_type,
        rank=rank,
        label=label,
        label_text=label_text,
        answer_bound=answer_bound,
        lexical_label=await lexical_label(
            oracle,
            config,
            rng,
            pc_type=pc_type,
            probed_ids=rendered.token_ids[token_index : token_index + (2 if needs_readout(pc_type) else 1)],
            layer=layer,
            rank=rank,
            served=served,
        ),
    )


class ProprioceptionUser(vf.User[ProprioceptionUserConfig, ProprioceptionState]):
    _process_task = None

    async def setup(self) -> None:
        self.oracle = RoutingOracle(self.config.oracle_base_url, self.config.oracle_model)

    async def setup_task(self, task) -> None:
        self._process_task = task

    @property
    def task_idx(self) -> int:
        return _current_task.get().idx

    @property
    def turn1_prompt(self) -> str | None:
        return _current_task.get().prompt

    @property
    def sentence(self) -> str:
        return getattr(_current_task.get(), "sentence", "") or ""

    @property
    def probe_mode(self) -> str:
        return getattr(_current_task.get(), "probe_mode", "conversation_token") or "conversation_token"

    @property
    def system_prompt(self) -> str | None:
        return _current_task.get().system_prompt

    async def _compose_single_turn(self) -> vf.Messages:
        rng = random.Random()
        context, question_template = render_token_context(rng, self.sentence)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": context})

        capture = await self.oracle.capture(messages)
        served = served_model(self.oracle.model)
        check_routing_shape(capture, served)
        routing_layers(self.config, served)
        layer = self.config.probed_layer
        if layer is None:
            raise ValueError("random_token probes one layer and needs probed_layer set")
        user_start, _ = capture.rendered.content_span(len(messages) - 1)
        context_ids, offsets = self.oracle.renderer.encode_content_with_offsets(context)
        if capture.rendered.token_ids[user_start : user_start + len(context_ids)] != context_ids:
            raise ValueError("the rendered user turn does not carry the context ids its offsets were read from")
        sentence_start = context.index(self.sentence)
        inside = [user_start + i for i in tokens_within(offsets, sentence_start, sentence_start + len(self.sentence))]
        candidates = [i for i in inside if capture.token_texts[i].strip()]
        if not candidates:
            raise ValueError("No probeable tokens inside the quoted sentence")
        pairs = [
            (i, j)
            for i in candidates
            for j in candidates
            if i != j and capture.experts_at(i, layer)[0] != capture.experts_at(j, layer)[0]
        ]
        if not pairs:
            raise ValueError("No probe/donor pair with distinct top-1 experts in the sentence")
        token_index, donor_index = rng.choice(pairs)
        occurrence, total = token_occurrence(capture.token_texts, token_index, (inside[0], inside[-1] + 1))
        token_ref = format_token_ref(capture.token_texts[token_index], occurrence, total)

        true_expert = capture.experts_at(token_index, layer)[0]
        distractor = capture.experts_at(donor_index, layer)[0]
        question = compose_token_question(
            rng,
            question_template,
            token_ref=token_ref,
            layer=layer,
            true_expert=true_expert,
            distractor_expert=distractor,
        )

        full_message = f"{context}\n\n{question.text}"
        check_messages = list(messages[:-1]) + [{"role": "user", "content": full_message}]
        check_ids = self.oracle.renderer.render_as_template(check_messages).token_ids
        if check_ids[: token_index + 1] != capture.token_ids[: token_index + 1]:
            raise ValueError("Token alignment drifted between context capture and full message")

        self.state.correct_letter = question.correct_letter
        self.state.true_expert = true_expert
        self.state.distractor_expert = distractor
        self.state.probed_token_index = token_index
        self.state.probe_token_ids = list(check_ids)
        self.state.question = full_message
        self.state.question_asked = True
        return [vf.UserMessage(content=full_message)]

    async def _compose_conversation_question(self, turn1_reply: str) -> vf.Messages:
        visible_reply = visible_assistant_content(turn1_reply, served_model(self.oracle.model).reasoning)
        if not visible_reply:
            raise ValueError("Turn-1 reply has no visible content outside the reasoning block")
        probe = await compose_conversation_probe(
            self.oracle,
            self.config,
            seed=f"{self.task_idx}:{turn1_reply}",
            system_prompt=self.system_prompt,
            turn1_prompt=self.turn1_prompt or "",
            visible_reply=visible_reply,
        )
        record_probe(self.state, probe)
        self.state.question_asked = True
        return [vf.UserMessage(content=probe.question)]

    async def respond(self, message: str) -> vf.Messages:
        task = await self._fetch_task(*self._state_channel())
        token = _current_task.set(task if task is not None else self._process_task)
        try:
            return await self._respond(message)
        finally:
            _current_task.reset(token)

    async def _respond(self, message: str) -> vf.Messages:
        if self.probe_mode == "chat_only":
            self.state.conversation_over = True
            return []
        if not self.state.question_asked:
            try:
                if self.probe_mode == "conversation_token":
                    return await self._compose_conversation_question(message)
                return await self._compose_single_turn()
            except Exception as error:
                self.state.oracle_failure = f"{type(error).__name__}: {error}"
                if self.config.on_failure == "error":
                    raise
                logger.warning("oracle capture failed, ending rollout: %s", self.state.oracle_failure)
                self.state.conversation_over = True
                return []
        self.state.conversation_over = True
        return []


class ProprioceptionTaskConfig(vf.TaskConfig):
    user: ProprioceptionUserConfig = ProprioceptionUserConfig()
    shared_carrier: bool = False
    shared_probe: bool = True
    carrier_max_tokens: int = 8192
    carrier_temperature: float = 1.0
    on_carrier_truncation: Literal["drop"] = "drop"
    max_prefix_tokens: int = 2560
    carrier_sample: bool = False


class ProprioceptionData(vf.TaskData):
    sentence: str = ""
    probe_mode: str = "conversation_token"


@dataclasses.dataclass
class CarrierTrunk:
    visible_reply: str
    probe: ConversationProbe | None
    seed: str
    sample: PolicySample
    temperature: float
    claimed: bool = False


_oracles: dict[tuple[str, str], tuple[asyncio.AbstractEventLoop, RoutingOracle]] = {}


def shared_oracle(config: ProprioceptionUserConfig) -> RoutingOracle:
    loop = asyncio.get_running_loop()
    key = (config.oracle_base_url, config.oracle_model)
    held = _oracles.get(key)
    if held is None or held[0] is not loop:
        held = (loop, RoutingOracle(config.oracle_base_url, config.oracle_model))
        _oracles[key] = held
    return held[1]


class ProprioceptionTask(vf.Task[ProprioceptionData, ProprioceptionState, ProprioceptionTaskConfig]):
    user = ProprioceptionUser

    @vf.stop
    async def conversation_over(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> bool:
        return trace.state.conversation_over

    def _context_matches(self, trace) -> bool | None:
        expected = trace.state.probe_token_ids
        branches = getattr(trace, "branches", None)
        if not expected or not branches:
            return None
        answered = list(branches[-1].token_ids)
        if len(answered) < len(expected):
            return None
        return answered[: len(expected)] == list(expected)

    @vf.reward
    async def expert_match(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        return await self.match_expert(trace)

    async def match_expert(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        correctness = await self._correctness(trace)
        if trace.state.lexical_label is not None and trace.state.lexical_label == trace.state.label:
            return correctness * cast(ProprioceptionTaskConfig, self.config).user.lexical_match_weight
        return correctness

    async def _correctness(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        if trace.state.oracle_failure is not None:
            return 0.0
        if self._context_matches(trace) is False:
            message = "the policy answered in a context that does not start with the sequence the oracle labeled"
            if cast(ProprioceptionTaskConfig, self.config).user.context_mismatch == "error":
                raise ValueError(message)
            logger.warning(message)
        reply = self._answer_reply(trace)
        if reply is None:
            return 0.0
        if trace.state.open_format:
            if trace.state.probe_type not in EXPERT_ID_TYPES:
                return self._graded_answer(trace.state, reply)
            if trace.state.true_expert is None:
                return 0.0
            return float(self._answered(trace.state, reply) == trace.state.true_expert)
        if trace.state.correct_letter is None:
            return 0.0
        return float(parse_letter_answer(reply) == trace.state.correct_letter)

    def _answer_distance(self, state: ProprioceptionState, reply: str) -> int | None:
        user = cast(ProprioceptionTaskConfig, self.config).user
        if state.label is None:
            return None
        markers = self._reasoning_markers
        if state.probe_type in SINGLE_TOKEN_TYPES:
            prefixes = CONV_ANSWER_PREFIXES[state.probe_type]
            if parse_bounded_token(reply, max_chars=user.answer_max_chars, prefixes=prefixes, markers=markers) is None:
                return None
            return int(
                not token_reply_matches(
                    reply,
                    state.label_text or "",
                    names_none=state.label == NONE_PARTNER,
                    max_chars=user.answer_max_chars,
                    prefixes=prefixes,
                    markers=markers,
                )
            )
        if state.answer_bound is None:
            return None
        if state.probe_type in ENTROPY_TYPES:
            answer = parse_bounded_decimal(reply, state.answer_bound, max_chars=user.answer_max_chars, markers=markers)
        else:
            answer = parse_bounded_integer(
                reply,
                state.answer_bound,
                max_chars=user.answer_max_chars,
                layer_is_answer=state.probe_type == "decision_depth",
                markers=markers,
            )
        return None if answer is None else abs(answer - state.label)

    def _graded_answer(self, state: ProprioceptionState, reply: str) -> float:
        distance = self._answer_distance(state, reply)
        if distance is None:
            return 0.0
        tolerance = cast(ProprioceptionTaskConfig, self.config).user.scalar_tolerance.get(state.probe_type, 0.0)
        if state.probe_type not in SCALAR_TYPES or tolerance <= 0:
            return float(distance == 0)
        return max(0.0, 1.0 - distance / tolerance)

    @property
    def _reasoning_markers(self) -> ReasoningMarkers:
        return served_model(cast(ProprioceptionTaskConfig, self.config).user.oracle_model).reasoning

    def _answered(self, state: ProprioceptionState, reply: str) -> int | None:
        user = cast(ProprioceptionTaskConfig, self.config).user
        num_experts = state.answer_bound
        if num_experts is None:
            raise ValueError("an expert-id probe carries no answer bound")
        if user.answer_format == "strict":
            return parse_bare_expert_id(reply, num_experts, self._reasoning_markers)
        if user.answer_format == "bounded":
            return parse_bounded_expert_id(
                reply, num_experts, max_chars=user.answer_max_chars, markers=self._reasoning_markers
            )
        return parse_expert_id(reply, num_experts)

    def _answer_node(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> vf.MessageNode | None:
        question = trace.state.question
        if question is None:
            return None
        asked = {
            i for i, node in enumerate(trace.nodes) if node.message.role == "user" and node.message.content == question
        }
        return next((node for node in reversed(trace.nodes) if node.sampled and node.parent in asked), None)

    def _answer_reply(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> str | None:
        node = self._answer_node(trace)
        if node is None:
            return None
        return content_text(node.message.content).strip()

    def _answer_tokens(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> int | None:
        node = self._answer_node(trace)
        if node is None or not node.token_ids:
            return None
        return sum(node.mask)

    @vf.reward
    async def answer_length_penalty(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        user = cast(ProprioceptionTaskConfig, self.config).user
        if user.answer_length_penalty <= 0:
            return 0.0
        num_tokens = self._answer_tokens(trace)
        if num_tokens is None:
            return 0.0
        span = max(user.answer_max_tokens - user.answer_expected_tokens, 1)
        overlong = min(max(num_tokens - user.answer_expected_tokens, 0) / span, 1.0)
        return -user.answer_length_penalty * overlong

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        reply = self._answer_reply(trace)
        if reply is None:
            return 0.0
        if trace.state.open_format:
            if trace.state.probe_type not in EXPERT_ID_TYPES:
                return float(self._answer_distance(trace.state, reply) is not None)
            return float(self._answered(trace.state, reply) is not None)
        return float(parse_letter_answer(reply) is not None)

    @vf.metric
    async def answer_exact(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        reply = self._answer_reply(trace)
        if reply is None or trace.state.oracle_failure is not None or not trace.state.open_format:
            return 0.0
        if trace.state.probe_type in EXPERT_ID_TYPES:
            return float(
                trace.state.true_expert is not None and self._answered(trace.state, reply) == trace.state.true_expert
            )
        return float(self._answer_distance(trace.state, reply) == 0)

    @vf.metric
    async def oracle_ok(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        return float(trace.state.oracle_failure is None)

    @vf.metric
    async def context_matches(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        return float(self._context_matches(trace) is not False)

    @vf.metric
    async def lexical_match(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState]) -> float:
        return float(trace.state.lexical_label is not None and trace.state.lexical_label == trace.state.label)


class SharedCarrierTask(ProprioceptionTask):
    user = None
    _lock: asyncio.Lock | None = None
    _trunk: CarrierTrunk | Exception | None = None
    _num_questions: int = 0

    async def setup(self, trace: vf.Trace[ProprioceptionData, ProprioceptionState], runtime) -> None:
        config = cast(ProprioceptionTaskConfig, self.config)
        trunk = await self._shared_trunk()
        if trunk.probe is not None:
            probe = trunk.probe
        else:
            ordinal = self._num_questions
            self._num_questions += 1
            probe = await compose_conversation_probe(
                shared_oracle(config.user),
                config.user,
                seed=f"{trunk.seed}:{ordinal}",
                system_prompt=self.data.system_prompt,
                turn1_prompt=self.carrier_prompt,
                visible_reply=trunk.visible_reply,
            )
            self._check_prefix(probe)
        record_probe(trace.state, probe)
        trace.state.question_asked = True
        if config.carrier_sample and not trunk.claimed:
            trunk.claimed = True
            trace.info["carrier_sample"] = {
                "prompt_token_ids": trunk.sample.prompt_token_ids,
                "token_ids": trunk.sample.token_ids,
                "logprobs": trunk.sample.logprobs,
                "temperature": trunk.temperature,
            }
        conversation = [
            vf.UserMessage(content=self.carrier_prompt),
            vf.AssistantMessage(content=trunk.visible_reply),
            vf.UserMessage(content=probe.question),
        ]
        trace.task = vf.TraceTask(type=trace.task.type, data=self.data.model_copy(update={"prompt": conversation}))

    @property
    def carrier_prompt(self) -> str:
        return cast(str, self.data.prompt)

    async def _shared_trunk(self) -> CarrierTrunk:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._trunk is None:
                try:
                    self._trunk = await self._build_trunk()
                except Exception as error:
                    self._trunk = error
        if isinstance(self._trunk, Exception):
            raise self._trunk
        return self._trunk

    async def _build_trunk(self) -> CarrierTrunk:
        config = cast(ProprioceptionTaskConfig, self.config)
        oracle = shared_oracle(config.user)
        messages = []
        if self.data.system_prompt:
            messages.append({"role": "system", "content": self.data.system_prompt})
        messages.append({"role": "user", "content": self.carrier_prompt})
        sample = await oracle.sample(
            messages, max_tokens=config.carrier_max_tokens, temperature=config.carrier_temperature
        )
        reply = sample.text
        if sample.finish_reason == "length":
            logger.warning(
                "carrier dropped: length (task %d, carrier_max_tokens=%d)", self.data.idx, config.carrier_max_tokens
            )
            raise ValueError(
                f"carrier reply hit carrier_max_tokens={config.carrier_max_tokens}; on_carrier_truncation=drop drops the group"
            )
        visible_reply = visible_assistant_content(reply, served_model(oracle.model).reasoning)
        if not visible_reply:
            logger.warning("carrier dropped: no visible content (task %d)", self.data.idx)
            raise ValueError("carrier reply has no visible content outside the reasoning block; the group is dropped")
        seed = f"{self.data.idx}:{reply}"
        probe = None
        if config.shared_probe:
            probe = await compose_conversation_probe(
                oracle,
                config.user,
                seed=seed,
                system_prompt=self.data.system_prompt,
                turn1_prompt=self.carrier_prompt,
                visible_reply=visible_reply,
            )
            self._check_prefix(probe)
        return CarrierTrunk(
            visible_reply=visible_reply,
            probe=probe,
            seed=seed,
            sample=sample,
            temperature=config.carrier_temperature,
        )

    def _check_prefix(self, probe: ConversationProbe) -> None:
        limit = cast(ProprioceptionTaskConfig, self.config).max_prefix_tokens
        if len(probe.probe_token_ids) > limit:
            logger.warning(
                "carrier dropped: prefix %d tokens > max_prefix_tokens=%d (task %d)",
                len(probe.probe_token_ids),
                limit,
                self.data.idx,
            )
            raise ValueError(
                f"rendered prefix of {len(probe.probe_token_ids)} tokens exceeds max_prefix_tokens={limit}"
            )

    @vf.group_reward
    async def expert_match(self, traces: list[vf.Trace[ProprioceptionData, ProprioceptionState]]) -> list[float]:
        try:
            return [await self.match_expert(trace) for trace in traces]
        finally:
            self._trunk = None


class ProprioceptionConfig(vf.TasksetConfig):
    num_tasks: int = 200
    seed: int = 0
    system_prompt: str = ""
    split: Literal["train", "heldout"] = "train"
    num_heldout_prompts: int = BANK_EVAL_PROMPTS
    num_train_prompts: int | None = None
    probe_mode: Literal["random_token", "conversation_token", "chat_only"] = "conversation_token"
    prompt_bank_path: str = ""
    endless: bool = False
    task: ProprioceptionTaskConfig = ProprioceptionTaskConfig()


class ProprioceptionTaskset(vf.Taskset[ProprioceptionTask, ProprioceptionConfig]):
    def __init__(self, config: ProprioceptionConfig) -> None:
        super().__init__(config)
        if config.task.shared_carrier and not config.endless:
            raise ValueError("task.shared_carrier needs endless = true so that a task instance serves one group only")
        user = config.task.user
        if config.probe_mode != "chat_only":
            routing_layers(user, served_model(user.oracle_model))
        if config.probe_mode == "random_token" and user.probed_layer is None:
            raise ValueError("random_token probes one layer and needs probed_layer set")
        if config.endless:
            self.__class__ = EndlessProprioceptionTaskset

    def bank(self) -> list[str]:
        if not self.config.prompt_bank_path:
            raise ValueError("a taskset needs prompt_bank_path: the carrier prompts it draws turn 1 from")
        train, heldout = load_prompts(self.config.prompt_bank_path, self.config.num_heldout_prompts)
        if self.config.split == "train":
            return train if self.config.num_train_prompts is None else train[: self.config.num_train_prompts]
        return heldout

    def task(self, idx: int, prompt: str) -> ProprioceptionTask:
        system_prompt = self.config.system_prompt or None
        if self.config.probe_mode == "random_token":
            data = ProprioceptionData(
                idx=idx, prompt=None, sentence=prompt, probe_mode=self.config.probe_mode, system_prompt=system_prompt
            )
            return ProprioceptionTask(data, self.config.task)
        data = ProprioceptionData(
            idx=idx, prompt=prompt, probe_mode=self.config.probe_mode, system_prompt=system_prompt
        )
        task_cls = SharedCarrierTask if self.config.task.shared_carrier else ProprioceptionTask
        return task_cls(data, self.config.task)

    def stream(self):
        rng = random.Random(self.config.seed)
        bank = self.bank()
        idx = 0
        while True:
            if idx % len(bank) == 0:
                rng.shuffle(bank)
            yield self.task(idx, bank[idx % len(bank)])
            idx += 1

    def load(self) -> list[ProprioceptionTask]:
        return list(itertools.islice(self.stream(), self.config.num_tasks))


class EndlessProprioceptionTaskset(ProprioceptionTaskset):
    INFINITE = True

    def load(self):
        return self.stream()


if __name__ == "__main__":
    ProprioceptionUser.run()
