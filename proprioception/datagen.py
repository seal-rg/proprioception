import json
import random
import time
from pathlib import Path

from proprioception.bank import BANK_EVAL_PROMPTS, load_prompts
from proprioception.families import ModelFamily
from proprioception.formats import (
    ENTROPY_TYPES,
    EXPERT_ID_TYPES,
    INT_ENTROPY_TYPES,
    ROUTING_TYPES,
    LENS_LAYER_TYPES,
    LENS_TYPES,
    PERCENT_TYPES,
    RANKED_TYPES,
    domain_of,
    draw_lens_layer,
    resolve_types,
    TOP_TOKEN_MAX_RANK,
    LENSES,
    SINGLE_TOKEN_TYPES,
)
from proprioception.convassembly import template_close_text
from proprioception.rendering import ChatRenderer, check_server_parity, segment_of_token, tokens_within
from proprioception.questions import (
    ANSWER_PREFIX_BANKS,
    ANSWER_PREFIXES,
    CONV_QUESTION_BANKS,
    DEPTH_PREFIXES,
    ENTROPY_PREFIXES,
    EXPERT_RANK_REFS,
    INT_ENTROPY_PREFIXES,
    PARTNER_PREFIXES,
    PERCENT_PREFIXES,
    PROBE_CONTEXTS,
    RANK_ANSWER_PREFIXES,
    RANK_REFS,
    SHARE_PREFIXES,
    format_token_ref,
    lens_templates_for_layer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/sft_proprioception"


def build(
    num_train: int,
    num_eval: int,
    base_url: str,
    seed: int,
    probed_layer: int,
    data_dir: Path,
    family: ModelFamily,
    concurrency: int = 1,
    domain: str = "routing",
    lens: str = "logit",
    types: list[str] | None = None,
    bank_eval_prompts: int = BANK_EVAL_PROMPTS,
    lens_token_prob_depth: float | None = None,
    answer_opening: str = "template",
) -> None:
    types = resolve_types(domain, types or [])
    lens_types = [t for t in types if t in LENS_TYPES]
    expert_id_types = [t for t in types if t in EXPERT_ID_TYPES]
    numeric_types = [t for t in types if t not in EXPERT_ID_TYPES]
    if lens_types and lens not in LENSES:
        raise ValueError(f"unknown lens {lens!r}; available: {list(LENSES)}")
    if "routing_rank" in types and not 2 <= family.num_experts_per_tok <= max(EXPERT_RANK_REFS):
        raise ValueError(
            f"routing_rank needs a top-k between 2 and {max(EXPERT_RANK_REFS)}; {family.name} uses {family.num_experts_per_tok}"
        )
    lens_layers = list(range(family.num_hidden_layers - 1))
    if probed_layer >= 0:
        if any(t in ROUTING_TYPES for t in types) and probed_layer not in family.moe_layers:
            raise ValueError(f"layer {probed_layer} is not an MoE layer; valid: {family.moe_layers}")
        if any(t in LENS_LAYER_TYPES for t in types) and probed_layer not in lens_layers:
            raise ValueError(f"layer {probed_layer} has no lens readout; valid: {lens_layers}")
    renderer = ChatRenderer(family.checkpoint)
    if base_url:
        check_server_parity(renderer, base_url, family.checkpoint)
    if answer_opening not in ("template", "generation"):
        raise ValueError(f"unknown answer opening {answer_opening!r}; 'template' or 'generation'")
    closing_ids: list[int] = []
    if answer_opening == "generation":
        opening = renderer.decode(renderer.generation_prompt)
        if family.reasoning.open in opening and family.reasoning.close not in opening:
            nothink = ChatRenderer.for_thinking(family.checkpoint, "nothink", family.thinking)
            closing_ids = renderer.encode_content_as_template(
                template_close_text(opening, nothink.decode(nothink.generation_prompt), family.reasoning)
            )
    train_sentences, eval_sentences = load_prompts(held_out=bank_eval_prompts)

    tokenize_trimmed = renderer.encode_content

    base_ids = tokenize_trimmed("x")
    probe_ids = tokenize_trimmed("x 0")
    extra = probe_ids[len(base_ids) :]
    if probe_ids[: len(base_ids)] != base_ids or len(extra) not in (1, 2):
        raise ValueError(f"tokenizer renders ' 0' as {len(extra)} tokens after its context")
    if len(extra) == 1:
        prefix_suffix, reference = "", base_ids
    else:
        prefix_suffix, reference = " ", base_ids + extra[:1]
    answer_space_ids = [] if prefix_suffix == "" else [reference[-1]]
    answer_token = {}
    multi_digit = bool(numeric_types)
    if not multi_digit:
        for expert in range(family.num_experts):
            ids = tokenize_trimmed(f"x {expert}")
            if len(ids) != len(reference) + 1 or ids[: len(reference)] != reference:
                multi_digit = True
                break
            answer_token[expert] = ids[-1]

    digit_token = {}
    dot_token = None
    value_token = {}
    if multi_digit:
        answer_token = {}
        for digit in range(10):
            ids = tokenize_trimmed(f"x 0.{digit}")
            tail = ids[len(reference) :]
            if len(tail) != 3 or ids[: len(reference)] != reference:
                raise ValueError(
                    f"'0.{digit}' does not tokenize as number, dot, digit after {reference!r} (got {tail})"
                )
            digit_token[digit] = tail[-1]
        if not numeric_types:
            for expert in (0, family.num_experts - 1):
                ids = tokenize_trimmed(f"x {expert}")
                if ids[len(reference) :] != [digit_token[int(d)] for d in str(expert)]:
                    raise ValueError(f"expert id {expert} does not tokenize as its digits")
            print(f"multi-digit answer encoding: ids span up to {len(str(family.num_experts - 1))} digit tokens")
            leading_zero = digit_token[0]
            for answer_prefix in ANSWER_PREFIXES + (RANK_ANSWER_PREFIXES if "routing_rank" in types else []):
                bare = tokenize_trimmed(f"{answer_prefix}{prefix_suffix}")
                extended = tokenize_trimmed(f"{answer_prefix} 0")
                if extended != bare + [leading_zero] or bare[len(bare) - len(answer_space_ids) :] != answer_space_ids:
                    raise ValueError(f"prefix {answer_prefix!r} does not leave a number as the next token")
        else:
            ids = tokenize_trimmed("x 3.4")
            tail = ids[len(reference) :]
            if len(tail) != 3 or tail[2] != digit_token[4]:
                raise ValueError(f"'3.4' does not tokenize as number, dot, digit (got {tail})")
            dot_token = tail[1]
            leading_three = tail[0]
            ids = tokenize_trimmed("x 34")
            tail = ids[len(reference) :]
            if tail == [digit_token[3], digit_token[4]]:
                if leading_three != digit_token[3]:
                    raise ValueError("digit runs must start with the same digit tokens they continue with")
                for value in ("58", "99", "10"):
                    ids = tokenize_trimmed(f"x {value}")
                    if ids[len(reference) :] != [digit_token[int(d)] for d in value]:
                        raise ValueError(f"two-digit value {value!r} does not tokenize as its digits")
                ids = tokenize_trimmed("x 12.7")
                if ids[len(reference) :] != [digit_token[1], digit_token[2], dot_token, digit_token[7]]:
                    raise ValueError(
                        f"'12.7' does not tokenize as digit, digit, dot, digit (got {ids[len(reference) :]})"
                    )
            elif len(tail) == 1:
                for value in range(100):
                    ids = tokenize_trimmed(f"x {value}")
                    if len(ids) != len(reference) + 1 or ids[: len(reference)] != reference:
                        raise ValueError(f"whole number {value} is not a single token")
                    value_token[value] = ids[-1]
                if leading_three != value_token[3]:
                    raise ValueError("a decimal's whole part must be the value token of that number")
                ids = tokenize_trimmed("x 12.7")
                if ids[len(reference) :] != [value_token[12], dot_token, digit_token[7]]:
                    raise ValueError(f"'12.7' does not tokenize as value, dot, digit (got {ids[len(reference) :]})")
                print("fused number encoding: whole numbers are one token each; only decimals span several")
                if expert_id_types:
                    for expert in range(family.num_experts):
                        ids = tokenize_trimmed(f"x {expert}")
                        if len(ids) != len(reference) + 1 or ids[: len(reference)] != reference:
                            raise ValueError(f"expert id {expert} is not a single token")
                        answer_token[expert] = ids[-1]
            else:
                raise ValueError(f"'34' tokenizes as neither its digits nor one token (got {tail})")
            largest_values = []
            if lens_types:
                largest_values.append(("layer index", family.num_hidden_layers - 1))
            if expert_id_types and not value_token:
                largest_values.append(("expert id", family.num_experts - 1))
            for name, largest in largest_values:
                if value_token:
                    if largest >= len(value_token):
                        raise ValueError(
                            f"{name}s run to {largest}, past the {len(value_token)}-entry fused value table"
                        )
                else:
                    for value in (0, largest):
                        ids = tokenize_trimmed(f"x {value}")
                        if ids[len(reference) :] != [digit_token[int(d)] for d in str(value)]:
                            raise ValueError(f"{name} {value} does not tokenize as its digits")
            leading_zero = value_token[0] if value_token else digit_token[0]
            numeric_prefixes: list[str] = []
            if any(t in PERCENT_TYPES for t in types):
                numeric_prefixes += PERCENT_PREFIXES
            if any(t in ENTROPY_TYPES for t in types):
                numeric_prefixes += ENTROPY_PREFIXES
            if any(t in INT_ENTROPY_TYPES for t in types):
                numeric_prefixes += INT_ENTROPY_PREFIXES
            if "decision_depth" in types:
                numeric_prefixes += DEPTH_PREFIXES
            if "routing" in types:
                numeric_prefixes += ANSWER_PREFIXES
            if "routing_rank" in types:
                numeric_prefixes += RANK_ANSWER_PREFIXES
            if "routing_share" in types:
                numeric_prefixes += SHARE_PREFIXES
            for answer_prefix in numeric_prefixes:
                bare = tokenize_trimmed(f"{answer_prefix}{prefix_suffix}")
                extended = tokenize_trimmed(f"{answer_prefix} 0")
                if extended != bare + [leading_zero] or bare[len(bare) - len(answer_space_ids) :] != answer_space_ids:
                    raise ValueError(f"prefix {answer_prefix!r} does not leave a number as the next token")
    else:
        for answer_prefix in ANSWER_PREFIXES + (RANK_ANSWER_PREFIXES if "routing_rank" in types else []):
            bare = tokenize_trimmed(f"{answer_prefix}{prefix_suffix}")
            extended = tokenize_trimmed(f"{answer_prefix} 0")
            if extended != bare + [answer_token[0]]:
                raise ValueError(f"prefix {answer_prefix!r} does not yield a single-token expert id")
    none_token = None
    if "routing_partner" in types:
        for answer_prefix in PARTNER_PREFIXES:
            bare = tokenize_trimmed(answer_prefix)
            extended = tokenize_trimmed(f"{answer_prefix} none")
            if len(extended) != len(bare) + 1 or extended[: len(bare)] != bare:
                raise ValueError(f"prefix {answer_prefix!r} does not leave ' none' as a single next token")
            if none_token is None:
                none_token = extended[-1]
            elif extended[-1] != none_token:
                raise ValueError(f"' none' tokenizes differently after {answer_prefix!r}")

    quote_token = None
    token_types = [t for t in types if t in SINGLE_TOKEN_TYPES]
    if token_types:
        closing = tokenize_trimmed('"')
        if len(closing) != 1:
            raise ValueError(f"a double quote tokenizes as {len(closing)} tokens")
        quote_token = closing[0]
        for answer_prefix in sorted({p for t in token_types for p in ANSWER_PREFIX_BANKS[t]}):
            bare = tokenize_trimmed(answer_prefix)
            quoted = tokenize_trimmed(f'{answer_prefix} "')
            if len(quoted) != len(bare) + 1 or quoted[: len(bare)] != bare:
                raise ValueError(f"prefix {answer_prefix!r} does not take its opening quote as one more token")

    banks = {t: (questions, ANSWER_PREFIX_BANKS[t]) for t, questions in CONV_QUESTION_BANKS.items()}
    types_by_domain: dict[str, list[str]] = {}
    for t in types:
        types_by_domain.setdefault(domain_of(t), []).append(t)
    domains_present = list(types_by_domain)

    def make_one(draw: random.Random, sentences: list[str]) -> dict | None:
        sentence = draw.choice(sentences)
        context_template = draw.choice(PROBE_CONTEXTS)
        layer = -1
        rank = 1
        pc_type = draw.choice(types_by_domain[draw.choice(domains_present)])
        if pc_type in ("top_token", "lens_top_token"):
            rank = draw.randint(1, TOP_TOKEN_MAX_RANK)
        elif pc_type == "routing_rank":
            rank = draw.randint(2, family.num_experts_per_tok)
        questions, prefixes = banks[pc_type]
        if pc_type in ROUTING_TYPES:
            layer = probed_layer if probed_layer >= 0 else draw.choice(family.moe_layers)
        elif pc_type in LENS_LAYER_TYPES:
            layer = (
                probed_layer
                if probed_layer >= 0
                else draw_lens_layer(draw, pc_type, len(lens_layers), lens_token_prob_depth)
            )
        if pc_type in LENS_LAYER_TYPES:
            questions = lens_templates_for_layer(questions, layer, family.num_hidden_layers)
        question_template = draw.choice(questions)
        answer_prefix = draw.choice(prefixes)
        context = context_template.format(sentence=sentence)
        lead = context_template.split("{sentence}")[0]
        sentence_chars = (len(lead), len(lead) + len(sentence))
        context_ids, offsets = renderer.encode_content_with_offsets(context)
        inside = tokens_within(offsets, *sentence_chars)
        probeable = inside if pc_type in ROUTING_TYPES else inside[:-1]
        candidates = [i for i in probeable if renderer.text(context_ids[i]).strip()]
        if not candidates:
            return None
        probe = draw.choice(candidates)
        same = [i for i in inside if context_ids[i] == context_ids[probe]]
        token_ref = format_token_ref(renderer.text(context_ids[probe]), same.index(probe) + 1, len(same))
        rank_ref = EXPERT_RANK_REFS[rank] if pc_type == "routing_rank" else RANK_REFS[rank]
        question = question_template.format(token_ref=token_ref, rank_ref=rank_ref, layer=layer)
        reply = f'{answer_prefix} "' if pc_type in SINGLE_TOKEN_TYPES else f"{answer_prefix}{prefix_suffix}"
        full = f"{context}\n\n{question}"
        full_ids, full_offsets = renderer.encode_content_with_offsets(full)
        keep = probe + 1 if pc_type in ROUTING_TYPES else probe + 2
        if full_ids[:keep] != context_ids[:keep]:
            return None
        spaced = bool(prefix_suffix) and reply.endswith(prefix_suffix)
        if answer_opening == "generation":
            rendered = renderer.render([{"role": "user", "content": full}], add_generation_prompt=True)
            reasoning_start = len(rendered.token_ids)
            reply_start = reasoning_start + len(closing_ids)
            input_ids = rendered.token_ids + closing_ids + renderer.encode_content(reply)
        else:
            rendered = renderer.render([{"role": "user", "content": full}, {"role": "assistant", "content": reply}])
            reply_start, _ = rendered.content_span(1)
            opening_texts = renderer.texts(rendered.token_ids[rendered.content_span(0)[1] : reply_start])
            marker_at = next(
                (i for i, t in enumerate(opening_texts) if family.reasoning.open in t or family.reasoning.close in t),
                None,
            )
            reasoning_start = reply_start if marker_at is None else rendered.content_span(0)[1] + marker_at
            input_ids = rendered.until_content_end(1)
            if spaced and not renderer.content_as_template("assistant", reply).endswith(prefix_suffix):
                input_ids = input_ids + answer_space_ids
        user_start, user_end = rendered.content_span(0)
        if rendered.token_ids[user_start : user_start + len(full_ids)] != full_ids:
            raise ValueError("the rendered user turn does not carry the content ids it was rendered from")
        inside_full = tokens_within(full_offsets, *sentence_chars)
        first_question = next(i for i, part in enumerate(segment_of_token(full_offsets, [len(context)])) if part)
        spans = {
            "user_surface_formatting": [[0, inside_full[0]], [inside_full[-1] + 1, first_question]],
            "user_surface_sentence": [[inside_full[0], inside_full[-1] + 1]],
            "user_pc_question": [[first_question, len(full_ids)]],
        }
        segments = {
            name: [[user_start + lo, user_start + hi] for lo, hi in ranges if hi > lo] for name, ranges in spans.items()
        }
        segments["assistant_pc_formatting"] = [[user_end, reasoning_start]]
        if reasoning_start < reply_start:
            segments["assistant_pc_reasoning"] = [[reasoning_start, reply_start]]
        if spaced and input_ids[-1:] != answer_space_ids:
            raise ValueError(f"the rendered reply {renderer.texts(input_ids[reply_start:])} lost its answer space")
        segments["assistant_pc_response"] = [[reply_start, len(input_ids)]]
        sample = {
            "input_ids": input_ids,
            "probe_pos": user_start + probe,
            "segments": segments,
        }
        if pc_type != "routing":
            sample["type"] = pc_type
        if pc_type in RANKED_TYPES:
            sample["rank"] = rank
        if layer >= 0:
            sample["layer"] = layer
        return sample

    def make_samples(sentences: list[str], count: int, label: str) -> list[dict]:
        started = time.time()
        done = 0

        def worker(index: int, share: int) -> list[dict]:
            nonlocal done
            draw = random.Random(f"{seed}:{label}:{index}")
            built: list[dict] = []
            attempts = 0
            while len(built) < share and attempts < share * 3:
                attempts += 1
                sample = make_one(draw, sentences)
                if sample is not None:
                    built.append(sample)
                    done += 1
                    if done % 20000 == 0:
                        rate = done / max(time.time() - started, 1e-9)
                        print(f"  {label} {done}/{count} | {rate:.0f} samples/s | {(count - done) / rate:.0f}s left")
            return built

        shares = [count // concurrency + (1 if i < count % concurrency else 0) for i in range(concurrency)]
        chunks = [worker(i, share) for i, share in enumerate(shares) if share]
        return [sample for chunk in chunks for sample in chunk]

    data_dir.mkdir(parents=True, exist_ok=True)
    train = make_samples(train_sentences, num_train, "train")
    evals = make_samples(eval_sentences, num_eval, "eval")
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "answer_token": answer_token,
                "digit_token": digit_token,
                "prefix_suffix": prefix_suffix,
                "dot_token": dot_token,
                "value_token": value_token,
                "none_token": none_token,
                "quote_token": quote_token,
                "domain": domain,
                "types": types,
                "lens": lens if lens_types else None,
            }
        )
    )
    (data_dir / "train.json").write_text(json.dumps(train))
    (data_dir / "eval.json").write_text(json.dumps(evals))
    print(f"built {len(train)} train / {len(evals)} eval samples -> {data_dir}")
