import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

import httpx
import torch
from safetensors.torch import save_file

from proprioception.batching import split_round
from proprioception import batching, checkpoint, distributed, export, primerl_backend
from proprioception.convassembly import (
    ConversationAssembler,
    generate_completions,
    reasoning_close_text,
    template_close_text,
)
from proprioception.bank import BANK, load_prompts
from proprioception.evaluation import (
    PairLabelCounts,
    collect_labels,
    intro_metrics,
    probe_pair,
    run_eval,
    tolerance_summary,
    type_summary,
    with_label_answers,
    write_eval_record,
)
from proprioception.families import resolve_family
from proprioception.batching import ANSWER_PATH_SEGMENT, COPY_SEGMENT, TURN_END_SEGMENT
from proprioception.formats import (
    SINGLE_TOKEN_TYPES,
    anchor_answer_widths,
    answer_encoding,
    check_probed_layers,
    check_sample_types,
    select_sample_types,
    with_turn_end,
)
from proprioception.losses import (
    anchor_kl_at,
    batch_losses,
    load_entropy_bits,
    marker_positions,
    model_forward,
    reference_anchor_lookup,
    sample_anchor_positions,
    segment_positions,
    segment_sample_losses,
    subset_lookup,
)
from proprioception.distributed import report_memory
from proprioception.oracle import reopens_reasoning
from proprioception.primerl_backend import decoder_layers
from proprioception.rendering import ChatRenderer
from proprioception.questions import PROPRIOCEPTION_SYSTEM_PROMPT

EVAL_ASSISTANT_FRACTION = 0.5
EVAL_SYSTEM_PROMPT_FRACTION = 0.5


def conv_role_metrics(
    role: str,
    accuracy: float,
    said: float,
    ctx,
    count: int,
    tolerance: dict,
    *,
    num_parsed: int,
    num_scored: int,
    accuracy_on_valid: float,
) -> dict:
    metrics = {
        "acc": accuracy,
        "acc_on_valid": accuracy_on_valid,
        "said": said,
        "ctx_acc": ctx[0],
        "ctx_n": ctx[1],
        "n": count,
        "n_parsed": num_parsed,
        "n_scored": num_scored,
    }
    if tolerance:
        metrics["tolerance"] = tolerance
        if len(tolerance) == 1:
            stats = next(iter(tolerance.values()))
            metrics.update(mae=stats["mae"], within_5=stats["within_5"], within_10=stats["within_10"])
    return {f"conv_{role}_{name}": value for name, value in metrics.items()}


def on_valid_accuracy(type_stats: dict) -> float:
    hits = sum(s["acc_on_valid"] * s["n_on_valid"] for s in type_stats.values() if s["n_on_valid"])
    total = sum(s["n_on_valid"] for s in type_stats.values())
    return hits / total if total else float("nan")


def sync_weights_to_server(model, sync_dir: Path, hf_config_dir: Path, base_url: str, round_idx: int) -> float:
    target = sync_dir / "policy"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.jinja", "*.txt"):
        for config_file in hf_config_dir.glob(pattern):
            if config_file.name != "model.safetensors.index.json" and not (target / config_file.name).exists():
                shutil.copy(config_file, target / config_file.name)
    (target / "model.safetensors.index.json").unlink(missing_ok=True)
    start = time.time()
    state = {k: v for k, v in distributed.full_state_dict(model).items() if k not in ("rope_cos", "rope_sin")}
    if not distributed.is_main():
        return time.time() - start
    state = {k.removeprefix("model.") if k.startswith("model.model.") else k: v for k, v in state.items()}
    state = {k: v.detach().to("cpu") for k, v in state.items()}
    state = primerl_backend.to_hub_layout(model, hf_config_dir, state)
    save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})
    response = httpx.post(f"{base_url.rstrip('/')}/update_weights", json={"weight_dir": str(target)}, timeout=600)
    response.raise_for_status()
    return time.time() - start


class NcclWeightPublisher:
    def __init__(self, model, base_url: str, port: int, device: str, *, num_receivers: int = 1, timeout: int = 600):
        from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcastSender

        if not isinstance(model, primerl_backend.RoutingModel):
            raise ValueError("the nccl transport publishes through prime-rl's converters, which need its backend")
        from prime_rl.trainer.rl.broadcast.nccl import disable_nccl_p2p_if_unavailable

        disable_nccl_p2p_if_unavailable()
        if distributed.is_distributed():
            torch.distributed.barrier()
        self.inner = model.publication_module
        self.base_url = base_url.rstrip("/")
        payload = {
            "host": "localhost",
            "port": port,
            "rank_offset": 0,
            "inference_world_size": num_receivers,
            "timeout": timeout,
        }
        request = self._post_in_thread("/init_broadcaster", payload)
        resolved = torch.device(device)
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.sender = NCCLWeightBroadcastSender("localhost", port, 0, num_receivers + 1, resolved, timeout)
        self._join(request)

    def _post(self, path: str, payload: dict | None) -> None:
        response = httpx.post(f"{self.base_url}{path}", json=payload, timeout=900)
        response.raise_for_status()

    def _post_in_thread(self, path: str, payload: dict | None) -> tuple[threading.Thread, list] | None:
        if not distributed.is_main():
            return None
        failures: list[BaseException] = []

        def run() -> None:
            try:
                self._post(path, payload)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        return thread, failures

    @staticmethod
    def _join(request: tuple[threading.Thread, list] | None) -> None:
        if request is None:
            return
        thread, failures = request
        thread.join()
        if failures:
            raise failures[0]

    def publish(self, round_idx: int) -> float:
        start = time.time()
        if distributed.is_main():
            self._post("/pause", None)
        request = self._post_in_thread("/update_weights", {"weight_dir": None})
        self.sender.broadcast_weights(self.inner, round_idx)
        self._join(request)
        if distributed.is_main():
            self._post("/resume", None)
        return time.time() - start


def window_mean(values: list[float], count: int) -> float:
    window = values[-count:] if count else []
    return sum(window) / len(window) if window else float("nan")


def trained_rows(readout: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (readout.to(labels.device) >= 0) & (labels >= 0)


def micro_batches(
    live: list,
    replay: list,
    batch_size: int,
    *,
    max_tokens: int = 0,
    world: int = 1,
    layout: str = "padded",
    completed: list | None = None,
) -> list[list]:
    streams = [
        batching.token_chunks(rows, batch_size, max_tokens, world, layout) for rows in (live, replay, completed or [])
    ]
    batches = []
    for index in range(max(len(chunks) for chunks in streams)):
        for chunks in streams:
            if index < len(chunks):
                batches.append(chunks[index])
    return batches


def train_on_policy_sft(args) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    rank, world = distributed.rank_and_world()
    if args.batch_size % world:
        raise ValueError(f"batch size {args.batch_size} does not divide across {world} ranks")
    if args.resume == "latest" and args.save_final and export.servable_exists(args.output):
        if distributed.is_main():
            print(f"{Path(args.output) / 'hf'} already exported; nothing to do")
        return
    family = resolve_family(args.model)
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    encoding = answer_encoding(meta, family, args.device)
    if args.lm_weight > 0 and TURN_END_SEGMENT in args.lm_segments:
        encoding = with_turn_end(encoding, family.checkpoint)
    answer_token_ids = encoding["answer_token_ids"]
    template_train = select_sample_types(json.loads((data_dir / "train.json").read_text()), args.types, "train")
    template_eval = select_sample_types(json.loads((data_dir / "eval.json").read_text()), args.types, "eval")
    check_probed_layers(template_train + template_eval, family)
    check_sample_types(template_train + template_eval, encoding)
    conv_types = sorted({sample.get("type", "routing") for sample in template_train})
    collate = batching.Collator(
        pad_id=1,
        device=device,
        layout=args.batch_layout or family.batch_layout,
        bucket=args.batch_bucket or family.batch_bucket,
    )

    if args.invariant_kernels:
        primerl_backend.use_invariant_kernels(True)
    model = family.load(
        args.init_checkpoint or family.checkpoint,
        device,
        activation_checkpointing=args.gradient_checkpointing,
        expert_parallel=args.expert_parallel,
        compile=args.compile,
        optimization_dtype=args.optimization_dtype,
    )
    if args.init_checkpoint and distributed.is_main():
        print(f"started from the checkpoint at {args.init_checkpoint}")
    if args.gradient_checkpointing:
        setattr(model, "gradient_checkpointing", True)
    model = distributed.shard_model(model, decoder_layers(model))
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=True,
    )
    ref_model = None
    if args.kl_weight > 0 or args.reply_kl_weight > 0:
        ref_model = family.load(
            args.anchor_checkpoint or family.checkpoint,
            device,
            activation_checkpointing=args.gradient_checkpointing,
            expert_parallel=args.expert_parallel,
            compile=args.compile,
            optimization_dtype="bfloat16",
        )
        ref_model = distributed.shard_model(ref_model, decoder_layers(ref_model))
        report_memory("reference load")
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    nothink_renderer = ChatRenderer.for_thinking(args.served_model, "nothink", family.thinking)
    nothink_opening = nothink_renderer.decode(nothink_renderer.generation_prompt)
    reasoning_close = reasoning_close_text(nothink_opening, family.reasoning)
    default_renderer = ChatRenderer(args.served_model)
    template_close = template_close_text(
        default_renderer.decode(default_renderer.generation_prompt), nothink_opening, family.reasoning
    )
    assembler = ConversationAssembler(
        args.server_url,
        args.served_model,
        args.prefix_suffix,
        multi_digit=encoding["digit_token_ids"] is not None and encoding.get("value_token_ids") is None,
        markers=family.reasoning,
        renderer=default_renderer,
        answer_instruction=args.answer_instruction,
        answer_parser=args.answer_parser,
        token_answer_bounded=args.token_answer_bounded,
        reasoning_close=reasoning_close,
        template_close=template_close,
        template_think=args.template_think,
        template_think_style=args.template_think_style,
    )
    answer_assemblers = {"default": assembler}
    if args.answer_nothink_fraction > 0 or args.eval_answer_nothink_fraction > 0:
        nothink_assembler = ConversationAssembler(
            args.server_url,
            args.served_model,
            args.prefix_suffix,
            multi_digit=assembler.multi_digit,
            markers=family.reasoning,
            check_server=False,
            renderer=nothink_renderer,
            answer_instruction=args.answer_instruction,
            answer_parser=args.answer_parser,
            token_answer_bounded=args.token_answer_bounded,
            reasoning_close=reasoning_close,
            template_close=template_close,
            template_think=args.template_think,
            template_think_style=args.template_think_style,
        )
        if nothink_assembler.seg_user_to_assistant == assembler.seg_user_to_assistant:
            raise ValueError(f"{args.model} opens its answer turns the same way with and without thinking")
        if nothink_assembler.seg_assistant_to_user != assembler.seg_assistant_to_user:
            raise ValueError(f"{args.model} renders the question turn differently in its no-think mode")
        answer_assemblers["nothink"] = nothink_assembler
    marker_ids = None
    if args.anchor_marker_positions:
        close = assembler.renderer.tokenizer.encode(family.reasoning.close, add_special_tokens=False)
        if len(close) != 1:
            raise ValueError(f"{args.model}'s reasoning close marker {family.reasoning.close!r} is not one token")
        marker_ids = (close[0], torch.tensor(sorted(assembler.end_token_ids)))
    seed_prompts, eval_seed_prompts = load_prompts(args.prompt_bank or BANK, args.bank_eval_prompts)
    sync_dir = Path(args.sync_dir)
    hf_config_dir = Path(args.served_model)
    publisher = None
    transport = args.weight_transport
    if transport == "auto":
        transport = "nccl" if family.backend == "prime" else "filesystem"
    if transport == "nccl":
        publisher = NcclWeightPublisher(
            model, args.server_url, args.nccl_port, device, num_receivers=max(1, args.server_gpus)
        )

    def publish_weights(round_idx: int) -> float:
        if publisher is not None:
            return publisher.publish(round_idx)
        return sync_weights_to_server(model, sync_dir, hf_config_dir, args.server_url, round_idx)

    pair_counts = PairLabelCounts(decay=args.lexical_lookup_decay, min_obs=args.lexical_lookup_min_count)
    log: dict[str, list] = {
        key: []
        for key in (
            "intro",
            "lm",
            "acc",
            "acc_conv",
            "acc_tmpl",
            "acc_completed",
            "conv_on_valid_hits",
            "conv_on_valid_n",
            "lex",
            "anchor",
            "path",
            "copy",
            "reply_kl",
            "said",
            "unclosed_said",
            "yield",
            "parse",
            "fallback",
            "templated",
            "think",
            "anstok",
            "reopen",
        )
    }
    checkpoint_dir = Path(args.output) / "checkpoints" if args.output else None
    start_round = 1
    resume_path: Path | None = Path(args.resume) if args.resume else None
    if args.resume == "latest":
        resume_path = checkpoint.latest_checkpoint(checkpoint_dir) if checkpoint_dir is not None else None
        if resume_path is None and distributed.is_main():
            print(f"no checkpoint under {checkpoint_dir}; starting fresh")
    if resume_path is not None:
        saved = checkpoint.load_checkpoint(resume_path, model, optimizer)
        start_round = saved["round"] + 1
        rng.setstate(saved["rng"])
        pair_counts.restore(saved["pair_counts"])
        log = saved["log"]
        log.setdefault("reopen", [])
        log.setdefault("fallback", [])
        log.setdefault("templated", [])
        log.setdefault("reply_kl", [])
        log.setdefault("path", [])
        log.setdefault("copy", [])
        log.setdefault("acc_completed", [])
        log.setdefault("unclosed_said", [])
        log.setdefault("conv_on_valid_hits", [])
        log.setdefault("conv_on_valid_n", [])
        if distributed.is_main():
            print(f"resumed from {resume_path}: {saved['round']} rounds done")
    elif args.lexical_match_weight < 1.0:
        seed_count = args.lexical_lookup_seed_samples
        seed_labels = collect_labels(
            model,
            template_train if seed_count <= 0 else template_train[:seed_count],
            answer_token_ids,
            collate,
            args.batch_size,
            encoding.get("num_layers"),
            encoding.get("num_experts_per_tok"),
            encoding.get("none_token"),
            max_tokens=args.micro_batch_tokens,
        )
        for sample, label in zip(template_train, seed_labels):
            pair_counts.update(probe_pair(sample), label)
        if distributed.is_main():
            print(f"seeded pair counts from {len(seed_labels)} template train labels")

    sync_seconds = publish_weights(0)
    if distributed.is_main():
        print(f"initial weight sync in {sync_seconds:.1f}s")

    generations_path = Path(args.output) / "generations.jsonl" if args.output else None

    def dump_generations(entries: list[dict]) -> None:
        if generations_path is None or not entries or not distributed.is_main():
            return
        generations_path.parent.mkdir(parents=True, exist_ok=True)
        with generations_path.open("a") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")

    def seed_block(round_idx: int) -> int:
        return args.seed * 1_000_000_000_000 + round_idx * 1_000_000

    def make_conv_questions(
        count: int,
        prompts: list[str],
        seeds: int,
        round_tag: str,
        *,
        assistant_fraction: float,
        system_prompt_fraction: float,
        answer_nothink_fraction: float,
        answer_max_tokens: int,
        fallback: str = "none",
        template_answer_fraction: float = 0.0,
    ) -> tuple[list[dict], float, list[dict]]:
        needed = -(-count // max(args.queries_per_conversation, 1))
        num_conversations = max(1, math.ceil(needed * args.carrier_overrequest))
        chosen = list(enumerate(rng.choice(prompts) for _ in range(num_conversations)))
        mine = distributed.shard_slice(chosen, rank, world)
        first = mine[0][0] if mine else 0
        completions = generate_completions(
            args.server_url,
            str(hf_config_dir),
            [assembler.generation_prompt_ids(prompt) for _, prompt in mine],
            temperature=args.gen_temperature,
            seed=seeds + first,
            max_tokens=args.gen_max_tokens,
        )
        replies = [assembler.full_reply_text(ids) for ids, _ in completions]
        gathered = distributed.all_gather_list([(i, p, r) for (i, p), r in zip(mine, replies)])
        dump_generations(
            [{"tag": round_tag, "type": "carrier", "conv": i, "prompt": p, "reply": r} for i, p, r in gathered]
        )
        reopen = sum(reopens_reasoning(r, family.reasoning) for _, _, r in gathered) / max(len(gathered), 1)
        replies_sampled = []
        if args.reply_kl_weight > 0:
            local = []
            for (_, prompt), (ids, _) in zip(mine[: args.reply_kl_samples], completions):
                prompt_ids = assembler.generation_prompt_ids(prompt)
                if ids:
                    local.append({"input_ids": prompt_ids + ids, "probe_pos": 0, "reply_start": len(prompt_ids)})
            replies_sampled = distributed.all_gather_list(local)[: args.reply_kl_samples]
        local_rng = random.Random(seeds + rank)
        prefixes = []
        templated = []
        for (conv_idx, prompt), (ids, _) in zip(mine, completions):
            visible = assembler.carrier_reply(ids)
            if not visible:
                continue
            system_prompt = PROPRIOCEPTION_SYSTEM_PROMPT if local_rng.random() < system_prompt_fraction else ""
            conversation = assembler.build_conversation(prompt, visible, system_prompt=system_prompt)
            template_answers = template_answer_fraction > 0 and local_rng.random() < template_answer_fraction
            for _ in range(args.queries_per_conversation):
                want_role = "assistant" if local_rng.random() < assistant_fraction else "user"
                nothink = "nothink" in answer_assemblers and local_rng.random() < answer_nothink_fraction
                pc_type = local_rng.choice(conv_types)
                mode = "template" if template_answers else "nothink" if nothink else "default"
                builder = assembler if template_answers else answer_assemblers[mode]
                probe = builder.build_probe(
                    conversation,
                    rng=local_rng,
                    want_role=want_role,
                    pc_type=pc_type,
                    moe_layers=family.moe_layers,
                    num_hidden_layers=family.num_hidden_layers,
                    num_experts_per_tok=family.num_experts_per_tok,
                    lens_token_prob_depth=args.lens_token_prob_depth,
                )
                if probe is None:
                    continue
                probe["conv"] = conv_idx
                probe["with_system"] = bool(system_prompt)
                probe["answer_mode"] = mode
                if template_answers:
                    templated.append(assembler.template_answer(probe, local_rng))
                else:
                    prefixes.append(probe)
        answers = []
        if prefixes:
            answers = generate_completions(
                args.server_url,
                str(hf_config_dir),
                [p["input_ids"] for p in prefixes],
                temperature=args.gen_temperature,
                seed=seeds + 100_000 + first * args.queries_per_conversation,
                max_tokens=answer_max_tokens,
            )
        probes = [
            answer_assemblers[probe["answer_mode"]].append_answer(
                probe,
                ids,
                num_experts=family.num_experts,
                num_hidden_layers=family.num_hidden_layers,
                terminated=terminated,
                fallback=fallback,
                rng=local_rng,
            )
            for probe, (ids, terminated) in zip(prefixes, answers)
        ] + templated
        return distributed.all_gather_list(probes)[:count], reopen, replies_sampled

    def said_accuracy(samples: list[dict]) -> float:
        said = [s for s in samples if s["model_answer"] is not None and "label" in s]
        return sum(names_label(s) for s in said) / len(said) if said else float("nan")

    def names_label(sample: dict) -> bool:
        pc_type = sample.get("type", "routing")
        if pc_type in SINGLE_TOKEN_TYPES:
            return assembler.names_label(pc_type, sample["model_answer"], sample["label"], encoding.get("none_token"))
        return sample["model_answer"] == sample["label"]

    def dump_answer_entries(samples: list[dict], round_tag: str) -> None:
        dump_generations(
            [
                {
                    "tag": round_tag,
                    "type": "answer",
                    "conv": s.get("conv"),
                    "with_system": s.get("with_system"),
                    "role": s["probe_role"],
                    "layer": s["layer"],
                    "question": s["question"],
                    "answer": s["answer_text"],
                    "model_answer": s["model_answer"],
                    "label": s.get("label"),
                    "think_ok": s["think_ok"],
                    "think_block": s["think_block"],
                    "answer_mode": s.get("answer_mode", "default"),
                    "fallback": s.get("fallback", ""),
                    "unclosed_answer": s.get("unclosed_answer"),
                }
                for s in samples
            ]
        )

    samples_per_round = args.batch_size * args.grad_accum
    replay_per_round = int(round(samples_per_round * args.sft_replay_fraction))
    live_per_round = samples_per_round - replay_per_round
    loadent = float("nan")
    micros_since_log = 0
    samples_since_log = 0
    conv_micros_since_log = 0
    tmpl_micros_since_log = 0
    completed_micros_since_log = 0
    on_valid_micros_since_log = 0
    start_time = time.time()
    for round_idx in range(start_round, args.rounds + 1):
        if args.lr_schedule == "cosine":
            if round_idx <= args.warmup_rounds:
                scale = round_idx / max(args.warmup_rounds, 1)
            else:
                progress = (round_idx - args.warmup_rounds) / max(args.rounds - args.warmup_rounds, 1)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale

        gen_start = time.time()
        wanted = live_per_round
        conv_questions, carrier_reopen, replies_sampled = make_conv_questions(
            wanted,
            seed_prompts,
            seed_block(round_idx),
            f"round-{round_idx}",
            assistant_fraction=args.assistant_fraction,
            system_prompt_fraction=args.system_prompt_fraction,
            answer_nothink_fraction=args.answer_nothink_fraction,
            answer_max_tokens=args.answer_max_tokens,
            fallback=args.unparsed_answer_fallback,
            template_answer_fraction=args.template_answer_fraction,
        )
        gen_seconds = time.time() - gen_start
        log["yield"].append(len(conv_questions) / max(wanted, 1))
        committed = [p.get("answer_pos") is not None and not p.get("fallback") for p in conv_questions]
        log["parse"].append(sum(committed) / max(len(conv_questions), 1))
        log["fallback"].append(sum(bool(p.get("fallback")) for p in conv_questions) / max(len(conv_questions), 1))
        log["templated"].append(
            sum(p.get("answer_mode") == "template" for p in conv_questions) / max(len(conv_questions), 1)
        )
        log["think"].append(sum(p["think_block"] for p in conv_questions) / max(len(conv_questions), 1))
        log["anstok"].append(sum(p["answer_tokens"] for p in conv_questions) / max(len(conv_questions), 1))
        log["reopen"].append(carrier_reopen)
        rng.shuffle(conv_questions)
        for position, probe in enumerate(conv_questions):
            probe["conv_row"] = position
        conv_labels = []
        replay_rows = rng.sample(template_train, replay_per_round) if replay_per_round else []
        completed_rows = [p for p in conv_questions if p.get("completed")]
        micros = micro_batches(
            [p for p in conv_questions if not p.get("completed")],
            replay_rows,
            args.batch_size,
            max_tokens=args.micro_batch_tokens,
            world=world,
            layout=collate.layout,
            completed=completed_rows,
        )
        round_rows = sum(len(whole) for whole in micros)

        optimizer.zero_grad(set_to_none=True)
        train_start = time.time()
        for micro, whole in enumerate(micros):
            micros_since_log += 1
            samples_since_log += len(whole)
            batch = distributed.shard_slice(whole, rank, world)
            filler = not batch
            live = 0.0 if filler else 1.0
            if filler:
                batch = [{**whole[0], "answer_pos": None, "answer_start": None}]
            batch = with_label_answers(model, batch, collate, encoding)
            collated = collate(batch)
            anchor_positions = None
            if ref_model is not None and args.kl_weight > 0:
                anchor_positions = sample_anchor_positions(
                    collated.valid_mask,
                    collated.rows,
                    torch.where(collated.readout >= 0, collated.readout, collated.last),
                    anchor_answer_widths(batch, encoding),
                    max_positions=args.anchor_positions,
                    always=None if marker_ids is None else marker_positions(collated.input_ids, *marker_ids),
                )
            path_positions = None
            head_positions = anchor_positions
            if args.answer_path_weight > 0:
                path_positions = segment_positions(collated, [ANSWER_PATH_SEGMENT]).nonzero(as_tuple=True)
                if anchor_positions is not None:
                    head_positions = tuple(torch.cat(pair) for pair in zip(anchor_positions, path_positions))
                else:
                    head_positions = path_positions
            copy_positions = segment_positions(collated, [COPY_SEGMENT]).nonzero(as_tuple=True)
            if copy_positions[0].numel():
                head_positions = (
                    copy_positions
                    if head_positions is None
                    else tuple(torch.cat(pair) for pair in zip(head_positions, copy_positions))
                )
            (
                intro_losses,
                lm_loss,
                correct,
                labels,
                routing_scores,
                logits_lookup,
                readout_valid,
                prefix_valid,
            ) = batch_losses(
                model,
                collated,
                answer_token_ids,
                args.lm_weight,
                lm_segments=tuple(args.lm_segments),
                digit_token_ids=encoding["digit_token_ids"],
                digit_span=encoding["max_digits"],
                extra_positions=head_positions,
                digit_reduction=args.digit_reduction,
                dot_token_id=encoding.get("dot_token"),
                value_token_ids=encoding.get("value_token_ids"),
                expert_token_ids=encoding.get("expert_token_ids"),
                num_lens_layers=encoding.get("num_layers"),
                num_experts_per_tok=encoding.get("num_experts_per_tok"),
                none_token_id=encoding.get("none_token"),
                depth_digits=encoding.get("depth_digits"),
                scalar_target_width=args.scalar_target_width,
                digit_supervision=args.digit_supervision,
                end_token_id=encoding.get("turn_end_token"),
                answer_close_token_id=encoding.get("answer_close_token"),
                answer_vocab=args.answer_vocab,
            )
            valid_mask = collated.valid_mask
            weights = torch.ones(len(batch))
            observations = []
            labels_host = labels.tolist()
            for row, sample in enumerate(batch):
                pair = probe_pair(sample)
                label = labels_host[row]
                if pair_counts.modal(pair) == label:
                    weights[row] = args.lexical_match_weight
                if filler:
                    continue
                if "conv_row" in sample:
                    conv_labels.append((sample["conv_row"], label))
                observations.append((pair, label))
            for pair, label in distributed.all_gather_list(observations):
                pair_counts.update(pair, label)
            trained = trained_rows(collated.readout, labels)
            weights = weights.to(device) * trained.float().to(device)
            weight_sum = distributed.all_reduce_sum_tensor(weights.sum())
            weighted_intro = (intro_losses * weights).sum()
            intro_loss = weighted_intro / weight_sum.clamp(min=1e-6)
            copy_losses = segment_sample_losses(collated, COPY_SEGMENT, logits_lookup, len(batch))
            weighted_copy = (copy_losses * weights).sum()
            share = len(whole) / round_rows
            loss = world * (weighted_intro + weighted_copy) / round_rows + share * live * args.lm_weight * lm_loss
            path_loss = torch.zeros((), device=device)
            if path_positions is not None and args.answer_path_rows == "correct":
                correct_rows = {
                    i
                    for i, s in enumerate(batch)
                    if s.get("model_answer") is not None and s["model_answer"] == labels_host[i]
                }
                path_positions = segment_positions(collated, [ANSWER_PATH_SEGMENT], samples=correct_rows).nonzero(
                    as_tuple=True
                )
            if path_positions is not None and path_positions[0].numel():
                path_rows, path_cols = path_positions
                path_loss = torch.nn.functional.cross_entropy(
                    logits_lookup(path_rows, path_cols).float(),
                    collated.input_ids[path_rows, path_cols + 1].to(device),
                )
                loss = loss + share * live * args.answer_path_weight * path_loss
            anchor_loss = torch.zeros((), device=device)
            if anchor_positions is not None:
                assert ref_model is not None
                ref_lookup = reference_anchor_lookup(ref_model, collated, anchor_positions)
                anchor_rows, anchor_cols = anchor_positions
                anchor_loss = live * args.kl_weight * anchor_kl_at(logits_lookup, ref_lookup, anchor_rows, anchor_cols)
                loss = loss + share * anchor_loss
                del ref_lookup
            loss.backward()
            is_completed = torch.tensor([bool(s.get("completed")) for s in batch], device=correct.device)
            is_conv = torch.tensor([bool(s.get("probe_role")) for s in batch], device=correct.device) & ~is_completed
            is_tmpl = ~is_conv & ~is_completed
            valid = readout_valid
            step_metrics = (
                intro_loss.detach().float() * world,
                lm_loss.detach().float() * live,
                correct.sum().float(),
                valid.sum().float(),
                correct[is_conv].sum().float(),
                (is_conv & valid).sum().float(),
                (is_conv & prefix_valid).sum().float(),
                correct[is_tmpl].sum().float(),
                is_tmpl.sum().float() * live,
                correct[is_completed].sum().float(),
                is_completed.sum().float() * live,
                (weights == args.lexical_match_weight).float().sum(),
                torch.tensor(live * len(batch)),
                anchor_loss.detach().float(),
                path_loss.detach().float() * live,
                weighted_copy.detach().float(),
            )
            stacked = torch.stack(tuple(value.to(device) for value in step_metrics))
            values = distributed.all_reduce_sum_tensor(stacked).tolist()
            (
                intro_v,
                lm_v,
                hits,
                n_valid,
                conv_hits,
                conv_n,
                conv_n_on_valid,
                tmpl_hits,
                tmpl_n,
                completed_hits,
                completed_n,
                lex_count,
                n_rows,
                anchor_v,
                path_v,
                copy_v,
            ) = values
            log["intro"].append(intro_v / world)
            log["lm"].append(lm_v / world)
            log["acc"].append(hits / max(n_valid, 1))
            if conv_n:
                log["acc_conv"].append(conv_hits / conv_n)
                conv_micros_since_log += 1
            if conv_n_on_valid:
                log["conv_on_valid_hits"].append(conv_hits)
                log["conv_on_valid_n"].append(conv_n_on_valid)
                on_valid_micros_since_log += 1
            if tmpl_n:
                log["acc_tmpl"].append(tmpl_hits / tmpl_n)
                tmpl_micros_since_log += 1
            if completed_n:
                log["acc_completed"].append(completed_hits / completed_n)
                completed_micros_since_log += 1
            log["lex"].append(lex_count / max(n_rows, 1))
            log["anchor"].append(anchor_v / world)
            log["path"].append(path_v / world)
            log["copy"].append(copy_v / max(weight_sum.item(), 1e-6))
            at_eval = round_idx % args.eval_every_nth == 0 or round_idx == args.rounds
            if at_eval and micro == len(micros) - 1:
                loadent = load_entropy_bits(family, routing_scores, valid_mask)
            del routing_scores, logits_lookup, intro_losses, valid_mask, loss
        reply_kl_sum = 0.0
        reply_positions = sum(len(r["input_ids"]) - r["reply_start"] for r in replies_sampled)
        if replies_sampled:
            assert ref_model is not None
            for whole in batching.token_chunks(
                replies_sampled, args.batch_size, args.reply_kl_tokens, world, collate.layout
            ):
                batch = distributed.shard_slice(whole, rank, world)
                live = 1.0 if batch else 0.0
                batch = [
                    {**row, "segments": {"assistant_pc_response": [[row["reply_start"], len(row["input_ids"])]]}}
                    for row in (batch or whole[:1])
                ]
                collated = collate(batch)
                positions = segment_positions(collated, ["assistant_pc_response"]).nonzero(as_tuple=True)
                columns = torch.unique(positions[1]) if getattr(model, "accepts_head_columns", False) else None
                logits, _ = model_forward(model, collated, head_columns=columns)
                seq_len = collated.input_ids.shape[1]
                policy = subset_lookup(logits, columns if columns is not None else torch.arange(seq_len), seq_len)
                reference = reference_anchor_lookup(ref_model, collated, positions)
                kl_sum = anchor_kl_at(policy, reference, *positions) * positions[0].numel() * live
                (world * args.reply_kl_weight * kl_sum / max(reply_positions, 1)).backward()
                reply_kl_sum += float(distributed.all_reduce_sum_tensor(kl_sum.detach().float()))
                del logits, policy, reference, kl_sum
        log["reply_kl"].append(reply_kl_sum / max(reply_positions, 1))
        if micros:
            distributed.clip_grad_norm(model, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for position, label in distributed.all_gather_list(conv_labels):
            conv_questions[position]["label"] = label
        log["said"].append(said_accuracy(conv_questions))
        unclosed = [p for p in conv_questions if p.get("unclosed_answer") is not None and "label" in p]
        log["unclosed_said"].append(
            sum(p["unclosed_answer"] == p["label"] for p in unclosed) / len(unclosed) if unclosed else float("nan")
        )
        dump_answer_entries(conv_questions, f"round-{round_idx}")
        train_seconds = time.time() - train_start
        sync_seconds = publish_weights(round_idx)

        if round_idx % args.log_every_nth == 0 and distributed.is_main():
            n = max(micros_since_log, 1)
            on_valid_n = sum(log["conv_on_valid_n"][-on_valid_micros_since_log:]) if on_valid_micros_since_log else 0
            on_valid_acc = (
                sum(log["conv_on_valid_hits"][-on_valid_micros_since_log:]) / on_valid_n if on_valid_n else float("nan")
            )
            print(
                f"round {round_idx} | proprio {sum(log['intro'][-n:]) / n:.4f} | lm {sum(log['lm'][-n:]) / n:.4f} "
                f"| train acc {sum(log['acc'][-n:]) / n:.4f} "
                f"(conv {window_mean(log['acc_conv'], conv_micros_since_log):.4f} / "
                f"tmpl {window_mean(log['acc_tmpl'], tmpl_micros_since_log):.4f} / "
                f"completed {window_mean(log['acc_completed'], completed_micros_since_log):.4f} / "
                f"conv on valid {on_valid_acc:.4f}) "
                f"| said acc {window_mean([v for v in log['said'][-args.log_every_nth :] if v == v], args.log_every_nth):.4f} "
                f"| unclosed said {window_mean([v for v in log['unclosed_said'][-args.log_every_nth :] if v == v], args.log_every_nth):.4f} "
                f"| lex frac {sum(log['lex'][-n:]) / n:.2f} | anchor {sum(log['anchor'][-n:]) / n:.3f} "
                f"| path {sum(log['path'][-n:]) / n:.3f} "
                f"| copy {sum(log['copy'][-n:]) / n:.3f} "
                f"| yield {sum(log['yield'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| parse {sum(log['parse'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| fallback {sum(log['fallback'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| templated {sum(log['templated'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| think {sum(log['think'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| anstok {sum(log['anstok'][-args.log_every_nth :]) / args.log_every_nth:.0f} "
                f"| reopen {sum(log['reopen'][-args.log_every_nth :]) / args.log_every_nth:.2f} "
                f"| reply kl {sum(log['reply_kl'][-args.log_every_nth :]) / args.log_every_nth:.3g} "
                f"| gen {gen_seconds:.0f}s train {train_seconds:.0f}s sync {sync_seconds:.0f}s "
                f"| {samples_since_log / (time.time() - start_time):.1f} samples/s"
            )
            start_time = time.time()
            micros_since_log = 0
            samples_since_log = 0
            conv_micros_since_log = 0
            tmpl_micros_since_log = 0
            completed_micros_since_log = 0
            on_valid_micros_since_log = 0

        if round_idx % args.eval_every_nth == 0 or round_idx == args.rounds:
            accuracy_now, entropy, _, ctx, template_types, _, _, tolerance = run_eval(
                model,
                template_eval,
                answer_token_ids,
                collate,
                args.batch_size,
                None,
                pair_counts,
                encoding,
                max_tokens=args.micro_batch_tokens,
            )
            eval_questions, eval_reopen, _ = make_conv_questions(
                args.conv_eval_samples,
                eval_seed_prompts,
                seed_block(round_idx) + 300_000,
                f"eval-{round_idx}",
                assistant_fraction=EVAL_ASSISTANT_FRACTION,
                system_prompt_fraction=EVAL_SYSTEM_PROMPT_FRACTION,
                answer_nothink_fraction=args.eval_answer_nothink_fraction,
                answer_max_tokens=args.conv_eval_answer_max_tokens,
            )
            conv_stats = {}
            for role in ("user", "assistant"):
                role_questions = [p for p in eval_questions if p["probe_role"] == role]
                if not role_questions:
                    continue
                acc_role, _, _, ctx_role, types_role, labels_role, _, tolerance_role = run_eval(
                    model,
                    role_questions,
                    answer_token_ids,
                    collate,
                    args.batch_size,
                    None,
                    pair_counts,
                    encoding,
                    max_tokens=args.micro_batch_tokens,
                )
                for probe, label in zip(role_questions, labels_role):
                    probe["label"] = label
                conv_stats[role] = (
                    acc_role,
                    said_accuracy(role_questions),
                    ctx_role,
                    len(role_questions),
                    tolerance_role,
                    sum(p["answer_pos"] is not None for p in role_questions),
                    sum(stats["n"] for stats in types_role.values()),
                    on_valid_accuracy(types_role),
                )
            dump_answer_entries(eval_questions, f"eval-{round_idx}")
            conv_text = " ".join(
                f"{role} readout {acc:.4f} (on valid {on_valid:.4f}) said {said:.4f} (ctx {ctx_role[0]:.4f} "
                f"n={ctx_role[1]}, n={n_role}, parsed={n_parsed}, scored={n_scored})"
                + tolerance_summary(tolerance_role)
                for role, (
                    acc,
                    said,
                    ctx_role,
                    n_role,
                    tolerance_role,
                    n_parsed,
                    n_scored,
                    on_valid,
                ) in conv_stats.items()
            )
            if distributed.is_main():
                print(
                    f"round {round_idx} | EVAL tmpl acc {accuracy_now:.4f} (ctx {ctx[0]:.4f}, n={ctx[1]}) "
                    f"| CONV {conv_text} | carrier reopen {eval_reopen:.2f} "
                    f"| label entropy {entropy:.2f} bits "
                    f"| load entropy {loadent:.2f} bits" + tolerance_summary(tolerance) + type_summary(template_types)
                )
                if round_idx == args.rounds and args.output:
                    conv_metrics = {
                        name: value
                        for role, (*stats, n_parsed, n_scored, on_valid) in conv_stats.items()
                        for name, value in conv_role_metrics(
                            role, *stats, num_parsed=n_parsed, num_scored=n_scored, accuracy_on_valid=on_valid
                        ).items()
                    }
                    write_eval_record(
                        Path(args.output) / "eval.json",
                        "on-policy-sft",
                        args,
                        intro_metrics(accuracy_now, ctx, entropy, tolerance=tolerance)
                        | conv_metrics
                        | {"types": template_types},
                        dataset=args.data_dir,
                        step=round_idx,
                    )
        exporting_after = round_idx == args.rounds and bool(args.output) and args.save_final
        if (
            args.save_every_nth
            and checkpoint_dir is not None
            and round_idx % args.save_every_nth == 0
            and not exporting_after
        ):
            written = checkpoint.save_checkpoint(
                checkpoint_dir / f"round-{round_idx}",
                model,
                optimizer,
                state={"round": round_idx, "rng": rng.getstate(), "pair_counts": pair_counts.counts, "log": log},
                threads=args.checkpoint_threads,
                background=args.checkpoint_background,
            )
            if distributed.is_main():
                held = "staged" if args.checkpoint_background else "written"
                print(f"round {round_idx} | checkpoint {held} in {written:.1f}s")
    previous = checkpoint.wait_for_checkpoint()
    if previous is not None and distributed.is_main():
        print(f"previous checkpoint written in {previous:.1f}s")
    if args.output and args.save_final:
        served = export.write_servable(model, family.checkpoint, Path(args.output) / "hf")
        if distributed.is_main():
            print(f"saved final checkpoint to {served}")
            if checkpoint_dir is not None:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
