import math
import random
import shutil
import time
from pathlib import Path

import torch

from proprioception import batching, checkpoint, distributed, export, primerl_backend
from proprioception.evaluation import (
    PairLabelCounts,
    collect_labels,
    intro_metrics,
    load_dataset,
    probe_pair,
    run_eval,
    seed_pair_counts,
    tolerance_summary,
    type_summary,
    with_label_answers,
    write_eval_record,
)
from proprioception.families import resolve_family
from proprioception.batching import RANDOM_ANCHOR, TURN_END_SEGMENT
from proprioception.formats import LENS_TYPES, anchor_answer_widths, with_turn_end
from proprioception.losses import (
    anchor_kl_at,
    batch_losses,
    load_entropy_bits,
    reference_anchor_lookup,
    sample_anchor_positions,
    segment_positions,
    update_expert_bias,
)
from proprioception.distributed import report_memory
from proprioception.primerl_backend import decoder_layers


def train_sft(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device
    rank, world = distributed.rank_and_world()
    if args.batch_size % world:
        raise ValueError(f"batch size {args.batch_size} does not divide across {world} ranks")
    if args.resume == "latest" and args.save_final and export.servable_exists(args.output):
        if distributed.is_main():
            print(f"{Path(args.output) / 'hf'} already exported; nothing to do")
        return
    family = resolve_family(args.model)
    if args.gradient_checkpointing and world > 1 and family.name in ("olmoe", "lfm2"):
        raise ValueError(f"gradient checkpointing on the {family.name!r} family does not compose with sharding")
    encoding, train_samples, eval_samples, collate = load_dataset(
        args.data_dir, args.types, family, device, layout=args.batch_layout, bucket=args.batch_bucket
    )
    if args.lm_weight > 0 and TURN_END_SEGMENT in args.lm_segments:
        encoding = with_turn_end(encoding, family.checkpoint)
    answer_token_ids = encoding["answer_token_ids"]

    if args.invariant_kernels:
        primerl_backend.use_invariant_kernels(True)
    model = family.load(
        family.checkpoint,
        device,
        activation_checkpointing=args.gradient_checkpointing,
        expert_parallel=args.expert_parallel,
        compile=args.compile,
        optimization_dtype=args.optimization_dtype,
    )
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing"):
            raise ValueError(f"model family {family.name!r} does not support gradient checkpointing")
        setattr(model, "gradient_checkpointing", True)
    model = distributed.shard_model(model, decoder_layers(model))
    model.train()
    report_memory("policy load")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=True,
    )

    if any(s.get("type") in LENS_TYPES for s in train_samples + eval_samples) and not getattr(
        model, "accepts_lens_request", False
    ):
        raise ValueError(f"the {family.name!r} backend does not accept lens requests yet")

    pair_counts = PairLabelCounts(decay=args.lexical_lookup_decay, min_obs=args.lexical_lookup_min_count)
    checkpoint_dir = Path(args.output) / "checkpoints" if args.output else None
    resume_path: Path | None = Path(args.resume) if args.resume else None
    if args.resume == "latest":
        resume_path = checkpoint.latest_checkpoint(checkpoint_dir) if checkpoint_dir is not None else None
        if resume_path is None and distributed.is_main():
            print(f"no checkpoint under {checkpoint_dir}; starting fresh")
    base_labels = collect_labels(
        model,
        eval_samples,
        answer_token_ids,
        collate,
        args.batch_size,
        encoding.get("num_layers"),
        encoding.get("num_experts_per_tok"),
        encoding.get("none_token"),
        max_tokens=args.micro_batch_tokens,
    )
    if args.lexical_match_weight < 1.0 and resume_path is None:
        seed_count = args.lexical_lookup_seed_samples
        seed_samples = train_samples if seed_count <= 0 else train_samples[:seed_count]
        seeded = seed_pair_counts(
            model,
            seed_samples,
            pair_counts,
            answer_token_ids,
            collate,
            args.batch_size,
            encoding,
            max_tokens=args.micro_batch_tokens,
        )
        print(f"seeded pair counts from {seeded} train labels")
    if resume_path is None:
        accuracy0, entropy0, _, ctx0, types0, _, _, tolerance0 = run_eval(
            model,
            eval_samples,
            answer_token_ids,
            collate,
            args.batch_size,
            base_labels,
            pair_counts,
            encoding,
            max_tokens=args.micro_batch_tokens,
        )
        if distributed.is_main():
            print(
                f"step 0 | eval acc {accuracy0:.4f} | ctx acc {ctx0[0]:.4f} (n={ctx0[1]}) "
                f"| label entropy {entropy0:.2f} bits | chance {1 / family.num_experts:.4f}"
                + tolerance_summary(tolerance0)
                + type_summary(types0)
            )

    if args.balance_bias_rate > 0 and family.name != "lfm2":
        raise ValueError("aux-loss-free balancing requires a family with an expert selection bias")
    if args.balance_bias_rate > 0 and world > 1:
        raise ValueError("aux-loss-free balancing updates per-rank bias state and needs a single rank")
    ref_model = None
    if args.kl_weight > 0:
        ref_model = family.load(
            family.checkpoint,
            device,
            activation_checkpointing=args.gradient_checkpointing,
            expert_parallel=args.expert_parallel,
            compile=args.compile,
            optimization_dtype=args.optimization_dtype,
        )
        ref_model = distributed.shard_model(ref_model, decoder_layers(ref_model))
        report_memory("reference load")
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    rng = random.Random(args.seed)
    log = {"intro": [], "lm": [], "acc": [], "lex": [], "anchor": []}
    start_step = 1
    if resume_path is not None:
        saved = checkpoint.load_checkpoint(resume_path, model, optimizer)
        start_step = saved["step"] + 1
        rng.setstate(saved["rng"])
        pair_counts.restore(saved["pair_counts"])
        log = saved["log"]
        if distributed.is_main():
            print(f"resumed from {resume_path}: {saved['step']} steps done")
    start_time = time.time()
    named_anchors = [name for name in args.kl_anchor_segments if name != RANDOM_ANCHOR]
    for step in range(start_step, args.steps + 1):
        if args.lr_schedule == "cosine":
            if step <= args.warmup_steps:
                scale = step / max(args.warmup_steps, 1)
            else:
                progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
        rows = rng.sample(train_samples, args.batch_size)
        chunks = batching.token_chunks(rows, args.batch_size, args.micro_batch_tokens, world, collate.layout)
        rank_rows = sum(len(distributed.shard_slice(chunk, rank, world)) for chunk in chunks)
        metrics = torch.zeros(5, device=device)
        for chunk in chunks:
            batch = distributed.shard_slice(chunk, rank, world)
            batch = with_label_answers(model, batch, collate, encoding)
            collated = collate(batch)
            anchor_positions = None
            if ref_model is not None and args.kl_weight > 0:
                anchor_positions = sample_anchor_positions(
                    collated.valid_mask,
                    collated.rows,
                    collated.readout,
                    anchor_answer_widths(batch, encoding),
                    max_positions=args.anchor_positions if RANDOM_ANCHOR in args.kl_anchor_segments else 0,
                    always=segment_positions(collated, named_anchors) if named_anchors else None,
                )
            intro_losses, lm_loss, correct, labels, routing_scores, logits_lookup, _, _ = batch_losses(
                model,
                collated,
                answer_token_ids,
                args.lm_weight,
                digit_token_ids=encoding["digit_token_ids"],
                digit_span=encoding["max_digits"],
                extra_positions=anchor_positions,
                digit_reduction=args.digit_reduction,
                lm_segments=tuple(args.lm_segments),
                dot_token_id=encoding.get("dot_token"),
                value_token_ids=encoding.get("value_token_ids"),
                expert_token_ids=encoding.get("expert_token_ids"),
                num_lens_layers=encoding.get("num_layers"),
                num_experts_per_tok=encoding.get("num_experts_per_tok"),
                none_token_id=encoding.get("none_token"),
                depth_digits=encoding.get("depth_digits"),
                scalar_target_width=args.scalar_target_width,
                drop_probe_label_mismatch=args.drop_probe_label_mismatch,
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
                observations.append((pair, label))
            for pair, label in distributed.all_gather_list(observations):
                pair_counts.update(pair, label)
            weights = weights.to(device)
            weight_sum = distributed.all_reduce_sum_tensor(weights.sum())
            intro_loss = (intro_losses * weights).sum() / weight_sum.clamp(min=1e-6)
            loss = world * args.intro_weight * intro_loss + args.lm_weight * lm_loss
            anchor_loss = torch.zeros((), device=device)
            if ref_model is not None:
                ref_lookup = reference_anchor_lookup(ref_model, collated, anchor_positions)
                if args.kl_weight > 0:
                    assert anchor_positions is not None
                    anchor_rows, anchor_cols = anchor_positions
                    anchor_loss = anchor_loss + args.kl_weight * anchor_kl_at(
                        logits_lookup, ref_lookup, anchor_rows, anchor_cols
                    )
                loss = loss + anchor_loss
            share = len(batch) / rank_rows
            (share * loss / args.grad_accum).backward()
            if args.balance_bias_rate > 0:
                update_expert_bias(family, model, routing_scores, valid_mask, args.balance_bias_rate)
            chunk_metrics = share * torch.stack(
                (
                    intro_loss.detach().float() * world,
                    lm_loss.detach().float().to(intro_loss.device),
                    correct.float().mean().to(intro_loss.device),
                    (weights < 1.0).float().mean().to(intro_loss.device),
                    anchor_loss.detach().float().to(intro_loss.device),
                )
            )
            metrics = metrics + chunk_metrics.to(metrics.device)
        if step == 1:
            report_memory("first forward and backward")
        if step % args.grad_accum == 0:
            distributed.clip_grad_norm(model, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = distributed.all_reduce_sum_tensor(metrics) / world
        for name, value in zip(("intro", "lm", "acc", "lex", "anchor"), metrics.tolist()):
            log[name].append(value)
        at_eval = step % args.eval_every_nth == 0 or step == args.steps
        loadent = load_entropy_bits(family, routing_scores, valid_mask) if at_eval else float("nan")
        del routing_scores

        if step % args.log_every_nth == 0 and distributed.is_main():
            n = args.log_every_nth
            rate = n * args.batch_size / (time.time() - start_time)
            print(
                f"step {step} | proprio {sum(log['intro'][-n:]) / n:.4f} | "
                f"lm {sum(log['lm'][-n:]) / n:.4f} | train acc {sum(log['acc'][-n:]) / n:.4f} | "
                f"lex frac {sum(log['lex'][-n:]) / n:.2f} | {rate:.1f} samples/s"
            )
            start_time = time.time()
        if step % args.eval_every_nth == 0 or step == args.steps:
            accuracy_now, entropy, agreement, ctx, type_stats, _, versus_base, tolerance = run_eval(
                model,
                eval_samples,
                answer_token_ids,
                collate,
                args.batch_size,
                base_labels,
                pair_counts,
                encoding,
                max_tokens=args.micro_batch_tokens,
            )
            n_ev = min(args.eval_every_nth, len(log["anchor"]))
            if distributed.is_main():
                print(
                    f"step {step} | EVAL acc {accuracy_now:.4f} | ctx acc {ctx[0]:.4f} (n={ctx[1]}) "
                    f"| label entropy {entropy:.2f} bits | pair entropy {pair_counts.pair_entropy_bits():.2f} bits "
                    f"| load entropy {loadent:.2f} bits "
                    f"| anchor {sum(log['anchor'][-n_ev:]) / n_ev:.3f} "
                    f"| base-label agreement {agreement:.4f} | vs base {versus_base[0]:.4f} "
                    f"| changed n={versus_base[1]}: live {versus_base[2]:.2f} base {versus_base[3]:.2f}"
                    + tolerance_summary(tolerance)
                    + type_summary(type_stats)
                )
                if step == args.steps and args.output:
                    write_eval_record(
                        Path(args.output) / "eval.json",
                        "sft",
                        args,
                        intro_metrics(accuracy_now, ctx, entropy, versus_base, tolerance) | {"types": type_stats},
                        dataset=args.data_dir,
                        step=step,
                    )
        exporting_after = step == args.steps and bool(args.output) and args.save_final
        if (
            args.save_every_nth
            and checkpoint_dir is not None
            and step % args.save_every_nth == 0
            and not exporting_after
        ):
            written = checkpoint.save_checkpoint(
                checkpoint_dir / f"step-{step}",
                model,
                optimizer,
                state={"step": step, "rng": rng.getstate(), "pair_counts": pair_counts.counts, "log": log},
                threads=args.checkpoint_threads,
                background=args.checkpoint_background,
            )
            if distributed.is_main():
                held = "staged" if args.checkpoint_background else "written"
                print(f"step {step} | checkpoint {held} in {written:.1f}s")
    previous = checkpoint.wait_for_checkpoint()
    if previous is not None and distributed.is_main():
        print(f"previous checkpoint written in {previous:.1f}s")
    if args.output and args.save_final:
        served = export.write_servable(model, family.checkpoint, Path(args.output) / "hf")
        if distributed.is_main():
            print(f"saved final checkpoint to {served}")
            if checkpoint_dir is not None:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
