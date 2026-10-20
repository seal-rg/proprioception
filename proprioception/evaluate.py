from pathlib import Path

import torch

from proprioception import distributed, primerl_backend
from proprioception.evaluation import (
    PairLabelCounts,
    collect_labels,
    intro_metrics,
    load_dataset,
    run_eval,
    seed_pair_counts,
    tolerance_summary,
    type_summary,
    write_eval_record,
)
from proprioception.families import resolve_family
from proprioception.formats import EXPERT_ID_TYPES, LENS_TYPES
from proprioception.primerl_backend import decoder_layers


def evaluate(args) -> None:
    torch.manual_seed(args.seed)
    rank, world = distributed.rank_and_world()
    if args.batch_size % world:
        raise ValueError(f"batch size {args.batch_size} does not divide across {world} ranks")
    family = resolve_family(args.model)
    encoding, train_samples, eval_samples, collate = load_dataset(args.data_dir, args.types, family, args.device)
    answer_token_ids = encoding["answer_token_ids"]
    if args.invariant_kernels:
        primerl_backend.use_invariant_kernels(True)

    def load(checkpoint):
        model = family.load(
            checkpoint, args.device, activation_checkpointing=False, expert_parallel=args.expert_parallel
        )
        model = distributed.shard_model(model, decoder_layers(model))
        model.eval()
        return model

    model = load(family.checkpoint)
    if any(s.get("type") in LENS_TYPES for s in eval_samples) and not getattr(model, "accepts_lens_request", False):
        raise ValueError(f"the {family.name!r} backend does not accept lens requests yet")
    base_labels = None
    if args.checkpoint:
        base_labels = collect_labels(
            model,
            eval_samples,
            answer_token_ids,
            collate,
            args.batch_size,
            encoding.get("num_layers"),
            encoding.get("num_experts_per_tok"),
            encoding.get("none_token"),
        )
        del model
        torch.cuda.empty_cache()
        model = load(args.checkpoint)

    pair_counts = PairLabelCounts(decay=args.lexical_lookup_decay, min_obs=args.lexical_lookup_min_count)
    seed_count = args.lexical_lookup_seed_samples
    seed_samples = train_samples if seed_count <= 0 else train_samples[:seed_count]
    seeded = seed_pair_counts(model, seed_samples, pair_counts, answer_token_ids, collate, args.batch_size, encoding)
    accuracy, entropy, agreement, ctx, type_stats, _, versus_base, tolerance = run_eval(
        model, eval_samples, answer_token_ids, collate, args.batch_size, base_labels, pair_counts, encoding
    )
    if not distributed.is_main():
        return
    line = (
        f"EVAL {args.checkpoint or family.checkpoint} on {args.data_dir} | acc {accuracy:.4f} "
        f"| ctx acc {ctx[0]:.4f} (n={ctx[1]}) | label entropy {entropy:.2f} bits | lexical seed {seeded}"
    )
    if all(s.get("type", "routing") in EXPERT_ID_TYPES for s in eval_samples):
        line += f" | chance {1 / family.num_experts:.4f}"
    if base_labels is not None:
        line += (
            f" | base-label agreement {agreement:.4f} | vs base {versus_base[0]:.4f}"
            f" | changed n={versus_base[1]}: live {versus_base[2]:.2f} base {versus_base[3]:.2f}"
        )
    print(line + tolerance_summary(tolerance) + type_summary(type_stats))
    if args.output:
        write_eval_record(
            Path(args.output) / f"eval-{Path(args.data_dir).name}.json",
            "eval",
            args,
            intro_metrics(accuracy, ctx, entropy, versus_base if base_labels is not None else None, tolerance)
            | {"types": type_stats},
            dataset=args.data_dir,
        )
