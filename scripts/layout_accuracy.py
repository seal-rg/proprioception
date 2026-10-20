import argparse
import json
import os
import time
from pathlib import Path

import torch

from proprioception import primerl_backend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--checkpoint", default="", help="servable checkpoint dir (<run>/hf); the base model when unset"
    )
    parser.add_argument("--invariant_kernels", action="store_true", help="deterministic, batch-invariant kernels")
    parser.add_argument("--layout", required=True, choices=["padded", "packed"])
    parser.add_argument("--num-questions", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    for name in ("data_dir", "checkpoint", "out"):
        setattr(args, name, os.path.expandvars(getattr(args, name)))
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()
    for key, value in (
        ("MASTER_ADDR", "localhost"),
        ("MASTER_PORT", "29770"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(key, value)
    from proprioception import batching
    from proprioception.evaluation import run_eval
    from proprioception.families import resolve_family
    from proprioception.formats import answer_encoding

    family = resolve_family(args.model)
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    encoding = answer_encoding(meta, family, "cuda:0")
    samples = json.loads((data_dir / "eval.json").read_text())[: args.num_questions]

    primerl_backend.use_invariant_kernels(args.invariant_kernels)
    model = family.load(
        args.checkpoint or family.checkpoint, "cuda:0", activation_checkpointing=False, expert_parallel=False
    )
    model.eval()
    collate = batching.Collator(pad_id=1, device="cuda:0", layout=args.layout, bucket=family.batch_bucket)

    start = time.time()
    accuracy, entropy, _, context, _, _, _, _ = run_eval(
        model, samples, encoding["answer_token_ids"], collate, args.batch_size, None, None, encoding
    )
    elapsed = time.time() - start
    invariant = args.invariant_kernels
    print(
        f"\n{args.model} | layout {args.layout} | invariant={invariant} | n={len(samples)}\n"
        f"  accuracy {accuracy:.4f} | context-dependent subset {context[0]:.4f} (n={context[1]}) | "
        f"label entropy {entropy:.2f} bits | {elapsed:.1f}s"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "layout": args.layout,
                    "invariant_kernels": invariant,
                    "accuracy": accuracy,
                    "context_accuracy": context[0],
                    "context_n": context[1],
                    "seconds": elapsed,
                    "n": len(samples),
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
