import collections
import dataclasses
import datetime
import json
import math
import time
from pathlib import Path

import torch

from proprioception import batching, distributed
from proprioception.formats import (
    RANKED_TYPES,
    answer_encoding,
    answer_closing,
    answer_extension_widths,
    answer_tokens,
    check_probed_layers,
    check_sample_types,
    select_sample_types,
)
from proprioception.losses import batch_losses, predicted_numbers


def with_label_answers(model, batch: list[dict], collate, encoding: dict) -> list[dict]:
    widths = answer_extension_widths(batch, encoding)
    if not distributed.all_reduce_max(int(any(widths))):
        return batch
    filler = encoding["filler_token"]
    end = int(encoding.get("turn_end_token") is not None)
    probe_batch = [
        {**s, "input_ids": s["input_ids"] + [filler] * (w + end + tail_width(s, w))} if w else s
        for s, w in zip(batch, widths)
    ]
    with torch.no_grad():
        _, _, _, labels, _, _, _, _ = batch_losses(
            model,
            collate(probe_batch),
            encoding["answer_token_ids"],
            lm_weight=0.0,
            num_lens_layers=encoding.get("num_layers"),
            num_experts_per_tok=encoding.get("num_experts_per_tok"),
            none_token_id=encoding.get("none_token"),
        )
    return extend_with_labels(batch, labels.cpu().tolist(), encoding)


def tail_width(sample: dict, width: int) -> int:
    tail = sample.get("tail")
    return len(tail["input_ids"]) + width if tail else 0


def extend_with_labels(batch: list[dict], labels: list[int], encoding: dict) -> list[dict]:
    widths = answer_extension_widths(batch, encoding)
    if not any(widths):
        return batch
    filler = encoding["filler_token"]
    end = [] if encoding.get("turn_end_token") is None else [encoding["turn_end_token"]]
    extended = []
    for sample, width, label in zip(batch, widths, labels):
        if not width:
            extended.append(sample)
            continue
        type = sample.get("type", "routing")
        answer = answer_tokens(type, label, encoding) if label >= 0 else []
        tokens = answer + answer_closing(type, encoding) if answer else []
        turn_end = end if tokens else [filler] * len(end)
        start = len(sample["input_ids"])
        segments = dict(sample.get("segments", {}))
        tail = sample.get("tail")
        if tail:
            copy_start = start + len(tokens) + len(tail["input_ids"])
            for name, ranges in tail["segments"].items():
                offset = start + len(tokens)
                segments[name] = segments.get(name, []) + [[offset + lo, offset + hi] for lo, hi in ranges]
            segments["assistant_pc_copy"] = [[copy_start, copy_start + len(tokens)]]
            appended = tokens + tail["input_ids"] + tokens + turn_end + [filler] * (2 * (width - len(tokens)))
        else:
            appended = tokens + turn_end + [filler] * (width - len(tokens))
        extended.append(
            {
                **{k: v for k, v in sample.items() if k != "tail"},
                "input_ids": sample["input_ids"] + appended,
                "segments": segments,
                "answer_start": start,
                "emitted_digits": len(answer),
                "label_from_probe": label,
            }
        )
    return extended


def load_dataset(data_dir: str | Path, types: list[str], family, device, *, layout: str = "", bucket: int = 0):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    encoding = answer_encoding(meta, family, device)
    train_samples = select_sample_types(json.loads((data_dir / "train.json").read_text()), types, "train")
    eval_samples = select_sample_types(json.loads((data_dir / "eval.json").read_text()), types, "eval")
    check_probed_layers(train_samples + eval_samples, family)
    check_sample_types(train_samples + eval_samples, encoding)
    collate = batching.Collator(
        pad_id=1, device=device, layout=layout or family.batch_layout, bucket=bucket or family.batch_bucket
    )
    return encoding, train_samples, eval_samples, collate


def seed_pair_counts(
    model, samples, pair_counts, answer_token_ids, collate, batch_size, encoding, *, max_tokens: int = 0
) -> int:
    labels = collect_labels(
        model,
        samples,
        answer_token_ids,
        collate,
        batch_size,
        encoding.get("num_layers"),
        encoding.get("num_experts_per_tok"),
        encoding.get("none_token"),
        max_tokens=max_tokens,
    )
    for sample, label in zip(samples, labels):
        pair_counts.update(probe_pair(sample), label)
    return len(labels)


def intro_metrics(accuracy, ctx, entropy, versus_base=None, tolerance=None) -> dict:
    metrics = {"acc": accuracy, "ctx_acc": ctx[0], "ctx_n": ctx[1], "label_entropy": entropy}
    if tolerance:
        metrics["tolerance"] = tolerance
        if len(tolerance) == 1:
            stats = next(iter(tolerance.values()))
            metrics.update(
                mae=stats["mae"],
                within_5=stats["within_5"],
                within_10=stats["within_10"],
                constant_mae=stats["constant_mae"],
                constant_within_10=stats["constant_within_10"],
            )
    if versus_base is not None and versus_base[0] == versus_base[0]:
        metrics.update(
            vs_base=versus_base[0], changed_n=versus_base[1], changed_live=versus_base[2], changed_base=versus_base[3]
        )
    return metrics


def git_head(repo: Path = Path(__file__).resolve().parents[1]) -> str:
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            return (repo / ".git" / head[5:]).read_text().strip()[:9]
        return head[:9]
    except OSError:
        return ""


STARTUP_COMMIT = git_head()


def write_eval_record(path: Path, stage: str, args, metrics: dict, *, dataset: str, step: int | None = None) -> None:
    record = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "commit": STARTUP_COMMIT,
        "dataset": Path(dataset).name,
        "step": step,
        "metrics": metrics,
        "config": dataclasses.asdict(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")


def probe_pair(sample: dict) -> tuple:
    return (
        sample["input_ids"][sample["probe_pos"]],
        sample.get("layer", -1),
        sample.get("type", "routing"),
        sample.get("rank", 1),
    )


class PairLabelCounts:
    def __init__(self, decay: float, min_obs: float):
        self.counts: dict[tuple[int, int, str, int], dict[int, float]] = {}
        self.decay = decay
        self.min_obs = min_obs

    def restore(self, counts: dict) -> None:
        self.counts = {
            (key if len(key) == 4 else (*key, 1)): counter
            for key, counter in counts.items()
            if len(key) == 4 or key[2] not in RANKED_TYPES
        }

    def modal(self, pair: tuple[int, int, str, int]) -> int | None:
        counter = self.counts.get(pair)
        if counter is None or sum(counter.values()) < self.min_obs:
            return None
        return max(counter, key=lambda k: counter[k])

    def update(self, pair: tuple[int, int, str, int], label: int) -> None:
        counter = self.counts.setdefault(pair, {})
        for k in list(counter):
            counter[k] *= self.decay
            if counter[k] < 1e-6:
                del counter[k]
        counter[label] = counter.get(label, 0.0) + 1.0

    def pair_entropy_bits(self) -> float:
        entropies = []
        for counter in self.counts.values():
            total = sum(counter.values())
            if total < self.min_obs:
                continue
            ratios = [v / total for v in counter.values()]
            entropies.append(-sum(p * math.log2(p) for p in ratios if p > 0.0))
        return sum(entropies) / len(entropies) if entropies else float("nan")


def run_eval(
    model,
    samples,
    answer_token_ids,
    collate,
    batch_size,
    base_labels=None,
    pair_counts=None,
    encoding=None,
    *,
    max_tokens: int = 0,
):
    model.eval()
    rank, world = distributed.rank_and_world()
    if base_labels is not None:
        base_labels = distributed.shard_slice(list(base_labels), rank, world)
    all_samples = samples
    samples = distributed.shard_slice(samples, rank, world)
    records: list[tuple] = []
    done = 0

    def score(batch, label_override=None):
        collated = collate(batch)
        return collated, batch_losses(
            model,
            collated,
            answer_token_ids,
            lm_weight=0.0,
            digit_token_ids=None if encoding is None else encoding["digit_token_ids"],
            digit_span=None if encoding is None else encoding["max_digits"],
            dot_token_id=None if encoding is None else encoding.get("dot_token"),
            value_token_ids=None if encoding is None else encoding.get("value_token_ids"),
            expert_token_ids=None if encoding is None else encoding.get("expert_token_ids"),
            num_lens_layers=None if encoding is None else encoding.get("num_layers"),
            depth_digits=None if encoding is None else encoding.get("depth_digits"),
            num_experts_per_tok=None if encoding is None else encoding.get("num_experts_per_tok"),
            none_token_id=None if encoding is None else encoding.get("none_token"),
            label_override=label_override,
        )

    with torch.no_grad():
        for chunk, real in distributed.uniform_batches(
            samples, batch_size, all_samples[0] if all_samples else None, max_tokens=max_tokens, layout=collate.layout
        ):
            batch = chunk if encoding is None else with_label_answers(model, chunk, collate, encoding)
            collated, (_, _, correct, labels, _, logits_lookup, readout_valid, prefix_valid) = score(batch)
            predicted = predicted_numbers(
                logits_lookup,
                collated.input_ids.to(labels.device),
                collated.rows,
                collated.readout,
                collated.last,
                labels,
                collated.types,
                encoding or {},
            )
            correct_base = None
            if base_labels is not None:
                chunk_base = base_labels[done : done + real] + [0] * (len(chunk) - real)
                base_batch = chunk if encoding is None else extend_with_labels(chunk, chunk_base, encoding)
                _, (_, _, correct_base, _, _, _, _, _) = score(
                    base_batch, torch.tensor(chunk_base, device=labels.device)
                )
            if not real:
                continue
            correct_host = correct[:real].cpu().tolist()
            labels_host = labels[:real].cpu().tolist()
            valid_host = readout_valid[:real].cpu().tolist()
            prefix_host = prefix_valid[:real].cpu().tolist()
            predicted_host = predicted[:real].cpu().tolist()
            base_host = None if correct_base is None else correct_base[:real].cpu().tolist()
            for row, sample in enumerate(chunk[:real]):
                ctx_eligible = False
                if pair_counts is not None and valid_host[row]:
                    modal = pair_counts.modal(probe_pair(sample))
                    ctx_eligible = modal is not None and modal != labels_host[row]
                agree = None if base_labels is None else labels_host[row] == base_labels[done + row]
                records.append(
                    (
                        sample.get("type", "routing"),
                        labels_host[row],
                        bool(correct_host[row]),
                        bool(valid_host[row]),
                        ctx_eligible,
                        agree,
                        None if base_host is None else bool(base_host[row]),
                        None if predicted_host[row] < 0 else predicted_host[row],
                        bool(prefix_host[row]),
                    )
                )
            done += real
    model.train()
    return summarize_records(distributed.all_gather_list(records))


def summarize_records(records: list[tuple]) -> tuple:
    total = sum(r[3] for r in records)
    total_correct = sum(r[2] for r in records)
    label_counts = collections.Counter(r[1] for r in records)
    entropy = (
        -sum((count / len(records)) * math.log2(count / len(records)) for count in label_counts.values())
        if records
        else float("nan")
    )
    agreements = [r[5] for r in records if r[5] is not None]
    agreement = sum(agreements) / len(agreements) if agreements else float("nan")
    ctx = context_stats(records)
    versus_base = versus_base_stats(records)
    type_stats = {}
    for pc_type in {r[0] for r in records}:
        of_type = [r for r in records if r[0] == pc_type]
        type_valid = [r[2] for r in of_type if r[3]]
        type_on_valid = [r[2] for r in of_type if r[8]]
        type_ctx = context_stats(of_type)
        type_base = versus_base_stats(of_type)
        type_stats[pc_type] = {
            "acc": sum(type_valid) / len(type_valid) if type_valid else float("nan"),
            "n": len(type_valid),
            "acc_on_valid": sum(type_on_valid) / len(type_on_valid) if type_on_valid else float("nan"),
            "n_on_valid": len(type_on_valid),
            "ctx_acc": type_ctx[0],
            "ctx_n": type_ctx[1],
            "vs_base": type_base[0],
            "changed_n": type_base[1],
            "changed_live": type_base[2],
            "changed_base": type_base[3],
        }
    tolerance = {}
    with_number = [r for r in records if r[3] and r[7] is not None]
    for pc_type in sorted({r[0] for r in with_number}):
        tolerance[pc_type] = distance_stats([r for r in with_number if r[0] == pc_type])
    labels = [r[1] for r in records]
    accuracy = total_correct / total if total else float("nan")
    return accuracy, entropy, agreement, ctx, type_stats, labels, versus_base, tolerance


def context_stats(records: list[tuple]) -> tuple[float, int]:
    ctx_total = sum(r[4] for r in records)
    ctx_correct = sum(r[2] for r in records if r[4])
    return (ctx_correct / ctx_total, int(ctx_total)) if ctx_total else (float("nan"), 0)


def versus_base_stats(records: list[tuple]) -> tuple[float, int, float, float]:
    scored_base = [r for r in records if r[3] and r[6] is not None]
    changed = [r for r in scored_base if r[5] is False]
    return (
        sum(r[6] for r in scored_base) / len(scored_base) if scored_base else float("nan"),
        len(changed),
        sum(r[2] for r in changed) / len(changed) if changed else float("nan"),
        sum(r[6] for r in changed) / len(changed) if changed else float("nan"),
    )


def best_constant_stats(labels: list[int]) -> dict:
    ordered = sorted(labels)
    median = ordered[len(ordered) // 2]

    def coverage(width: int) -> float:
        best = reach = 0
        for low in range(len(ordered)):
            while reach < len(ordered) and ordered[reach] <= ordered[low] + 2 * width:
                reach += 1
            best = max(best, reach - low)
        return best / len(ordered)

    return {
        "mae": sum(abs(label - median) for label in ordered) / len(ordered),
        "within_5": coverage(5),
        "within_10": coverage(10),
    }


def distance_stats(records: list[tuple]) -> dict:
    errors = [abs(r[7] - r[1]) for r in records]
    baseline = best_constant_stats([r[1] for r in records])
    return {
        "mae": sum(errors) / len(errors),
        "within_5": sum(e <= 5 for e in errors) / len(errors),
        "within_10": sum(e <= 10 for e in errors) / len(errors),
        "n": len(errors),
        "constant_mae": baseline["mae"],
        "constant_within_5": baseline["within_5"],
        "constant_within_10": baseline["within_10"],
    }


def tolerance_summary(tolerance: dict) -> str:
    if not tolerance:
        return ""
    if len(tolerance) == 1:
        stats = next(iter(tolerance.values()))
        return (
            f" | mae {stats['mae']:.2f} (constant {stats['constant_mae']:.2f}) | within 5 {stats['within_5']:.4f}"
            f" | within 10 {stats['within_10']:.4f} (constant {stats['constant_within_10']:.4f}, n={stats['n']})"
        )
    return "".join(
        f" | {name} mae {s['mae']:.2f}/{s['constant_mae']:.2f} within 5 {s['within_5']:.4f}"
        f" within 10 {s['within_10']:.4f}/{s['constant_within_10']:.4f} (n={s['n']})"
        for name, s in sorted(tolerance.items())
    )


def type_summary(type_stats: dict) -> str:
    if set(type_stats) == {"routing"}:
        return ""
    return " | types " + " ".join(
        f"{pc_type} {s['acc']:.4f} (n={s['n']}, ctx {s['ctx_acc']:.4f} n={s['ctx_n']})"
        for pc_type, s in sorted(type_stats.items())
    )


def collect_labels(
    model,
    samples,
    answer_token_ids,
    collate,
    batch_size,
    num_lens_layers=None,
    num_experts_per_tok=None,
    none_token_id=None,
    *,
    max_tokens: int = 0,
):
    rank, world = distributed.rank_and_world()
    mine = distributed.shard_slice(list(samples), rank, world)
    labels = []
    started = time.time()
    done = 0
    with torch.no_grad():
        for chunk, real in distributed.uniform_batches(
            mine, batch_size, samples[0] if samples else None, max_tokens=max_tokens, layout=collate.layout
        ):
            if done and done % (batch_size * 500) == 0 and distributed.is_main():
                rate = done / max(time.time() - started, 1e-9)
                print(
                    f"  labels {done}/{len(mine)} per rank | {rate:.0f} samples/s | {(len(mine) - done) / rate:.0f}s left"
                )
            _, _, _, batch_labels, _, _, _, _ = batch_losses(
                model,
                collate(chunk),
                answer_token_ids,
                lm_weight=0.0,
                num_lens_layers=num_lens_layers,
                num_experts_per_tok=num_experts_per_tok,
                none_token_id=none_token_id,
            )
            labels.extend(batch_labels.cpu().tolist()[:real])
            done += real
    return distributed.all_gather_list(labels)
