import dataclasses

import torch


@dataclasses.dataclass
class Batch:
    input_ids: torch.Tensor
    rows: torch.Tensor
    last: torch.Tensor
    probe: torch.Tensor
    layer: torch.Tensor
    segments: dict[str, list[tuple[int, int, int]]]
    segment_samples: dict[str, list[int]]
    readout: torch.Tensor
    emitted: torch.Tensor
    probe_labels: torch.Tensor | None
    valid_mask: torch.Tensor
    position_ids: torch.Tensor | None
    types: list[str]
    ranks: torch.Tensor
    partner_span: torch.Tensor


def _merge_segments(
    per_sample: list[dict[str, list[tuple[int, int]]]], offsets: list[int], rows: list[int]
) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, list[int]]]:
    merged: dict[str, list[tuple[int, int, int]]] = {name: [] for name in SEGMENT_NAMES}
    owners: dict[str, list[int]] = {name: [] for name in SEGMENT_NAMES}
    for index, (segments, offset, row) in enumerate(zip(per_sample, offsets, rows)):
        for name, ranges in segments.items():
            merged[name].extend((row, offset + lo, offset + hi) for lo, hi in ranges)
            owners[name].extend(index for _ in ranges)
    return merged, owners


SEGMENT_NAMES = (
    "user_surface_formatting",
    "user_surface_sentence",
    "user_pc_question",
    "assistant_pc_formatting",
    "assistant_pc_reasoning",
    "assistant_pc_response",
    "assistant_pc_copy",
    "assistant_proprioception",
    "assistant_turn_end",
    "assistant_answer_path",
)
TURN_END_SEGMENT = "assistant_turn_end"
ANSWER_PATH_SEGMENT = "assistant_answer_path"
COPY_SEGMENT = "assistant_pc_copy"
RANDOM_ANCHOR = "random_global"


def sample_segments(sample: dict) -> dict[str, list[tuple[int, int]]]:
    stored = sample["segments"]
    ranges = {name: [tuple(r) for r in stored.get(name, [])] for name in stored}
    answer_start = sample.get("answer_start")
    if answer_start is None:
        answer_start = sample.get("answer_pos")
    if answer_start is not None and answer_start < len(sample["input_ids"]):
        ranges["assistant_proprioception"] = [(int(answer_start), len(sample["input_ids"]))]
    return {name: ranges.get(name, []) for name in SEGMENT_NAMES}


def partner_span(sample: dict, offset: int = 0) -> tuple[int, int]:
    stored = sample.get("partner_span")
    if stored is not None:
        start, end = stored
        return offset + int(start), offset + int(end)
    ranges = (sample.get("segments") or {}).get("user_surface_sentence") or []
    if not ranges:
        return -1, -1
    start, end = ranges[0]
    return offset + int(start), offset + int(end)


def readout_position(sample: dict) -> int:
    if sample.get("answer_start") is not None:
        return sample["answer_start"] - 1
    if "answer_pos" not in sample:
        return len(sample["input_ids"]) - 1
    if sample["answer_pos"] is None:
        return -1
    return sample["answer_pos"] - 1


def split_round(items: list, parts: int) -> list[list]:
    chunks = []
    start = 0
    for index in range(parts):
        end = start + len(items) // parts + (1 if index < len(items) % parts else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def token_chunks(
    rows: list, max_rows: int, max_tokens: int, world: int = 1, layout: str = "padded", *, keep_order: bool = False
) -> list[list]:
    if not rows:
        return []
    if not max_tokens:
        return split_round(rows, -(-len(rows) // max_rows))
    ordered = rows if keep_order else sorted(rows, key=lambda row: len(row["input_ids"]), reverse=True)

    def rank_cost(chunk: list) -> int:
        per_rank = -(-len(chunk) // world)
        lengths = sorted((len(row["input_ids"]) for row in chunk), reverse=True)
        return sum(lengths[:per_rank]) if layout == "packed" else per_rank * lengths[0]

    chunks: list[list] = []
    current: list = []
    for start in range(0, len(ordered), world):
        group = ordered[start : start + world]
        if current and (len(current) + len(group) > max_rows or rank_cost(current + group) > max_tokens):
            chunks.append(current)
            current = []
        current = current + group
    chunks.append(current)
    return chunks


@dataclasses.dataclass
class Collator:
    pad_id: int
    device: str
    layout: str
    bucket: int = 1

    def __call__(self, batch: list[dict]) -> Batch:
        return build(batch, self.pad_id, self.device, layout=self.layout, bucket=self.bucket)


def build(batch: list[dict], pad_id: int, device: str, *, layout: str, bucket: int = 1) -> Batch:
    if layout == "padded":
        return _padded(batch, pad_id, device)
    if layout == "packed":
        return _packed(batch, pad_id, device, bucket)
    if layout == "unmeasured":
        raise ValueError("this family's batch layout has not been measured; run the packing leakage check first")
    raise ValueError(f"unknown batch layout {layout!r}")


def _common(batch: list[dict], device: str) -> dict:
    return {
        "probe_labels": (
            torch.tensor([s.get("label_from_probe", -1) for s in batch], device=device)
            if any("label_from_probe" in s for s in batch)
            else None
        ),
        "layer": torch.tensor([s.get("layer", -1) for s in batch], device=device),
        "emitted": torch.tensor([s.get("emitted_digits", 1) for s in batch], device=device),
        "types": [s.get("type", "routing") for s in batch],
        "ranks": torch.tensor([s.get("rank", 1) for s in batch], device=device),
    }


def _padded(batch: list[dict], pad_id: int, device: str) -> Batch:
    width = max(len(s["input_ids"]) for s in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    for row, sample in enumerate(batch):
        input_ids[row, : len(sample["input_ids"])] = torch.tensor(sample["input_ids"])
    last = torch.tensor([len(s["input_ids"]) - 1 for s in batch], device=device)
    segments, segment_samples = _merge_segments(
        [sample_segments(s) for s in batch], offsets=[0] * len(batch), rows=list(range(len(batch)))
    )
    return Batch(
        input_ids=input_ids.to(device),
        rows=torch.arange(len(batch), device=device),
        last=last,
        probe=torch.tensor([s["probe_pos"] for s in batch], device=device),
        segments=segments,
        segment_samples=segment_samples,
        readout=torch.tensor([readout_position(s) for s in batch], device=device),
        valid_mask=(torch.arange(width, device=device)[None, :] <= last[:, None]).float(),
        position_ids=None,
        partner_span=torch.tensor([partner_span(s) for s in batch], dtype=torch.long, device=device).view(-1, 2),
        **_common(batch, device),
    )


def _packed(batch: list[dict], pad_id: int, device: str, bucket: int = 1) -> Batch:
    input_ids: list[int] = []
    position_ids: list[int] = []
    last: list[int] = []
    probe: list[int] = []
    readout: list[int] = []
    per_sample: list[dict] = []
    offsets: list[int] = []
    partner_spans: list[tuple[int, int]] = []
    for sample in batch:
        start = len(input_ids)
        partner_spans.append(partner_span(sample, start) if partner_span(sample)[0] >= 0 else (-1, -1))
        tokens = sample["input_ids"]
        input_ids.extend(tokens)
        position_ids.extend(range(len(tokens)))
        last.append(start + len(tokens) - 1)
        probe.append(start + sample["probe_pos"])
        local_readout = readout_position(sample)
        readout.append(-1 if local_readout < 0 else start + local_readout)
        per_sample.append(sample_segments(sample))
        offsets.append(start)
    real_tokens = len(input_ids)
    padding = (-real_tokens) % bucket
    input_ids.extend([pad_id] * padding)
    position_ids.extend(range(padding))
    mask = torch.zeros((1, real_tokens + padding), device=device)
    mask[0, :real_tokens] = 1.0
    segments, segment_samples = _merge_segments(per_sample, offsets, rows=[0] * len(per_sample))
    return Batch(
        input_ids=torch.tensor([input_ids], device=device),
        rows=torch.zeros(len(batch), dtype=torch.long, device=device),
        last=torch.tensor(last, device=device),
        probe=torch.tensor(probe, device=device),
        segments=segments,
        segment_samples=segment_samples,
        readout=torch.tensor(readout, device=device),
        valid_mask=mask,
        position_ids=torch.tensor([position_ids], device=device),
        partner_span=torch.tensor(partner_spans, dtype=torch.long, device=device).view(-1, 2),
        **_common(batch, device),
    )
