import math

import torch

from proprioception.batching import TURN_END_SEGMENT
from proprioception.formats import (
    ENTROPY_MAX_BITS,
    ENTROPY_MAX_DECIBITS,
    ENTROPY_TYPES,
    INT_ENTROPY_TYPES,
    WHOLE_VALUE_TYPES,
    EXPERT_ID_TYPES,
    LENS_TYPES,
    LOGITS_TYPES,
    NUMERIC_TYPES,
    PERCENT_MAX,
    PERCENT_TYPES,
    ROUTING_TYPES,
    SCALAR_TYPES,
    SINGLE_TOKEN_TYPES,
    TOP_TOKEN_MAX_RANK,
    NumberFormat,
    answer_closing,
    answer_tokens,
    answer_width,
    number_format,
)


def spelled_number_loss(
    logits_lookup,
    ids,
    batch_idx,
    readout,
    labels,
    emitted,
    digit_token_ids,
    active,
    form: NumberFormat,
    dot_token_id=None,
    reduction="sum",
    *,
    last,
    digit_supervision="correct_prefix",
    own_end=True,
    answer_vocab=None,
):
    device = ids.device
    num_samples = readout.numel()
    label = labels.to(device)
    count = emitted.to(device)
    start = readout.to(device)
    rows_all = batch_idx.to(device)
    width = form.max_int_digits
    integer = label // 10 if form.decimal else label
    fraction = label % 10

    powers = 10 ** torch.arange(width - 1, -1, -1, device=device)
    num_int_digits = (integer.unsqueeze(-1) >= powers).sum(-1).clamp(min=1)
    num_tokens = num_int_digits + (2 if form.decimal else 0)
    offsets = torch.arange(form.max_tokens + 1, device=device)
    shift = (num_int_digits.unsqueeze(-1) - 1 - offsets).clamp(min=0)
    digits = (integer.unsqueeze(-1) // (10**shift)) % 10
    lookup = torch.tensor(digit_token_ids, device=device)
    targets_grid = lookup[digits]
    if form.decimal:
        if dot_token_id is None:
            raise ValueError("a decimal answer needs the dot token from the dataset encoding")
        at_dot = offsets.unsqueeze(0) == num_int_digits.unsqueeze(-1)
        at_tenths = offsets.unsqueeze(0) == (num_int_digits + 1).unsqueeze(-1)
        targets_grid = torch.where(at_dot, torch.full_like(targets_grid, int(dot_token_id)), targets_grid)
        targets_grid = torch.where(at_tenths, lookup[fraction].unsqueeze(-1).expand_as(targets_grid), targets_grid)

    ids_dev = ids.to(device)
    follows = (start + 1 + count).clamp(min=0, max=ids.shape[1] - 1)
    end_target = ids_dev[rows_all, follows]
    is_end = offsets.unsqueeze(0) == num_tokens.unsqueeze(-1)
    spelling_grid = targets_grid
    targets_grid = torch.where(is_end, end_target.unsqueeze(-1), targets_grid)

    last_col = last.to(device)
    positions_grid = start.unsqueeze(-1) + offsets
    emitted_grid = ids_dev[rows_all.unsqueeze(-1), (positions_grid + 1).clamp(min=0, max=ids.shape[1] - 1)]
    matches = (emitted_grid == spelling_grid) | (offsets.unsqueeze(0) >= num_tokens.unsqueeze(-1))
    matches &= offsets.unsqueeze(0) < count.unsqueeze(-1)
    context_ok = torch.ones_like(matches)
    context_ok[:, 1:] = matches[:, :-1].cumprod(dim=1).bool()
    supervised = (offsets.unsqueeze(0) < num_tokens.unsqueeze(-1)) & (offsets.unsqueeze(0) <= count.unsqueeze(-1))
    if own_end:
        supervised |= is_end & (count > num_tokens).unsqueeze(-1)
    if digit_supervision == "correct_prefix":
        supervised &= context_ok
    supervised &= (start >= 0).unsqueeze(-1) & active.to(device).bool().unsqueeze(-1)
    supervised &= positions_grid <= last_col.unsqueeze(-1)
    supervised &= ~(is_end & ((start + 1 + count) > last_col).unsqueeze(-1))
    whole = context_ok.gather(1, (num_tokens - 1).unsqueeze(-1)).squeeze(-1) & (count >= num_tokens)
    whole &= (start >= 0) & active.to(device).bool()

    owner_grid = torch.arange(num_samples, device=device).unsqueeze(-1).expand_as(positions_grid)
    keep = supervised.reshape(-1).nonzero(as_tuple=True)[0]
    row_index = rows_all.unsqueeze(-1).expand_as(positions_grid).reshape(-1)[keep]
    position_index = positions_grid.reshape(-1)[keep]
    target_index = targets_grid.reshape(-1)[keep]
    owner_index = owner_grid.reshape(-1)[keep]
    per_row = torch.zeros(num_samples, device=device)
    correct = torch.zeros(num_samples, dtype=torch.bool, device=device)
    if keep.numel() == 0:
        return per_row, correct, whole
    selected = logits_lookup(row_index, position_index).float()
    scored = selected
    if answer_vocab is not None:
        scored = restrict_logits(selected, answer_vocab.to(device), ~is_end.reshape(-1)[keep])
    losses = torch.nn.functional.cross_entropy(scored, target_index, reduction="none")
    counts = torch.zeros(num_samples, device=device).index_add_(0, owner_index, torch.ones_like(losses))
    per_row = torch.zeros(num_samples, device=device).index_add_(0, owner_index, losses)
    if reduction == "mean":
        per_row = per_row / counts.clamp(min=1)
    hits = (selected.argmax(-1) == target_index).float()
    row_hits = torch.zeros(num_samples, device=device).index_add_(0, owner_index, hits)
    correct = (row_hits == counts) & whole
    return per_row, correct, whole


def soft_value_loss(logits, values, value_lookup, width):
    log_probs = torch.log_softmax(logits.float(), dim=-1)[:, value_lookup]
    grid = torch.arange(value_lookup.numel(), device=logits.device, dtype=torch.float32)
    target = torch.softmax(-(grid.unsqueeze(0) - values.unsqueeze(-1).float()).abs() / width, dim=-1)
    return -(target * log_probs).sum(-1)


def restrict_logits(logits, allowed, rows):
    masked = torch.full_like(logits, float("-inf"))
    masked[:, allowed] = logits[:, allowed]
    return torch.where(rows.unsqueeze(-1), masked, logits)


def fixed_slot_loss(
    logits_lookup,
    ids,
    batch_idx,
    readout,
    values,
    slots,
    digit_lookup,
    active,
    last,
    reduction="sum",
    value_lookup=None,
    emitted=None,
    value_soft_width=0.0,
    digit_supervision="correct_prefix",
    answer_vocab=None,
):
    device = values.device
    num_samples = values.numel()
    width = len(slots)
    columns = []
    for slot_type, slot_value in slots:
        if slot_type == "digit":
            columns.append(digit_lookup[(values.clamp(min=0) // slot_value) % 10])
        elif slot_type == "value":
            if value_lookup is None:
                raise ValueError("a fused value slot needs the dataset's value table")
            columns.append(value_lookup[(values.clamp(min=0) // slot_value) % value_lookup.numel()])
        else:
            columns.append(torch.full_like(values, slot_value))
    targets_grid = torch.stack(columns, dim=-1)
    offsets = torch.arange(width, device=device)
    positions_grid = readout.unsqueeze(-1) + offsets
    ids_dev = ids.to(device)
    emitted_grid = ids_dev[batch_idx.unsqueeze(-1), (positions_grid + 1).clamp(min=0, max=ids.shape[1] - 1)]
    context_ok = torch.ones(num_samples, width, dtype=torch.bool, device=device)
    context_ok[:, 1:] = (emitted_grid == targets_grid)[:, :-1].cumprod(dim=1).bool()
    supervised = (values >= 0) & (readout >= 0) & active.bool()
    supervised = supervised.unsqueeze(-1) & (positions_grid <= last.to(device).unsqueeze(-1))
    if emitted is not None:
        supervised = supervised & (offsets < emitted.to(device).unsqueeze(-1))
    whole = (supervised & context_ok).all(dim=1)
    if digit_supervision == "correct_prefix":
        supervised = supervised & context_ok
    keep = supervised.reshape(-1).nonzero(as_tuple=True)[0]
    per_row = torch.zeros(num_samples, device=device)
    correct = torch.zeros(num_samples, dtype=torch.bool, device=device)
    if keep.numel() == 0:
        return per_row, correct, whole
    row_index = batch_idx.unsqueeze(-1).expand_as(positions_grid).reshape(-1)[keep]
    position_index = positions_grid.reshape(-1)[keep]
    target_index = targets_grid.reshape(-1)[keep]
    owner_index = torch.arange(num_samples, device=device).unsqueeze(-1).expand_as(positions_grid).reshape(-1)[keep]
    selected = logits_lookup(row_index, position_index).float()
    scored = selected
    if answer_vocab is not None:
        scored = restrict_logits(selected, answer_vocab.to(device), torch.ones_like(target_index, dtype=torch.bool))
    losses = torch.nn.functional.cross_entropy(scored, target_index, reduction="none")
    if value_soft_width > 0:
        assert value_lookup is not None
        slot_index = offsets.unsqueeze(0).expand(num_samples, -1).reshape(-1)[keep]
        for k, (slot_type, divisor) in enumerate(slots):
            if slot_type != "value":
                continue
            at_slot = slot_index == k
            if at_slot.any():
                value_index = (values[owner_index[at_slot]].clamp(min=0) // divisor) % value_lookup.numel()
                losses[at_slot] = soft_value_loss(scored[at_slot], value_index, value_lookup, value_soft_width)
    counts = torch.zeros(num_samples, device=device).index_add_(0, owner_index, torch.ones_like(losses))
    per_row = per_row.index_add_(0, owner_index, losses)
    if reduction == "mean":
        per_row = per_row / counts.clamp(min=1)
    hits = (selected.argmax(-1) == target_index).float()
    row_hits = torch.zeros(num_samples, device=device).index_add_(0, owner_index, hits)
    correct = (row_hits == counts) & whole
    return per_row, correct, whole


def compute_labels(
    model,
    batch,
    routing_scores,
    logits_lookup,
    ids,
    batch_idx,
    probe,
    out_device,
    lens=None,
    num_experts_per_tok=None,
    none_token_id=None,
):
    types = batch.types
    labels = torch.full((len(types),), -1, dtype=torch.long, device=out_device)
    routing_rows = [i for i, type in enumerate(types) if type in ROUTING_TYPES]
    if routing_rows:
        sub = torch.tensor(routing_rows, device=out_device)
        layer = batch.layer.to(out_device)[sub]
        rows, positions = batch_idx[sub], probe[sub]
        sub_types = [types[i] for i in routing_rows]
        from_scores = list(range(len(sub_types)))
        if hasattr(model, "selected_experts"):
            top1 = torch.tensor(
                [j for j, t in enumerate(sub_types) if t == "routing"], dtype=torch.long, device=out_device
            )
            if top1.numel():
                labels[sub[top1]] = (
                    model.selected_experts(layer[top1], rows[top1], positions[top1]).to(out_device).detach()
                )
            from_scores = [j for j, t in enumerate(sub_types) if t != "routing"]
        if from_scores:
            if isinstance(routing_scores, dict):
                moe_layers = sorted(routing_scores)
            else:
                moe_layers = [index for index, entry in enumerate(routing_scores) if entry is not None]
            slot = torch.full((max(moe_layers) + 1,), -1, dtype=torch.long, device=out_device)
            slot[torch.tensor(moe_layers, device=out_device)] = torch.arange(len(moe_layers), device=out_device)
            stacked = torch.stack([routing_scores[index].to(out_device) for index in moe_layers]).float().detach()
            inner = torch.tensor(from_scores, device=out_device)
            layers_i, rows_i, positions_i = layer[inner], rows[inner], positions[inner]
            scores = stacked[slot[layers_i], rows_i, positions_i]
            inner_types = [sub_types[j] for j in from_scores]
            for type in set(inner_types):
                pick = torch.tensor([j for j, t in enumerate(inner_types) if t == type], device=out_device)
                outer = sub[inner[pick]]
                if type == "routing":
                    labels[outer] = scores[pick].argmax(-1)
                elif type == "routing_rank":
                    ranks = batch.ranks.to(out_device)[outer]
                    ordered = scores[pick].topk(int(ranks.max()), dim=-1).indices
                    labels[outer] = ordered.gather(1, (ranks - 1).unsqueeze(1)).squeeze(1)
                elif type == "routing_share":
                    if num_experts_per_tok is None:
                        raise ValueError("routing_share labels need the top-k from the dataset encoding")
                    if not hasattr(model, "routing_weights"):
                        raise ValueError("routing_share labels need a model that reports its combination weights")
                    selected = scores[pick].topk(num_experts_per_tok, dim=-1).indices
                    weights = model.routing_weights(layers_i[pick], scores[pick]).gather(1, selected)
                    share = weights[:, 0] / weights.sum(-1).clamp(min=1e-12)
                    labels[outer] = (share * 100).round().long().clamp(0, PERCENT_MAX)
                else:
                    if none_token_id is None:
                        raise ValueError("routing_partner labels need the none token from the dataset encoding")
                    sentence = batch.partner_span.to(out_device)[outer]
                    for k, sample_index in enumerate(outer.tolist()):
                        start, end = int(sentence[k, 0]), int(sentence[k, 1])
                        row, position = int(rows_i[pick[k]]), int(positions_i[pick[k]])
                        if start < 0 or not (start <= position < end):
                            continue
                        span_top = stacked[slot[layers_i[pick[k]]], row, start:end].argmax(-1)
                        own = span_top[position - start]
                        different = ids[row, start:end] != ids[row, position]
                        matches = ((span_top == own) & different).nonzero(as_tuple=True)[0] + start
                        if matches.numel() == 0:
                            labels[sample_index] = int(none_token_id)
                        else:
                            distance = (matches - position).abs()
                            nearest = matches[distance == distance.min()].min()
                            labels[sample_index] = ids[row, nearest]
    output_rows = [i for i, type in enumerate(types) if type in LOGITS_TYPES]
    if output_rows:
        sub = torch.tensor(output_rows, device=out_device)
        rows = batch_idx[sub]
        log_probs = torch.log_softmax(logits_lookup(rows, probe[sub]).float(), dim=-1)
        observed = ids[rows, (probe[sub] + 1).clamp(max=ids.shape[1] - 1)]
        top = log_probs.topk(TOP_TOKEN_MAX_RANK).indices
        sub_types = [types[i] for i in output_rows]
        for type in set(sub_types):
            inner = torch.tensor([j for j, k in enumerate(sub_types) if k == type], device=out_device)
            outer = sub[inner]
            if type == "top_token":
                ranks = batch.ranks.to(out_device)[outer]
                labels[outer] = top[inner].gather(1, (ranks - 1).unsqueeze(1)).squeeze(1)
            elif type == "unpicked_token":
                first, second = top[inner, 0], top[inner, 1]
                labels[outer] = torch.where(first == observed[inner], second, first)
            elif type == "token_prob":
                percent = (log_probs[inner].gather(1, observed[inner].unsqueeze(1)).squeeze(1).exp() * 100).round()
                labels[outer] = percent.long().clamp(0, PERCENT_MAX)
            else:
                labels[outer] = entropy_label(type, log_probs[inner])
    if lens is not None:
        lens_owner, lens_layers, lens_logits, num_layers = lens
        lens_logits = lens_logits.to(out_device)
        for type in ("lens_top_token", "lens_token_prob", "lens_entropy", "lens_int_entropy"):
            entries = [j for j, i in enumerate(lens_owner) if types[i] == type]
            if not entries:
                continue
            ent = torch.tensor(entries, device=out_device)
            samples = torch.tensor([lens_owner[j] for j in entries], device=out_device)
            log_probs = torch.log_softmax(lens_logits[ent].float(), dim=-1)
            if type == "lens_top_token":
                ranks = batch.ranks.to(out_device)[samples]
                top = log_probs.topk(TOP_TOKEN_MAX_RANK).indices
                labels[samples] = top.gather(1, (ranks - 1).unsqueeze(1)).squeeze(1)
            elif type == "lens_token_prob":
                observed = ids[batch_idx[samples], (probe[samples] + 1).clamp(max=ids.shape[1] - 1)]
                percent = (log_probs.gather(1, observed.unsqueeze(1)).squeeze(1).exp() * 100).round()
                labels[samples] = percent.long().clamp(0, PERCENT_MAX)
            else:
                labels[samples] = entropy_label(type, log_probs)
        depth_entries = [j for j, i in enumerate(lens_owner) if types[i] == "decision_depth"]
        if depth_entries:
            grid = torch.tensor(depth_entries, device=out_device).view(-1, num_layers)
            depth_samples = torch.tensor(sorted({lens_owner[j] for j in depth_entries}), device=out_device)
            lens_argmax = lens_logits[grid].float().argmax(-1)
            final_argmax = logits_lookup(batch_idx[depth_samples], probe[depth_samples]).float().argmax(-1)
            agree = (lens_argmax == final_argmax.unsqueeze(1)).int()
            trailing = agree.flip(1).cumprod(1).sum(1)
            labels[depth_samples] = torch.where(trailing > 0, num_layers - trailing, torch.full_like(trailing, -1))
    return labels.detach()


def model_forward(model, batch, head_columns=None, lens_request=None):
    kwargs = {} if head_columns is None else {"head_columns": head_columns}
    if lens_request is not None:
        kwargs["lens_request"] = lens_request
    if batch.position_ids is not None:
        kwargs["position_ids"] = batch.position_ids
    return model(batch.input_ids, **kwargs)


class HeadLookup:
    def __init__(self, logits, head_columns, seq_len):
        self.logits = logits
        self.slot = torch.full((seq_len,), head_columns.numel(), dtype=torch.long, device=logits.device)
        self.slot[head_columns.to(logits.device)] = torch.arange(head_columns.numel(), device=logits.device)

    def index(self, rows, cols):
        return rows.to(self.logits.device), self.slot[cols.to(self.logits.device)]

    def __call__(self, rows, cols):
        return self.logits[self.index(rows, cols)]


def subset_lookup(logits, head_columns, seq_len):
    return HeadLookup(logits, head_columns, seq_len)


def entropy_label(type: str, log_probs: torch.Tensor) -> torch.Tensor:
    bits = -(log_probs.exp() * log_probs).sum(-1) / math.log(2)
    if type in INT_ENTROPY_TYPES:
        return bits.round().long().clamp(0, ENTROPY_MAX_BITS)
    return (bits * 10).round().long().clamp(0, ENTROPY_MAX_DECIBITS)


def answer_end_positions(ids, batch_idx, readout, last, labels, types, encoding, end_token_id):
    labels_host, readout_host, last_host, rows_host = (
        labels.tolist(),
        readout.tolist(),
        last.tolist(),
        batch_idx.tolist(),
    )
    rows, expected = [], []
    for i, (type, label, start, end) in enumerate(zip(types, labels_host, readout_host, last_host)):
        if label < 0 or start < 0:
            continue
        answer = answer_tokens(type, label, encoding)
        closing = answer_closing(type, encoding)
        if start + len(answer) + len(closing) > end:
            continue
        rows.append(rows_host[i])
        expected.append((start, answer, closing))
    if not rows:
        return [], [], []
    flat_rows = [row for row, (_, answer, closing) in zip(rows, expected) for _ in answer + closing]
    flat_columns = [start + 1 + k for start, answer, closing in expected for k in range(len(answer) + len(closing))]
    written = ids[torch.tensor(flat_rows, device=ids.device), torch.tensor(flat_columns, device=ids.device)].tolist()
    keep_rows, keep_columns, targets, offset = [], [], [], 0
    for row, (start, answer, closing) in zip(rows, expected):
        width = len(answer) + len(closing)
        if written[offset : offset + width] == answer + closing:
            for k, token in enumerate([*closing, end_token_id]):
                keep_rows.append(row)
                keep_columns.append(start + len(answer) + k)
                targets.append(int(token))
        offset += width
    return keep_rows, keep_columns, targets


def batch_losses(
    model,
    batch,
    answer_token_ids,
    lm_weight,
    digit_token_ids=None,
    digit_span=None,
    minimal_head=None,
    extra_positions=None,
    digit_reduction="sum",
    lm_segments=("user_surface_sentence", "user_pc_question"),
    dot_token_id=None,
    value_token_ids=None,
    expert_token_ids=None,
    num_lens_layers=None,
    depth_digits=None,
    label_override=None,
    num_experts_per_tok=None,
    none_token_id=None,
    scalar_target_width=0.0,
    digit_supervision="correct_prefix",
    drop_probe_label_mismatch=False,
    end_token_id=None,
    answer_close_token_id=None,
    answer_vocab="full",
):
    ids, probe = batch.input_ids, batch.probe
    turn_end = lm_weight > 0 and TURN_END_SEGMENT in lm_segments
    if turn_end and end_token_id is None:
        raise ValueError(f"the {TURN_END_SEGMENT} segment needs the template's turn-end token")
    targets_lm = sorted(
        {
            (row, token)
            for name in lm_segments
            if name != TURN_END_SEGMENT
            for row, start, end in batch.segments[name]
            for token in range(max(start, 1), end)
        }
    )
    emitted = batch.emitted
    types = batch.types
    encoding = {
        "digit_token_ids": digit_token_ids,
        "max_digits": digit_span,
        "depth_digits": depth_digits,
        "value_token_ids": value_token_ids,
        "dot_token": dot_token_id,
        "answer_token_ids": answer_token_ids,
        "expert_token_ids": expert_token_ids,
        "answer_close_token": answer_close_token_id,
    }
    fused_percent = value_token_ids is not None
    answer_grid_width = 1
    if digit_token_ids is not None:
        if emitted is None:
            raise ValueError("digit answers need the emitted-digit counts from collate")
        if digit_span is None:
            raise ValueError("digit answers need the answer width from the dataset encoding")
        for type in set(types):
            width = answer_width(type, encoding) if type != "decision_depth" or depth_digits is not None else 0
            answer_grid_width = max(answer_grid_width, width + (0 if fused_percent else 1))
    if turn_end:
        answer_grid_width += 1 + max(len(answer_closing(type, encoding)) for type in set(types))
    if minimal_head is None:
        minimal_head = getattr(model, "accepts_head_columns", False)

    lens_request = None
    lens_owner: list[int] = []
    lens_layer_list: list[int] = []
    if any(type in LENS_TYPES for type in types):
        if not getattr(model, "accepts_lens_request", False):
            raise ValueError("lens question types need a model that accepts lens requests")
        if "decision_depth" in types and num_lens_layers is None:
            raise ValueError("decision_depth needs the layer count from the dataset encoding")
        layer_host = batch.layer.tolist()
        depth_layers = 0 if num_lens_layers is None else int(num_lens_layers)
        for i, type in enumerate(types):
            if type == "decision_depth":
                lens_layer_list.extend(range(depth_layers))
                lens_owner.extend([i] * depth_layers)
            elif type in LENS_TYPES:
                lens_layer_list.append(layer_host[i])
                lens_owner.append(i)
        owner_index = torch.tensor(lens_owner, device=ids.device)
        lens_request = (
            torch.tensor(lens_layer_list, device=ids.device),
            batch.rows.to(ids.device)[owner_index],
            batch.probe.to(ids.device)[owner_index],
        )

    span_rows = [row for row, _ in targets_lm] if lm_weight > 0 else []
    span_cols = [token - 1 for _, token in targets_lm] if lm_weight > 0 else []
    head_columns = None
    if minimal_head:
        grid = batch.readout.clamp(min=0).unsqueeze(-1) + torch.arange(answer_grid_width, device=ids.device)
        needed = [grid.clamp(max=ids.shape[1] - 1).reshape(-1)]
        if any(type in LOGITS_TYPES or type == "decision_depth" for type in types):
            needed.append(batch.probe.to(ids.device))
        if span_cols:
            needed.append(torch.tensor(span_cols, device=ids.device))
        if extra_positions is not None:
            needed.append(extra_positions[1].to(ids.device))
        head_columns = torch.unique(torch.cat(needed))

    logits, routing_scores = model_forward(model, batch, head_columns=head_columns, lens_request=lens_request)
    out_device = logits.device
    if head_columns is None:
        head_columns = torch.arange(ids.shape[1], device=out_device)
    logits_lookup = subset_lookup(logits, head_columns, ids.shape[1])
    readout = batch.readout.to(out_device)
    batch_idx = batch.rows.to(out_device)
    ids_out = ids.to(out_device)
    lens = (lens_owner, lens_layer_list, model.lens_logits, num_lens_layers) if lens_owner else None
    labels = compute_labels(
        model,
        batch,
        routing_scores,
        logits_lookup,
        ids_out,
        batch_idx,
        probe.to(out_device),
        out_device,
        lens=lens,
        num_experts_per_tok=num_experts_per_tok,
        none_token_id=none_token_id,
    )
    if label_override is not None:
        labels = label_override.to(out_device)
    readout_valid = readout >= 0
    if drop_probe_label_mismatch and batch.probe_labels is not None:
        probe_labels = batch.probe_labels.to(out_device)
        readout_valid &= ~((probe_labels >= 0) & (labels != probe_labels))
    last = batch.last.to(out_device)

    intro_losses = torch.zeros(len(types), device=out_device)
    correct = torch.zeros(len(types), dtype=torch.bool, device=out_device)

    if digit_token_ids is None:
        routing_table = answer_token_ids
    elif expert_token_ids is not None:
        routing_table = expert_token_ids
    else:
        routing_table = value_token_ids if fused_percent else None
    single_rows = [
        i
        for i, type in enumerate(types)
        if (type in EXPERT_ID_TYPES and routing_table is not None)
        or type in SINGLE_TOKEN_TYPES
        or (fused_percent and type in WHOLE_VALUE_TYPES)
    ]
    if single_rows:
        sub = torch.tensor(single_rows, device=out_device)
        usable = readout_valid[sub] & (labels[sub] >= 0)
        labels_token = labels[sub].clamp(min=0)
        answer_logits = logits_lookup(batch_idx[sub], readout[sub].clamp(min=0)).float()
        scored = answer_logits
        for table, table_types in ((routing_table, EXPERT_ID_TYPES), (value_token_ids, WHOLE_VALUE_TYPES)):
            if table is None:
                continue
            in_table = torch.tensor([types[i] in table_types for i in single_rows], device=out_device)
            if in_table.any():
                labels_token[in_table] = table.to(out_device)[labels_token[in_table]]
                if answer_vocab == "answer":
                    scored = restrict_logits(scored, table.to(out_device), in_table)
        losses = torch.nn.functional.cross_entropy(scored, labels_token, reduction="none")
        if scalar_target_width > 0 and fused_percent:
            assert value_token_ids is not None
            soft = torch.tensor(
                [j for j, i in enumerate(single_rows) if types[i] in PERCENT_TYPES], dtype=torch.long, device=out_device
            )
            if soft.numel():
                losses[soft] = soft_value_loss(
                    scored[soft],
                    labels[sub][soft].clamp(min=0),
                    value_token_ids.to(out_device),
                    scalar_target_width,
                )
        intro_losses[sub] = losses * usable.float()
        correct[sub] = (answer_logits.argmax(-1) == labels_token) & usable
        readout_valid[sub] = usable

    prefix_valid = readout_valid.clone()
    numeric_vocab = None
    if answer_vocab == "answer" and digit_token_ids is not None:
        numeric_tokens = [*digit_token_ids, *([] if dot_token_id is None else [int(dot_token_id)])]
        numeric_vocab = torch.tensor(numeric_tokens, device=out_device)
        if fused_percent:
            assert value_token_ids is not None
            numeric_vocab = torch.unique(torch.cat([numeric_vocab, value_token_ids.to(out_device)]))
    spelled_types = [
        type
        for type in NUMERIC_TYPES
        if type in types
        and digit_token_ids is not None
        and not (type in EXPERT_ID_TYPES and routing_table is not None)
        and not (fused_percent and type not in EXPERT_ID_TYPES)
    ]
    for type in spelled_types:
        if scalar_target_width > 0 and type in SCALAR_TYPES:
            raise ValueError("soft scalar targets need a tokenizer that fuses whole numbers into one token each")
        sub = torch.tensor([i for i, k in enumerate(types) if k == type], device=out_device)
        form = number_format(type, encoding) if type != "decision_depth" or depth_digits is not None else None
        if form is None:
            raise ValueError(f"{type} answers need their digit width from the dataset encoding")
        usable = readout_valid[sub] & (labels[sub] >= 0)
        intro_losses[sub], correct[sub], whole = spelled_number_loss(
            logits_lookup,
            ids_out,
            batch_idx[sub],
            readout[sub],
            labels[sub].clamp(min=0),
            emitted.to(out_device)[sub],
            digit_token_ids,
            usable,
            form,
            dot_token_id,
            digit_reduction,
            last=last[sub],
            digit_supervision=digit_supervision,
            own_end=not turn_end,
            answer_vocab=numeric_vocab,
        )
        readout_valid[sub] = usable
        prefix_valid[sub] = usable & whole

    fused_decimal = [type for type in ENTROPY_TYPES if type in types and fused_percent]
    if fused_decimal and digit_token_ids is not None:
        assert value_token_ids is not None
        if dot_token_id is None:
            raise ValueError("decimal answers need the dot token from the dataset encoding")
        digit_lookup = torch.tensor(digit_token_ids, device=out_device)
        slots = [("value", 10), ("literal", int(dot_token_id)), ("digit", 1)]
        for type in fused_decimal:
            sub = torch.tensor([i for i, k in enumerate(types) if k == type], device=out_device)
            usable = readout_valid[sub] & (labels[sub] >= 0)
            intro_losses[sub], correct[sub], whole = fixed_slot_loss(
                logits_lookup,
                ids_out,
                batch_idx[sub],
                readout[sub],
                labels[sub],
                slots,
                digit_lookup,
                usable,
                last[sub],
                digit_reduction,
                value_lookup=value_token_ids.to(out_device),
                emitted=None if emitted is None else emitted.to(out_device)[sub],
                value_soft_width=scalar_target_width / 10,
                digit_supervision=digit_supervision,
                answer_vocab=numeric_vocab,
            )
            readout_valid[sub] = usable
            prefix_valid[sub] = usable & whole
    if digit_token_ids is None:
        for type in NUMERIC_TYPES:
            if type in EXPERT_ID_TYPES or (fused_percent and type not in ENTROPY_TYPES):
                continue
            rows = [i for i, k in enumerate(types) if k == type]
            if rows:
                readout_valid[torch.tensor(rows, device=out_device)] = False
    prefix_valid &= readout_valid

    lm_loss = torch.zeros((), device=out_device)
    end_rows, end_cols, end_targets = (
        answer_end_positions(ids_out, batch_idx, readout, last, labels, types, encoding, end_token_id)
        if turn_end
        else ([], [], [])
    )
    if span_cols or end_cols:
        rows_index = torch.tensor(span_rows + end_rows, device=out_device, dtype=torch.long)
        cols_index = torch.tensor(span_cols + end_cols, device=out_device, dtype=torch.long)
        targets = torch.cat(
            [
                ids_out[rows_index[: len(span_cols)], cols_index[: len(span_cols)] + 1],
                torch.tensor(end_targets, device=out_device, dtype=torch.long),
            ]
        )
        losses = torch.nn.functional.cross_entropy(
            logits_lookup(rows_index, cols_index).float(), targets, reduction="none"
        )
        lm_loss = losses.sum() / torch.tensor(float(losses.numel()), device=out_device).clamp(min=1.0)
    return intro_losses, lm_loss, correct, labels, routing_scores, logits_lookup, readout_valid, prefix_valid


def predicted_numbers(logits_lookup, ids, batch_idx, readout, last, labels, types, encoding):
    device = labels.device
    predicted = torch.full_like(labels, -1)
    scalar_rows = [i for i, type in enumerate(types) if type in SCALAR_TYPES]
    digit_ids = encoding.get("digit_token_ids")
    value_ids = encoding.get("value_token_ids")
    if not scalar_rows or (digit_ids is None and value_ids is None):
        return predicted
    digit_lookup = None if digit_ids is None else torch.as_tensor(digit_ids, device=device)
    value_lookup = None if value_ids is None else torch.as_tensor(value_ids, device=device)
    seq_len = ids.shape[1]

    def best(logits, table):
        return logits[..., table].argmax(-1)

    for type in sorted({types[i] for i in scalar_rows}):
        sub = torch.tensor([i for i in scalar_rows if types[i] == type], device=device)
        decimal = type in ENTROPY_TYPES
        label = labels[sub]
        start = readout[sub].to(device)
        end = last[sub].to(device)
        form = None if value_lookup is not None else number_format(type, encoding)
        integer = label.clamp(min=0) // 10 if decimal else label.clamp(min=0)
        max_int_digits = len(str(PERCENT_MAX)) if form is None else form.max_int_digits
        powers = 10 ** torch.arange(max_int_digits - 1, -1, -1, device=device)
        num_int_digits = (integer.unsqueeze(-1) >= powers).sum(-1).clamp(min=1)
        if form is None:
            width = 3 if decimal else 1
            num_tokens = torch.full_like(label, width)
        else:
            width = form.max_tokens
            num_tokens = num_int_digits + (2 if decimal else 0)
        offsets = torch.arange(width, device=device)
        columns = (start.unsqueeze(-1) + offsets).clamp(min=0, max=seq_len - 1)
        rows = batch_idx[sub].to(device).unsqueeze(-1).expand_as(columns)
        logits = logits_lookup(rows.reshape(-1), columns.reshape(-1)).float().view(len(sub), width, -1)
        if value_lookup is not None:
            value = best(logits[:, 0], value_lookup)
            if decimal:
                assert digit_lookup is not None
                value = value * 10 + best(logits[:, 2], digit_lookup)
        else:
            assert digit_lookup is not None
            digits = best(logits, digit_lookup)
            place = (num_int_digits.unsqueeze(-1) - 1 - offsets.unsqueeze(0)).clamp(min=0)
            weight = torch.where(offsets.unsqueeze(0) < num_int_digits.unsqueeze(-1), 10**place, 0)
            value = (digits * weight).sum(-1)
            if decimal:
                tenths = digits.gather(1, (num_int_digits + 1).clamp(max=width - 1).unsqueeze(-1)).squeeze(-1)
                value = value * 10 + tenths
        usable = (label >= 0) & (start >= 0) & (start + num_tokens - 1 <= end)
        predicted[sub] = torch.where(usable, value, torch.full_like(value, -1))
    return predicted


def sample_anchor_positions(valid_mask, sample_rows, readout, answer_widths=None, max_positions=256, always=None):
    mask = valid_mask.clone()
    widths = [1] * int(readout.numel()) if answer_widths is None else answer_widths
    for offset in range(max(widths)):
        cols = (readout + offset).clamp(min=0, max=mask.shape[1] - 1)
        keep_row = torch.tensor([offset < w for w in widths], device=mask.device)
        mask[sample_rows[keep_row], cols[keep_row]] = 0.0
    forced = torch.zeros_like(mask, dtype=torch.bool) if always is None else always.bool() & mask.bool()
    rows, cols = (mask.bool() & ~forced).nonzero(as_tuple=True)
    budget = max(max_positions - int(forced.sum()), 0)
    if rows.numel() > budget:
        keep = torch.randperm(rows.numel(), device=rows.device)[:budget]
        rows, cols = rows[keep], cols[keep]
    forced_rows, forced_cols = forced.nonzero(as_tuple=True)
    return torch.cat([forced_rows, rows]), torch.cat([forced_cols, cols])


def segment_sample_losses(batch, name, logits_lookup, num_samples):
    rows, cols, owners = [], [], []
    for (row, start, end), owner in zip(batch.segments[name], batch.segment_samples[name]):
        for position in range(max(start - 1, 0), end - 1):
            rows.append(row)
            cols.append(position)
            owners.append(owner)
    losses = torch.zeros(num_samples, device=batch.input_ids.device)
    if not rows:
        return losses
    rows_t = torch.tensor(rows, device=batch.input_ids.device)
    cols_t = torch.tensor(cols, device=batch.input_ids.device)
    owners_t = torch.tensor(owners, device=batch.input_ids.device)
    per_position = torch.nn.functional.cross_entropy(
        logits_lookup(rows_t, cols_t).float(), batch.input_ids[rows_t, cols_t + 1], reduction="none"
    )
    counts = torch.zeros(num_samples, device=losses.device).index_add_(0, owners_t, torch.ones_like(per_position))
    return losses.index_add_(0, owners_t, per_position) / counts.clamp(min=1)


def segment_positions(batch, names, samples=None):
    mask = torch.zeros(batch.input_ids.shape, dtype=torch.bool, device=batch.input_ids.device)
    for name in names:
        spans = batch.segments[name]
        if samples is not None:
            spans = [span for span, owner in zip(spans, batch.segment_samples[name]) if owner in samples]
        for row, start, end in spans:
            mask[row, max(start - 1, 0) : end - 1] = True
    return mask


def marker_positions(input_ids, close_id, end_ids):
    after_close = input_ids == close_id
    before_end = torch.zeros_like(after_close)
    before_end[:, :-1] = torch.isin(input_ids[:, 1:], end_ids.to(input_ids.device))
    return after_close | before_end


def _anchor_kl_terms(logits, ref_logits):
    log_p = torch.log_softmax(logits.float(), dim=-1)
    log_q = torch.log_softmax(ref_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def anchor_kl(logits, ref_logits):
    return _anchor_kl_terms(logits, ref_logits).mean()


ANCHOR_KL_CHUNK_ELEMENTS = 1 << 26


class _ChunkedAnchorKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, rows, cols, ref_logits, ref_rows, ref_cols, chunk):
        per_position = torch.empty(rows.numel(), dtype=torch.float32, device=logits.device)
        for start in range(0, rows.numel(), chunk):
            part = slice(start, start + chunk)
            per_position[part] = _anchor_kl_terms(
                logits[rows[part], cols[part]], ref_logits[ref_rows[part], ref_cols[part]].to(logits.device)
            )
        ctx.save_for_backward(logits, rows, cols, ref_logits, ref_rows, ref_cols)
        ctx.chunk = chunk
        return per_position.mean()

    @staticmethod
    def backward(ctx, grad):
        logits, rows, cols, ref_logits, ref_rows, ref_cols = ctx.saved_tensors
        grad_logits = torch.zeros_like(logits)
        share = grad.expand(rows.numel()) / rows.numel()
        for start in range(0, rows.numel(), ctx.chunk):
            part = slice(start, start + ctx.chunk)
            with torch.enable_grad():
                gathered = logits[rows[part], cols[part]].detach().requires_grad_()
                terms = _anchor_kl_terms(gathered, ref_logits[ref_rows[part], ref_cols[part]].to(logits.device))
                (part_grad,) = torch.autograd.grad(terms, gathered, share[part])
            grad_logits.index_put_((rows[part], cols[part]), part_grad, accumulate=True)
        return grad_logits, None, None, None, None, None, None


def anchor_kl_at(policy, reference, rows, cols):
    policy_rows, policy_cols = policy.index(rows, cols)
    ref_rows, ref_cols = reference.index(rows, cols)
    chunk = max(1, ANCHOR_KL_CHUNK_ELEMENTS // policy.logits.shape[-1])
    return _ChunkedAnchorKL.apply(policy.logits, policy_rows, policy_cols, reference.logits, ref_rows, ref_cols, chunk)


def reference_anchor_lookup(ref_model, collated, anchor_positions):
    ref_columns = None
    if anchor_positions is not None and getattr(ref_model, "accepts_head_columns", False):
        ref_columns = torch.unique(anchor_positions[1])
    with torch.no_grad():
        ref_logits, _ = model_forward(ref_model, collated, head_columns=ref_columns)
    seq_len = collated.input_ids.shape[1]
    if ref_columns is None:
        ref_columns = torch.arange(seq_len, device=ref_logits.device)
    return subset_lookup(ref_logits, ref_columns, seq_len)


def update_expert_bias(family, model, routing_scores, valid_mask, rate):
    with torch.no_grad():
        for layer_idx in family.moe_layers:
            moe = model.layers[layer_idx].feed_forward
            top = torch.topk(routing_scores[layer_idx].float(), family.num_experts_per_tok, dim=-1).indices
            flat = top[valid_mask.bool()].flatten()
            counts = torch.bincount(flat, minlength=family.num_experts).float()
            moe.expert_bias -= rate * torch.sign(counts - counts.mean()).to(moe.expert_bias.device)


def load_entropy_bits(family, routing_scores, valid_mask):
    entropies = []
    with torch.no_grad():
        for layer_idx in family.moe_layers:
            top = torch.topk(routing_scores[layer_idx].float(), family.num_experts_per_tok, dim=-1).indices
            counts = torch.bincount(top[valid_mask.bool()].flatten(), minlength=family.num_experts).float()
            p = counts / counts.sum().clamp(min=1.0)
            entropies.append(-torch.special.xlogy(p, p).sum() / math.log(2))
    return torch.stack(entropies).mean().item()
