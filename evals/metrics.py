import numpy as np


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def bootstrap_ci(values, *, num_resamples: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(num_resamples, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def difference_ci(successes_a: int, n_a: int, successes_b: int, n_b: int) -> tuple[float, float] | None:
    ci_a, ci_b = wilson_ci(successes_a, n_a), wilson_ci(successes_b, n_b)
    if ci_a is None or ci_b is None:
        return None
    p_a, p_b = successes_a / n_a, successes_b / n_b
    lower = p_a - p_b - np.sqrt((p_a - ci_a[0]) ** 2 + (ci_b[1] - p_b) ** 2)
    upper = p_a - p_b + np.sqrt((ci_a[1] - p_a) ** 2 + (p_b - ci_b[0]) ** 2)
    return float(lower), float(upper)


def expected_calibration_error(confidence, correct, *, num_bins: int = 10) -> float | None:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.size == 0:
        return None
    edges = np.linspace(0, 1, num_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def brier(confidence, correct) -> float | None:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - correct) ** 2)) if confidence.size else None


def auroc(score, positive) -> float | None:
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    num_pos, num_neg = positive.sum(), (~positive).sum()
    if num_pos == 0 or num_neg == 0:
        return None
    order = score.argsort()
    ranks = np.empty(score.size, dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < score.size:
        j = i
        while j + 1 < score.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg))


def calibration_summary(confidence, correct) -> dict:
    return {
        "n": int(len(confidence)),
        "accuracy": mean(correct),
        "mean_confidence": mean(confidence),
        "ece": expected_calibration_error(confidence, correct),
        "brier": brier(confidence, correct),
        "auroc": auroc(confidence, correct),
    }
