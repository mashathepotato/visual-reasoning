from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _as_binary(values: Iterable[int]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1 or not np.all(np.isin(array, [0, 1])):
        raise ValueError("Expected a one-dimensional binary sequence")
    return array


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC with average ranks for ties; returns NaN for a single class."""
    y = _as_binary(labels)
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != y.shape or not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite and have the same shape as labels")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_classification_metrics(
    labels: Sequence[int],
    positive_probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float | int]:
    y = _as_binary(labels)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    if probabilities.shape != y.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and have the same shape as labels")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if y.size == 0:
        raise ValueError("Cannot score an empty sample")

    predictions = (probabilities >= float(threshold)).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    positive_recall = tp / max(1, tp + fn)
    negative_recall = tn / max(1, tn + fp)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1 - eps)
    log_loss = -float(np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    return {
        "n": int(y.size),
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(0.5 * (positive_recall + negative_recall)),
        "auc": float(binary_auc(y.tolist(), probabilities.tolist())),
        "brier": float(np.mean((probabilities - y) ** 2)),
        "log_loss": log_loss,
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("Invalid bootstrap settings")
    data = [float(value) for value in values]
    if any(not math.isfinite(value) for value in data):
        raise ValueError("values must be finite")
    rng = random.Random(int(seed))
    means: List[float] = []
    for _ in range(int(samples)):
        means.append(sum(rng.choice(data) for _ in data) / len(data))
    means.sort()
    alpha = 1.0 - float(confidence)
    lower = means[max(0, int((alpha / 2) * samples))]
    upper = means[min(samples - 1, int((1 - alpha / 2) * samples) - 1)]
    return float(lower), float(upper)
