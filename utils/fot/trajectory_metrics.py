"""Process-sensitive metrics for continuous visual trajectories."""

from __future__ import annotations

from typing import Dict, Sequence

import torch


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0)


def maze_temporal_metrics(
    predicted_frames: Sequence[torch.Tensor],
    true_frames: torch.Tensor,
    *,
    activation_threshold: float = 0.5,
    leakage_threshold: float = 0.1,
    monotonicity_tolerance: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Return one value per example for timing, leakage, and prefix quality.

    ``predicted_frames`` is a length-T sequence of ``(B,1,H,W)`` tensors and
    ``true_frames`` is ``(B,T,1,H,W)``. The final trace is treated as the set of
    pixels that may eventually activate; activation outside it is off-path.
    """
    prediction = torch.stack(list(predicted_frames), dim=1).clamp(0.0, 1.0)
    if prediction.shape != true_frames.shape:
        raise ValueError(f"Trajectory shape mismatch: {prediction.shape} vs {true_frames.shape}")
    truth = true_frames.clamp(0.0, 1.0)
    pred_binary = prediction >= float(activation_threshold)
    true_binary = truth >= 0.5
    final_path = true_binary[:, -1]
    batch, frame_count = prediction.shape[:2]

    intersections = (pred_binary & true_binary).sum(dim=(2, 3, 4)).float()
    unions = (pred_binary | true_binary).sum(dim=(2, 3, 4)).float()
    prefix_iou = _safe_ratio(intersections, unions)
    intermediate = prefix_iou[:, 1:-1] if frame_count > 2 else prefix_iou[:, 1:]

    future_intensity_terms = []
    future_low_terms = []
    future_high_terms = []
    for step in range(1, frame_count - 1):
        future = final_path & ~true_binary[:, step]
        count = future.sum(dim=(1, 2, 3)).float()
        future_intensity_terms.append(_safe_ratio((prediction[:, step] * future).sum(dim=(1, 2, 3)), count))
        future_low_terms.append(
            _safe_ratio(((prediction[:, step] >= leakage_threshold) & future).sum(dim=(1, 2, 3)).float(), count)
        )
        future_high_terms.append(
            _safe_ratio((pred_binary[:, step] & future).sum(dim=(1, 2, 3)).float(), count)
        )

    step_numbers = torch.arange(frame_count, device=prediction.device).view(1, frame_count, 1, 1, 1)
    never = torch.full_like(step_numbers.expand_as(pred_binary), frame_count)
    true_first = torch.where(true_binary, step_numbers, never).amin(dim=1)
    pred_first = torch.where(pred_binary, step_numbers, never).amin(dim=1)
    path_count = final_path.sum(dim=(1, 2, 3)).float()
    activation_time_mae = _safe_ratio(
        ((pred_first - true_first).abs().float() * final_path).sum(dim=(1, 2, 3)), path_count
    ) / max(1, frame_count - 1)
    premature = _safe_ratio(((pred_first < true_first) & final_path).sum(dim=(1, 2, 3)).float(), path_count)
    missed = _safe_ratio(((pred_first >= frame_count) & final_path).sum(dim=(1, 2, 3)).float(), path_count)

    path_mask_over_time = final_path[:, None].expand(-1, frame_count - 1, -1, -1, -1)
    decreases = (prediction[:, 1:] + monotonicity_tolerance) < prediction[:, :-1]
    monotonicity_violations = _safe_ratio(
        (decreases & path_mask_over_time).sum(dim=(1, 2, 3, 4)).float(),
        path_count * max(1, frame_count - 1),
    )
    off_path = ~final_path
    off_path_count = off_path.sum(dim=(1, 2, 3)).float()
    off_path_rate = _safe_ratio(
        ((prediction[:, -1] >= leakage_threshold) & off_path).sum(dim=(1, 2, 3)).float(), off_path_count
    )

    def mean_terms(terms):
        if not terms:
            return torch.zeros(batch, device=prediction.device)
        return torch.stack(terms, dim=1).mean(dim=1)

    return {
        "intermediate_prefix_iou": intermediate.mean(dim=1),
        "activation_time_mae_normalized": activation_time_mae,
        "premature_activation_rate": premature,
        "missed_path_activation_rate": missed,
        "future_path_mean_intensity": mean_terms(future_intensity_terms),
        "future_path_activation_rate_at_0_1": mean_terms(future_low_terms),
        "future_path_activation_rate_at_0_5": mean_terms(future_high_terms),
        "monotonicity_violation_rate": monotonicity_violations,
        "off_path_activation_rate_at_0_1": off_path_rate,
    }
