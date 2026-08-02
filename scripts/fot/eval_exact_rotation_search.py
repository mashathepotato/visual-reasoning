from __future__ import annotations

"""Deterministic rendered-rotation search baseline on fixed pair manifests."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils.fot.metrics import binary_classification_metrics
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.rotation_dataset import RotationPairDataset
from utils.fot.rotation_ops import rotate_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact deterministic rotation search.")
    parser.add_argument("--task", choices=("tetris", "colored"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-step", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(dataset: RotationPairDataset, *, device: torch.device, angle_step: int,
             maximum: int | None, score_scale: float | None = None):
    angles = list(range(0, 360, int(angle_step)))
    rows: List[Dict[str, Any]] = []
    for index in range(min(len(dataset), maximum if maximum is not None else len(dataset))):
        sample = dataset[index]
        source = sample["source"].to(device); target = sample["target"].to(device)
        angle_tensor = torch.tensor(angles, device=device, dtype=source.dtype)
        source_batch = source.unsqueeze(0).expand(len(angles), -1, -1, -1)
        target_batch = target.unsqueeze(0).expand(len(angles), -1, -1, -1)
        original_rotations = rotate_tensor(source_batch, angle_tensor)
        flipped = torch.flip(source, dims=[2]).unsqueeze(0).expand(len(angles), -1, -1, -1)
        flipped_rotations = rotate_tensor(flipped, angle_tensor)
        original_errors = torch.mean((original_rotations - target_batch) ** 2, dim=(1, 2, 3)).cpu().tolist()
        flipped_errors = torch.mean((flipped_rotations - target_batch) ** 2, dim=(1, 2, 3)).cpu().tolist()
        best_original = min(original_errors); best_flipped = min(flipped_errors); raw_score = best_flipped - best_original
        best_angle = angles[int(np.argmin(original_errors))]
        true_alignment = (-float(sample["angle_deg"])) % 360.0
        angle_error = abs(((best_angle - true_alignment + 180.0) % 360.0) - 180.0)
        rows.append({"sample_id": sample["sample_id"], "base_id": sample["base_id"],
            "angle_deg": float(sample["angle_deg"]), "label": int(sample["label"]), "score": raw_score,
            "original_error": best_original, "flipped_error": best_flipped, "best_angle": best_angle,
            "angle_error_deg": angle_error if int(sample["label"]) == 1 else None})
    if score_scale is None:
        score_scale = max(float(np.std([row["score"] for row in rows])), 1e-6)
    probabilities = [float(1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -row["score"] / score_scale))))) for row in rows]
    for row, probability in zip(rows, probabilities):
        row["positive_probability"] = probability; row["prediction"] = int(probability >= 0.5)
    metrics = dict(binary_classification_metrics([row["label"] for row in rows], probabilities))
    same_angle_errors = [row["angle_error_deg"] for row in rows if row["angle_error_deg"] is not None]
    metrics["same_angle_mae_deg"] = float(np.mean(same_angle_errors))
    metrics["mean_min_original_mse"] = float(np.mean([row["original_error"] for row in rows]))
    return metrics, rows, score_scale


def main() -> None:
    args = parse_args(); device = torch.device(args.device); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (args.manifest or REPO_ROOT / "data" / "splits" / f"{args.task}_rotation_v1.json").resolve()
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1,
        "experiment": {"name": f"exact_rotation_search_{args.task}", "seed": 0, "device": args.device,
                       "preliminary": bool(args.preliminary)}, "arguments": resolved})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    started = time.perf_counter(); validation, validation_rows, scale = evaluate(
        RotationPairDataset(manifest, "validation"), device=device, angle_step=args.angle_step, maximum=args.max_eval)
    metrics: Dict[str, Any] = {"validation": validation}
    write_json(args.output_dir / "predictions_validation.json", {"predictions": validation_rows})
    for split in ("test_id", "test_ood_angle"):
        split_metrics, rows, _ = evaluate(RotationPairDataset(manifest, split), device=device,
                                          angle_step=args.angle_step, maximum=args.max_eval, score_scale=scale)
        metrics[split] = split_metrics; write_json(args.output_dir / f"predictions_{split}.json", {"predictions": rows})
    summary = {"experiment_name": f"exact_rotation_search_{args.task}", "task": f"{args.task}_rotation",
        "model": "deterministic_rendered_rotation_search", "seed": 0, "parameter_count": 0,
        "train_samples": 0, "metrics": metrics, "elapsed_seconds": time.perf_counter() - started,
        "preliminary": bool(args.preliminary), "protocol_notes": "No learning; searches both rotation and mirror hypotheses."}
    write_json(args.output_dir / "summary.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
