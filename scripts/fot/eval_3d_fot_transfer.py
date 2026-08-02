from __future__ import annotations

"""Zero-shot Ganis-Kievit transfer evaluation for a trained 2-D FoT flow."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils.fot.checkpoint_utils import load_state_dict
from utils.fot.dino_utils import create_dinov3, dino_embed_fm_gray64
from utils.fot.integrators import apply_heun_steps
from utils.fot.metrics import binary_classification_metrics
from utils.fot.models import CondEncoder, CorrectorUNet, FastRotator
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.supervised_models import count_parameters
from utils.fot.torch_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2-D FoT flow transfer on Ganis-Kievit blocks.")
    parser.add_argument("--source-model", choices=("tetris", "colored"), required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--angle-step", type=int, default=15)
    parser.add_argument("--fm-steps", type=int, default=6)
    parser.add_argument("--max-eval", type=int, default=78)
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device):
    if args.source_model == "tetris":
        state = load_state_dict(args.flow_checkpoint, device)
        flow = FastRotator(in_ch=1, out_ch=1, backbone_dim=int(state["cond_proj.weight"].shape[1]),
                           flow_dim=int(state["inc.net.0.weight"].shape[0])).to(device)
        flow.load_state_dict(state); flow.eval()
        corrector = None
        if args.corrector:
            state_c = load_state_dict(args.corrector, device)
            corrector = CorrectorUNet(in_ch=1, out_ch=1, base_ch=int(state_c["inc.net.0.weight"].shape[0])).to(device)
            corrector.load_state_dict(state_c); corrector.eval()
        return flow, create_dinov3(device=device), corrector
    checkpoint = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    encoder = CondEncoder(in_ch=3, emb_dim=int(checkpoint.get("emb_dim", 256))).to(device)
    flow = FastRotator(in_ch=3, out_ch=3, backbone_dim=int(checkpoint.get("emb_dim", 256)),
                       flow_dim=int(checkpoint.get("flow_dim", 64))).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"]); flow.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval(); flow.eval()
    return flow, encoder, None


@torch.no_grad()
def score_hypothesis(args, flow, conditioner, corrector, source: torch.Tensor, target: torch.Tensor, angles: List[int]) -> Tuple[float, int]:
    if args.source_model == "tetris":
        condition = dino_embed_fm_gray64(source, conditioner)
        low, high = -1.0, 1.0
    else:
        source = ((source + 1.0) * 0.5).repeat(1, 3, 1, 1)
        target = ((target + 1.0) * 0.5).repeat(1, 3, 1, 1)
        condition = conditioner(source)
        low, high = 0.0, 1.0
    best_error = float("inf"); best_angle = 0
    for angle in angles:
        predicted = apply_heun_steps(model=flow, x0=source, cond_emb=condition,
            target_angle_deg=torch.tensor([[float(angle)]], device=source.device), steps=args.fm_steps,
            clamp_min=low, clamp_max=high, corrector=corrector)
        error = float(torch.mean((predicted - target) ** 2).cpu())
        if error < best_error:
            best_error, best_angle = error, angle
    return best_error, best_angle


def main() -> None:
    args = parse_args(); set_seed(args.seed, deterministic=True); device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1,
        "experiment": {"name": f"fot_{args.source_model}_to_ganis3d", "seed": args.seed,
                       "device": args.device, "preliminary": bool(args.preliminary)},
        "arguments": {key: value for key, value in resolved.items() if key != "seed"}})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    flow, conditioner, corrector = load_models(args, device)
    rows_raw = list(np.load(REPO_ROOT / "data" / "test_balanced.npy", allow_pickle=True))[:args.max_eval]
    angles = list(range(0, 360, args.angle_step)); rows: List[Dict[str, Any]] = []; started = time.perf_counter()
    for index, item in enumerate(rows_raw):
        source = torch.as_tensor(item["x0"], dtype=torch.float32, device=device).unsqueeze(0)
        target = torch.as_tensor(item["x1"], dtype=torch.float32, device=device).unsqueeze(0)
        original_error, original_angle = score_hypothesis(args, flow, conditioner, corrector, source, target, angles)
        flipped_error, flipped_angle = score_hypothesis(args, flow, conditioner, corrector,
                                                          torch.flip(source, dims=[3]), target, angles)
        raw_score = flipped_error - original_error
        rows.append({"sample_id": f"ganis3d-{index:03d}-{item.get('name', '')}",
            "label": 1 if item.get("label") == "same" else 0,
            "prediction": int(original_error <= flipped_error), "score": raw_score,
            "original_error": original_error, "flipped_error": flipped_error,
            "best_angle": original_angle, "best_flipped_angle": flipped_angle})
    score_scale = max(float(np.std([row["score"] for row in rows])), 1e-6)
    for row in rows:
        row["positive_probability"] = float(1.0 / (1.0 + np.exp(-row["score"] / score_scale)))
    metrics = binary_classification_metrics([r["label"] for r in rows], [r["positive_probability"] for r in rows])
    write_json(args.output_dir / "predictions_test_ganis3d.json", {"predictions": rows})
    summary: Dict[str, Any] = {"experiment_name": f"fot_{args.source_model}_to_ganis3d", "task": "ganis3d",
        "model": f"{args.source_model}_rotation_flow_transfer", "seed": args.seed,
        "parameter_count": count_parameters(flow) + count_parameters(conditioner),
        "train_samples": None, "metrics": {"test_ganis3d": metrics},
        "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary),
        "protocol_notes": "zero-shot 2-D-to-3-D transfer; checked-in test identities overlap legacy training data"}
    write_json(args.output_dir / "summary.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
