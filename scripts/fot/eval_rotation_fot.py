from __future__ import annotations

"""Evaluate the trained FoT flow+PPO rotation pipeline on fixed manifests."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO

from utils.fot.checkpoint_utils import load_state_dict
from utils.fot.dino_utils import create_dinov3
from utils.fot.envs_rotation import RotationEnvColorsFM, RotationEnvFM
from utils.fot.metrics import binary_classification_metrics
from utils.fot.models import CondEncoder, CorrectorUNet, FastRotator
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.rotation_dataset import RotationPairDataset
from utils.fot.torch_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FoT flow+PPO on fixed rotation manifests.")
    parser.add_argument("--task", choices=("tetris", "colored"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector", type=Path, default=None)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--fm-steps", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--max-eval", type=int, default=250)
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace, device: torch.device):
    if args.task == "tetris":
        state = load_state_dict(args.flow_checkpoint, device)
        flow_dim = int(state["inc.net.0.weight"].shape[0])
        backbone_dim = int(state["cond_proj.weight"].shape[1])
        flow = FastRotator(in_ch=1, out_ch=1, backbone_dim=backbone_dim, flow_dim=flow_dim).to(device)
        flow.load_state_dict(state)
        corrector = None
        if args.corrector is not None:
            corr_state = load_state_dict(args.corrector, device)
            base = int(corr_state["inc.net.0.weight"].shape[0])
            corrector = CorrectorUNet(in_ch=1, out_ch=1, base_ch=base).to(device)
            corrector.load_state_dict(corr_state)
            corrector.eval()
        dino = create_dinov3(device=device)
        return RotationEnvFM(image_shape=(3, 64, 64), fm_model=flow, corrector=corrector,
                             dino_model=dino, fm_steps=args.fm_steps, max_steps=args.max_episode_steps,
                             device=device)
    checkpoint = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    emb_dim = int(checkpoint.get("emb_dim", 256))
    flow_dim = int(checkpoint.get("flow_dim", 64))
    encoder = CondEncoder(in_ch=3, emb_dim=emb_dim).to(device)
    flow = FastRotator(in_ch=3, out_ch=3, backbone_dim=emb_dim, flow_dim=flow_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    flow.load_state_dict(checkpoint["model_state_dict"])
    return RotationEnvColorsFM(image_shape=(3, 64, 64), fm_model=flow, encoder=encoder,
                               fm_steps=args.fm_steps, max_steps=args.max_episode_steps, device=device)


def rollout(env, policy: PPO, dataset: RotationPairDataset, maximum: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(min(len(dataset), int(maximum))):
        sample = dataset[index]
        mirrored = int(sample["label"]) == 0
        obs, _ = env.reset(options={"pair": (sample["source"], sample["target"], mirrored)})
        terminated = truncated = False
        reward_sum = 0.0
        last_action = -1
        info: Dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            last_action = int(action)
            obs, reward, terminated, truncated, info = env.step(last_action)
            reward_sum += float(reward)
        committed = bool(terminated and last_action in (6, 7))
        rows.append({
            "sample_id": sample["sample_id"], "base_id": sample["base_id"],
            "angle_deg": float(sample["angle_deg"]), "label": int(sample["label"]),
            "best_error": float(info.get("best_error", info.get("error", math.inf))),
            "best_angle": float(info.get("best_angle", 0.0)), "steps": int(env.step_count),
            "total_rotation": float(info.get("total_rotation", 0.0)), "reward": reward_sum,
            "committed": committed, "policy_prediction": 1 if last_action == 6 else (0 if last_action == 7 else None),
        })
    return rows


def select_threshold(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    errors = np.asarray([row["best_error"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    candidates = np.unique(errors)
    best = (-1.0, 0.03)
    for threshold in candidates:
        accuracy = float(np.mean((errors < threshold).astype(np.int64) == labels))
        if accuracy > best[0]:
            best = accuracy, float(threshold)
    return best[1], max(float(np.std(errors)), 1e-6)


def score(rows: List[Dict[str, Any]], threshold: float, scale: float) -> Dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(1.0 / (1.0 + math.exp(max(-40.0, min(40.0, (row["best_error"] - threshold) / scale))))) for row in rows]
    metrics = dict(binary_classification_metrics(labels, probabilities))
    metrics.update({
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "commit_rate": float(np.mean([row["committed"] for row in rows])),
        "mean_total_rotation_deg": float(np.mean([row["total_rotation"] for row in rows])),
    })
    for row, probability in zip(rows, probabilities):
        row["positive_probability"] = probability
        row["prediction"] = int(probability >= 0.5)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (args.manifest or REPO_ROOT / "data" / "splits" / f"{args.task}_rotation_v1.json").resolve()
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1,
               "experiment": {"name": f"fot_flow_ppo_{args.task}", "seed": args.seed,
                              "device": args.device, "preliminary": bool(args.preliminary)},
               "arguments": {key: value for key, value in resolved.items() if key != "seed"}})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    started = time.perf_counter()
    env = build_env(args, device)
    policy = PPO.load(str(args.controller), device=device)
    validation_rows = rollout(env, policy, RotationPairDataset(manifest, "validation"), args.max_eval)
    threshold, scale = select_threshold(validation_rows)
    metrics: Dict[str, Any] = {}
    for split in ("validation", "test_id", "test_ood_angle"):
        rows = validation_rows if split == "validation" else rollout(
            env, policy, RotationPairDataset(manifest, split), args.max_eval)
        metrics[split] = score(rows, threshold, scale)
        write_json(args.output_dir / f"predictions_{split}.json", {"predictions": rows})
    summary = {"experiment_name": f"fot_flow_ppo_{args.task}", "task": f"{args.task}_rotation",
               "model": "flow_matching_plus_ppo", "seed": args.seed, "train_samples": None,
               "validation_selected_error_threshold": threshold, "metrics": metrics,
               "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary)}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
