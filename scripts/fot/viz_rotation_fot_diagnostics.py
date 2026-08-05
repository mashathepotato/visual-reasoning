from __future__ import annotations

"""Render flow-only and PPO-controlled rotation diagnostics."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fot.eval_rotation_fot import build_env  # noqa: E402
from utils.fot.rotation_dataset import RotationPairDataset  # noqa: E402
from utils.fot.rotation_ops import rotate_tensor  # noqa: E402


ACTION_NAMES = {
    0: "rotate -30",
    1: "rotate +30",
    2: "rotate -15",
    3: "rotate +15",
    4: "rotate -2",
    5: "rotate +2",
    6: "commit SAME",
    7: "commit DIFFERENT",
    8: "rotate 180",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("tetris", "colored"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test_ood_angle")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--fm-steps", type=int, default=6)
    parser.add_argument("--angle-step", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--no-corrector", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def tensor_image(value: torch.Tensor, *, residual: bool = False) -> Image.Image:
    value = value.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    value = value[:3].permute(1, 2, 0).numpy()
    if residual:
        magnitude = np.clip(value.mean(axis=2) * 4.0, 0.0, 1.0)
        value = np.stack((magnitude, 0.25 * magnitude, np.zeros_like(magnitude)), axis=2)
    array = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize((192, 192), Image.Resampling.NEAREST)


def titled(value: torch.Tensor, title: str, *, residual: bool = False) -> Image.Image:
    body = tensor_image(value, residual=residual)
    out = Image.new("RGB", (body.width, body.height + 38), "white")
    out.paste(body, (0, 38))
    ImageDraw.Draw(out).multiline_text((4, 3), title, fill="black", spacing=2)
    return out


def concat_row(images: List[Image.Image], pad: int = 8) -> Image.Image:
    width = sum(image.width for image in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (width, max(image.height for image in images)), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + pad
    return out


def stack_rows(rows: List[Image.Image], pad: int = 16) -> Image.Image:
    out = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows) + pad * (len(rows) - 1)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height + pad
    return out


def current_image(env) -> torch.Tensor:
    return env.current_source_obs if hasattr(env, "current_source_obs") else env.current_source


def make_env_args(args: argparse.Namespace) -> Namespace:
    root = REPO_ROOT / "models" / "runs" / "mps_paper_suite" / "fot" / args.task / f"seed{args.seed}"
    flow = root / "flow" / ("rotator.pth" if args.task == "tetris" else "best_checkpoint.pt")
    corrector = root / "flow" / "corrector.pth" if args.task == "tetris" and not args.no_corrector else None
    return Namespace(task=args.task, flow_checkpoint=flow, corrector=corrector,
                     fm_steps=args.fm_steps, max_episode_steps=args.max_episode_steps)


def select_examples(dataset: RotationPairDataset) -> List[Dict[str, Any]]:
    selected = []
    for wanted in (1, 0):
        for index in range(len(dataset)):
            sample = dataset[index]
            angle = abs(float(sample["angle_deg"])) % 180.0
            if int(sample["label"]) == wanted and 25.0 <= angle <= 155.0:
                selected.append(sample)
                break
    return selected


@torch.no_grad()
def flow_scan(env, sample: Dict[str, Any], angles: List[int]) -> Tuple[torch.Tensor, int, float]:
    env.reset(options={"pair": (sample["source"], sample["target"], int(sample["label"]) == 0)})
    best_image = current_image(env).clone()
    best_angle = 0
    best_error = float("inf")
    target = sample["target"].to(env.device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    for angle in angles:
        env._apply_rotation(float(angle))
        image = current_image(env)
        error = float(torch.mean((image - target) ** 2).cpu())
        if error < best_error:
            best_image, best_angle, best_error = image.clone(), int(angle), error
    return best_image, best_angle, best_error


@torch.no_grad()
def exact_scan(sample: Dict[str, Any]) -> Tuple[torch.Tensor, int, float]:
    source = sample["source"]
    target = sample["target"]
    best_image = source.clone()
    best_angle = 0
    best_error = float("inf")
    for angle in range(0, 360, 2):
        image = rotate_tensor(source, float(angle))
        error = float(torch.mean((image - target) ** 2))
        if error < best_error:
            best_image, best_angle, best_error = image.clone(), angle, error
    return best_image, best_angle, best_error


def policy_rollout(env, policy: PPO, sample: Dict[str, Any]) -> Dict[str, Any]:
    obs, _ = env.reset(options={"pair": (sample["source"], sample["target"], int(sample["label"]) == 0)})
    frames = [current_image(env).clone()]
    errors = [float(env._alignment_error().detach().cpu())]
    actions: List[int] = []
    terminated = truncated = False
    info: Dict[str, Any] = {}
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        action_i = int(action)
        obs, _, terminated, truncated, info = env.step(action_i)
        actions.append(action_i)
        frames.append(current_image(env).clone())
        errors.append(float(info.get("error", np.nan)))
    keep = sorted(set((0, 1, 2, max(0, len(frames) // 2), len(frames) - 1)))
    return {"frames": [frames[index] for index in keep], "frame_indices": keep,
            "frame_errors": [errors[index] for index in keep], "actions": actions,
            "terminated": bool(terminated), "truncated": bool(truncated), "info": info}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    env_args = make_env_args(args)
    env = build_env(env_args, device)
    policy = PPO.load(str(REPO_ROOT / "models" / "runs" / "mps_paper_suite" / "fot" / args.task /
                          f"seed{args.seed}" / "ppo_controller.zip"), device=device)
    dataset = RotationPairDataset(REPO_ROOT / "data" / "splits" / f"{args.task}_rotation_v1.json", args.split)
    rows: List[Image.Image] = []
    records: List[Dict[str, Any]] = []
    for sample in select_examples(dataset):
        label = "SAME" if int(sample["label"]) == 1 else "DIFFERENT"
        exact, exact_angle, exact_error = exact_scan(sample)
        flow, flow_angle, flow_error = flow_scan(env, sample, list(range(0, 360, args.angle_step)))
        residual = torch.abs(flow - sample["target"].to(device).unsqueeze(0))
        rows.append(concat_row([
            titled(sample["source"], f"{label}: source\nrender angle={sample['angle_deg']:.0f}"),
            titled(sample["target"], "target"),
            titled(exact, f"exact-search best\nangle={exact_angle}, mse={exact_error:.4f}"),
            titled(flow, f"FLOW best\nangle={flow_angle}, mse={flow_error:.4f}"),
            titled(residual, "|flow-target| x4", residual=True),
        ]))
        rollout = policy_rollout(env, policy, sample)
        policy_tiles = []
        for index, frame, error in zip(rollout["frame_indices"], rollout["frames"], rollout["frame_errors"]):
            action = "start" if index == 0 else ACTION_NAMES[rollout["actions"][index - 1]]
            policy_tiles.append(titled(frame, f"PPO step {index}: {action}\nmse={error:.4f}"))
        policy_tiles.append(titled(sample["target"], "target"))
        rows.append(concat_row(policy_tiles))
        records.append({"sample_id": sample["sample_id"], "label": label, "render_angle_deg": sample["angle_deg"],
                        "exact_best_angle": exact_angle, "exact_best_mse": exact_error,
                        "flow_best_angle": flow_angle, "flow_best_mse": flow_error,
                        "ppo_actions": [ACTION_NAMES[action] for action in rollout["actions"]],
                        "ppo_terminated": rollout["terminated"], "ppo_truncated": rollout["truncated"],
                        "ppo_final_info": rollout["info"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stack_rows(rows).save(args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(records, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
