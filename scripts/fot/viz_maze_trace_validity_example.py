from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_baselines import DEFAULT_MAZE_CELLS, DEFAULT_UPSCALE, build_maze_trace_samples  # noqa: E402

# Reuse helpers/solver from the evaluator.
from scripts.fot.eval_maze_trace_validity import (  # noqa: E402
    FoTMazeSolver,
    _dice_soft,
    _extract_maze_from_image,
)


def _nearest_resize(img: Image.Image, *, scale: int) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), resample=Image.Resampling.NEAREST)


def _render_base_maze_64(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
    """Return 64x64 RGB uint8 image (walls black, free white, start/goal colored)."""
    import torch

    from utils.fot.maze_ops import one_hot_point, resize_nn

    walls = grid.astype(np.float32)  # 1=wall, 0=free
    walls_64 = (resize_nn(torch.tensor(walls).float(), 64).detach().cpu().numpy()[0] > 0.5)
    rgb = np.ones((64, 64, 3), dtype=np.uint8) * 255
    rgb[walls_64] = 0

    start_64 = (resize_nn(torch.tensor(one_hot_point(grid.shape, start)).float(), 64).detach().cpu().numpy()[0] > 0.5)
    goal_64 = (resize_nn(torch.tensor(one_hot_point(grid.shape, goal)).float(), 64).detach().cpu().numpy()[0] > 0.5)
    rgb[start_64] = np.array([50, 220, 50], dtype=np.uint8)
    rgb[goal_64] = np.array([50, 50, 220], dtype=np.uint8)
    return rgb


def _overlay_trace(
    base_rgb: np.ndarray,
    *,
    trace_64: np.ndarray,
    color: Tuple[int, int, int],
    alpha_scale: float,
) -> np.ndarray:
    """Alpha-blend a soft trace (0..1) onto base RGB."""
    out = base_rgb.astype(np.float32).copy()
    t = np.clip(np.asarray(trace_64, dtype=np.float32), 0.0, 1.0)
    a = np.clip(t * float(alpha_scale), 0.0, 1.0)[..., None]
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = (1.0 - a) * out + a * col
    return np.clip(out, 0, 255).astype(np.uint8)


def _title(img: Image.Image, text: str) -> Image.Image:
    pad = 18
    out = Image.new("RGB", (img.width, img.height + pad), color=(255, 255, 255))
    out.paste(img, (0, pad))
    d = ImageDraw.Draw(out)
    d.text((4, 2), text, fill=(0, 0, 0))
    return out


def _concat_row(imgs: List[Image.Image], *, pad: int = 8) -> Image.Image:
    w = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width + pad
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a FoT maze trace-validity example and save to disk.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--i", type=int, default=0, help="Sample index within the generated set.")
    p.add_argument("--n", type=int, default=32, help="How many samples to generate (must be > i).")
    p.add_argument("--maze-cells", type=int, default=DEFAULT_MAZE_CELLS)
    p.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE)
    p.add_argument("--sketcher", type=str, required=True)
    p.add_argument("--rollout", choices=("sketcher", "controller"), default="sketcher")
    p.add_argument("--controller", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threshold", type=float, default=0.37118908762931824)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import random
    import torch

    if args.i < 0 or args.i >= args.n:
        raise SystemExit("--i must satisfy 0 <= i < n")

    samples = build_maze_trace_samples(
        random.Random(int(args.seed)),
        int(args.n),
        maze_cells=int(args.maze_cells),
        upscale=int(args.upscale),
    )
    img_in, gt = samples[int(args.i)]

    grid, start, goal, trace_grid = _extract_maze_from_image(img_in, upscale=int(args.upscale))

    solver = FoTMazeSolver(
        sketcher_path=Path(str(args.sketcher)),
        device=str(args.device),
        img_size=64,
        rollout=str(args.rollout),
        controller_path=Path(args.controller) if args.controller else None,
    )
    trace_pred_64, _ = solver.rollout_trace(grid=grid, start=start, goal=goal)

    from utils.fot.maze_ops import resize_nn

    trace_obs = trace_grid.astype(np.float32)
    trace_obs[start] = 1.0
    trace_obs[goal] = 1.0
    trace_obs_64 = (
        resize_nn(torch.tensor(trace_obs).float(), 64).detach().cpu().numpy()[0].astype(np.float32)
    )

    score = _dice_soft(trace_pred_64, trace_obs_64)
    pred = "YES" if float(score) >= float(args.threshold) else "NO"

    # Panels.
    in_big = _nearest_resize(img_in, scale=4)
    in_big = _title(in_big, f"Input (gt={gt})")

    base_64 = _render_base_maze_64(grid, start, goal)
    pred_rgb = _overlay_trace(base_64, trace_64=trace_pred_64, color=(255, 230, 0), alpha_scale=0.9)
    pred_img = Image.fromarray(pred_rgb, mode="RGB")
    pred_img = _nearest_resize(pred_img, scale=4)
    pred_img = _title(pred_img, f"FoT {args.rollout} trace (yellow)")

    both_rgb = _overlay_trace(base_64, trace_64=trace_obs_64, color=(255, 50, 50), alpha_scale=0.9)
    both_rgb = _overlay_trace(both_rgb, trace_64=trace_pred_64, color=(255, 230, 0), alpha_scale=0.6)
    both_img = Image.fromarray(both_rgb, mode="RGB")
    both_img = _nearest_resize(both_img, scale=4)
    both_img = _title(both_img, f"Overlay (pred={pred}, score={score:.3f})")

    row = _concat_row([in_big, pred_img, both_img], pad=12)
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
