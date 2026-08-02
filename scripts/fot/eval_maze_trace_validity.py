from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import wilson_accuracy_ci  # noqa: E402
from utils.fot.reproducibility import write_json  # noqa: E402
from utils.llm_baselines import (  # noqa: E402
    DEFAULT_MAZE_CELLS,
    DEFAULT_UPSCALE,
    build_maze_trace_samples,
)

START_RGB = (50, 220, 50)
GOAL_RGB = (50, 50, 220)
TRACE_RGB = (255, 50, 50)

Method = Literal["fot", "oracle"]
Rollout = Literal["sketcher", "controller"]


def _downsample_to_grid(img: Image.Image, *, upscale: int) -> np.ndarray:
    """Inverse of nearest-neighbor upscale used by `utils.llm_baselines.render_maze`."""
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    if h % upscale != 0 or w % upscale != 0:
        raise ValueError(f"Image shape {arr.shape} not divisible by upscale={upscale}")
    oy = upscale // 2
    ox = upscale // 2
    return arr[oy::upscale, ox::upscale, :].copy()


def _rgb_mask(grid_rgb: np.ndarray, rgb: Sequence[int]) -> np.ndarray:
    target = np.array(list(rgb), dtype=np.uint8).reshape(1, 1, 3)
    return np.all(grid_rgb == target, axis=-1)


def _find_color(grid_rgb: np.ndarray, rgb: Sequence[int]) -> Tuple[int, int]:
    mask = _rgb_mask(grid_rgb, rgb)
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError(f"Could not find color={list(rgb)} in grid")
    return int(ys[0]), int(xs[0])


def _neighbors_4n(y: int, x: int) -> List[Tuple[int, int]]:
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def _bfs_reachable(mask: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """Return boolean visited mask for 4-neighborhood BFS restricted to `mask`."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    sy, sx = start
    if not (0 <= sy < h and 0 <= sx < w) or not bool(mask[sy, sx]):
        return visited

    from collections import deque

    q = deque([(sy, sx)])
    visited[sy, sx] = True
    while q:
        y, x = q.popleft()
        for ny, nx in _neighbors_4n(y, x):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bool(mask[ny, nx]):
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def predict_maze_trace_yes_no_oracle(img: Image.Image, *, upscale: int) -> Tuple[str, float]:
    """Oracle image-based verifier for the maze-trace validity task (YES/NO).

    Uses only the highlighted trace connectivity/structure (plus the start/goal markers).
    """
    grid = _downsample_to_grid(img, upscale=upscale)
    start = _find_color(grid, START_RGB)
    goal = _find_color(grid, GOAL_RGB)

    trace = _rgb_mask(grid, TRACE_RGB)
    on_path = trace.copy()
    on_path[start] = True
    on_path[goal] = True

    # Connectivity: every highlighted cell must be connected to start, and goal must be reachable.
    visited = _bfs_reachable(on_path, start)
    if not bool(visited[goal]) or visited.sum() != on_path.sum():
        return "NO", 0.0

    # Degree constraints for a single simple chain from start to goal:
    # - start and goal are endpoints (degree 1)
    # - all other highlighted cells have degree 2
    deg = np.zeros_like(on_path, dtype=np.int32)
    ys, xs = np.where(on_path)
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = 0
        for ny, nx in _neighbors_4n(y, x):
            if 0 <= ny < on_path.shape[0] and 0 <= nx < on_path.shape[1] and bool(on_path[ny, nx]):
                d += 1
        deg[y, x] = d

    if deg[start] != 1 or deg[goal] != 1:
        return "NO", 0.0

    deg_vals = deg[on_path]
    if np.any((deg_vals != 1) & (deg_vals != 2)):
        return "NO", 0.0
    if int(np.sum(deg_vals == 1)) != 2:
        return "NO", 0.0

    return "YES", 1.0


def _extract_maze_from_image(
    img: Image.Image,
    *,
    upscale: int,
    wall_thr: int = 10,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], np.ndarray]:
    """Extract (grid, start, goal, trace_mask) from a rendered maze image.

    - grid: uint8 array (H,W), where 1=wall and 0=free.
    - start/goal: (y,x) in grid coordinates.
    - trace_mask: bool array (H,W) for highlighted trace pixels (red), excluding start/goal markers.
    """
    grid_rgb = _downsample_to_grid(img, upscale=upscale)
    walls = np.all(grid_rgb < wall_thr, axis=-1)
    grid = walls.astype(np.uint8)

    start = _find_color(grid_rgb, START_RGB)
    goal = _find_color(grid_rgb, GOAL_RGB)
    trace = _rgb_mask(grid_rgb, TRACE_RGB)
    return grid, start, goal, trace


def _f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    tgt_b = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred_b, tgt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(tgt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), tgt_b).sum())
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


def _dice_soft(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    inter = float(np.sum(p * t))
    denom = float(np.sum(p) + np.sum(t))
    return float((2.0 * inter) / (denom + eps)) if denom > 0 else 0.0


class FoTMazeSolver:
    def __init__(
        self,
        *,
        sketcher_path: Path,
        device: str,
        img_size: int = 64,
        max_steps: int = 180,
        rollout: Rollout = "sketcher",
        controller_path: Optional[Path] = None,
    ):
        import torch

        from utils.fot.checkpoint_utils import load_state_dict
        from utils.fot.models import MazeSketcher

        self.device = torch.device(device)
        self.img_size = int(img_size)
        self.max_steps = int(max_steps)
        self.rollout: Rollout = rollout

        sd = load_state_dict(str(sketcher_path), self.device)
        flow_dim = int(sd["inc.net.0.weight"].shape[0])
        sketcher = MazeSketcher(cond_ch=3, flow_dim=flow_dim).to(self.device)
        sketcher.load_state_dict(sd)
        sketcher.eval()
        for p in sketcher.parameters():
            p.requires_grad = False
        self.sketcher = sketcher

        self.controller = None
        self.action_steps = {0: 1, 1: 2, 2: 4, 3: 0}
        if self.rollout == "controller":
            if controller_path is None:
                raise ValueError("controller_path is required when rollout='controller'")
            from stable_baselines3 import PPO

            self.controller = PPO.load(str(controller_path), device=device)

    def rollout_trace(self, *, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
        import torch

        from utils.fot.maze_ops import bfs_shortest_path, build_cond, one_hot_point, resize_nn, segment_frames_from_path

        path = bfs_shortest_path(grid, start, goal)
        frames = segment_frames_from_path(path)
        frames_len = max(2, len(frames))

        cond = build_cond(grid, start, goal, self.img_size).unsqueeze(0).to(self.device)  # (1,3,H,W)
        goal_ch = one_hot_point(grid.shape, goal)
        goal_mask = resize_nn(torch.tensor(goal_ch).float(), self.img_size).to(self.device)  # (1,H,W)
        trace = torch.zeros((1, 1, self.img_size, self.img_size), device=self.device)

        progress = 0
        step_count = 0
        terminated = False
        truncated = False

        if self.rollout == "sketcher":
            # Deterministic rollout: advance one FM step per iteration until the end.
            while progress < frames_len - 1:
                t = torch.tensor([[progress / max(1, frames_len - 1)]], device=self.device)
                with torch.no_grad():
                    delta = self.sketcher(trace, cond, t)
                trace = (trace + delta).clamp(0.0, 1.0)
                progress += 1
            terminated = True
        else:
            if self.controller is None:
                raise RuntimeError("controller not loaded")
            while not (terminated or truncated):
                trace_np = trace[0].detach().cpu().numpy()  # (1,H,W)
                cond_np = cond[0].detach().cpu().numpy()  # (3,H,W)
                obs = np.concatenate([cond_np, trace_np], axis=0).astype(np.float32)  # (4,H,W)

                action, _ = self.controller.predict(obs, deterministic=True)
                n = self.action_steps.get(int(action), 1)

                for _ in range(int(n)):
                    if progress >= frames_len - 1:
                        break
                    t = torch.tensor([[progress / max(1, frames_len - 1)]], device=self.device)
                    with torch.no_grad():
                        delta = self.sketcher(trace, cond, t)
                    trace = (trace + delta).clamp(0.0, 1.0)
                    progress += 1

                goal_on_trace = (trace[0, 0] * goal_mask[0]).sum().item() > 0.5
                if progress >= frames_len - 1 or goal_on_trace:
                    terminated = True

                step_count += 1
                truncated = step_count >= self.max_steps

        trace_arr = trace[0, 0].detach().cpu().numpy().astype(np.float32)
        success = bool(terminated) and not bool(truncated)
        return trace_arr, success


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    auc: float
    seed: int
    maze_cells: int
    upscale: int
    method: Method
    threshold: float
    rollout: str
    calibrated: bool
    calibrate_seed: int
    calibrate_n: int
    calibrate_best_acc: float
    sketcher: str
    controller: str
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate maze trace validity (YES/NO) using FoT rollout or an oracle checker.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--maze-cells", type=int, default=DEFAULT_MAZE_CELLS)
    p.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE)
    p.add_argument("--method", type=str, default="fot", choices=["fot", "oracle"])
    p.add_argument("--rollout", type=str, default="sketcher", choices=["sketcher", "controller"])
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for mapping scores -> YES/NO (FoT method only).",
    )
    p.add_argument("--calibrate-seed", type=int, default=1, help="Seed for threshold calibration dataset (FoT only).")
    p.add_argument("--calibrate-n", type=int, default=0, help="If >0, calibrate threshold on this many samples.")
    p.add_argument("--sketcher", type=str, default="models/runs/2026-04-20_fm_maze_sketcher/sketcher.pth")
    p.add_argument("--controller", type=str, default="", help="Required if --rollout controller")
    p.add_argument("--device", type=str, default="cpu", help="Torch device for FoT rollout (default: cpu).")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--summary-out", type=str, default="", help="Optional standard run summary for suite aggregation.")
    p.add_argument("--save-results", action="store_true", help="Include per-sample predictions in the JSON output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    method: Method = str(args.method)  # type: ignore[assignment]
    # `build_maze_trace_samples` expects `random.Random`; seed compatibility:
    import random

    def build_samples(seed: int, n: int):
        py_rng = random.Random(int(seed))
        return build_maze_trace_samples(py_rng, int(n), maze_cells=int(args.maze_cells), upscale=int(args.upscale))

    y_true: List[int] = []
    y_score: List[float] = []
    y_pred: List[str] = []
    rows: List[Dict[str, Any]] = []

    sketcher_path = Path(str(args.sketcher))
    controller_path = Path(str(args.controller)) if str(args.controller) else None

    solver: Optional[FoTMazeSolver] = None
    if method == "fot":
        solver = FoTMazeSolver(
            sketcher_path=sketcher_path,
            device=str(args.device),
            img_size=64,
            max_steps=180,
            rollout=str(args.rollout),  # type: ignore[arg-type]
            controller_path=controller_path,
        )

    threshold = float(args.threshold)
    calibrated = False
    calibrate_best_acc = float("nan")
    if method == "fot" and int(args.calibrate_n) > 0:
        assert solver is not None
        calibrated = True

        val_scores: List[float] = []
        val_true: List[int] = []
        for img, gt in build_samples(int(args.calibrate_seed), int(args.calibrate_n)):
            gt_i = 1 if gt == "YES" else 0
            grid, start, goal, trace_grid = _extract_maze_from_image(img, upscale=int(args.upscale))
            trace_pred_64, _ = solver.rollout_trace(grid=grid, start=start, goal=goal)

            import torch

            from utils.fot.maze_ops import resize_nn

            trace_obs = trace_grid.astype(np.float32)
            trace_obs[start] = 1.0
            trace_obs[goal] = 1.0
            trace_obs_64 = resize_nn(torch.tensor(trace_obs).float(), 64).detach().cpu().numpy()  # (H,W)
            score = _dice_soft(trace_pred_64, trace_obs_64)

            val_scores.append(float(score))
            val_true.append(gt_i)

        # Pick the threshold that maximizes calibration accuracy (ties -> higher threshold).
        val_scores_a = np.asarray(val_scores, dtype=np.float32)
        val_true_a = np.asarray(val_true, dtype=np.int32)
        best_acc = -1.0
        best_th = threshold
        for th in np.unique(val_scores_a):
            pred = (val_scores_a >= th).astype(np.int32)
            acc = float(np.mean(pred == val_true_a))
            if acc > best_acc or (acc == best_acc and float(th) > float(best_th)):
                best_acc = acc
                best_th = float(th)
        threshold = float(best_th)
        calibrate_best_acc = float(best_acc)

    samples = build_samples(int(args.seed), int(args.n))

    t0 = time.time()
    for i, (img, gt) in enumerate(samples):
        gt_i = 1 if gt == "YES" else 0
        if method == "oracle":
            pred, score = predict_maze_trace_yes_no_oracle(img, upscale=int(args.upscale))
        else:
            assert solver is not None
            grid, start, goal, trace_grid = _extract_maze_from_image(img, upscale=int(args.upscale))
            trace_pred_64, _ = solver.rollout_trace(grid=grid, start=start, goal=goal)

            # Resize observed trace to match FoT internal resolution.
            import torch

            from utils.fot.maze_ops import resize_nn

            trace_obs = trace_grid.astype(np.float32)
            trace_obs[start] = 1.0
            trace_obs[goal] = 1.0
            trace_obs_64 = resize_nn(torch.tensor(trace_obs).float(), 64).detach().cpu().numpy()  # (H,W)

            score = _dice_soft(trace_pred_64, trace_obs_64)
            pred = "YES" if float(score) >= threshold else "NO"

        y_true.append(gt_i)
        y_pred.append(pred)
        y_score.append(float(score))
        if args.save_results:
            rows.append({"i": i, "gt": gt, "pred": pred, "score": float(score)})

    acc = float(np.mean([p == ("YES" if t == 1 else "NO") for p, t in zip(y_pred, y_true)])) if y_true else 0.0
    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")

    m = Metrics(
        n=int(args.n),
        accuracy=acc,
        auc=auc,
        seed=int(args.seed),
        maze_cells=int(args.maze_cells),
        upscale=int(args.upscale),
        method=method,
        threshold=float(threshold),
        rollout=str(args.rollout),
        calibrated=bool(calibrated),
        calibrate_seed=int(args.calibrate_seed),
        calibrate_n=int(args.calibrate_n),
        calibrate_best_acc=float(calibrate_best_acc),
        sketcher=str(sketcher_path),
        controller=str(controller_path) if controller_path is not None else "",
        device=str(args.device),
    )

    dt = time.time() - t0
    print(f"maze_trace[{m.method}]: n={m.n} acc={m.accuracy * 100:.2f}% auc={m.auc:.4f} (dt={dt:.1f}s)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"metrics": asdict(m)}
        if args.save_results:
            payload["results"] = rows
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Wrote:", out_path)

    if args.summary_out:
        correct = int(round(acc * len(y_true)))
        ci_low, ci_high = wilson_accuracy_ci(correct, len(y_true))
        write_json(Path(args.summary_out), {
            "experiment_name": f"maze_trace_{method}_{args.rollout}",
            "task": "maze_trace",
            "model": f"{method}_{args.rollout}",
            "seed": int(args.seed),
            "metrics": {"test": {"n": len(y_true), "accuracy": acc, "auc": auc,
                "accuracy_ci95_low": ci_low, "accuracy_ci95_high": ci_high,
                "accuracy_ci_method": "wilson_test_items", "threshold": threshold}},
            "elapsed_seconds": dt,
            "preliminary": False,
        })


if __name__ == "__main__":
    main()
