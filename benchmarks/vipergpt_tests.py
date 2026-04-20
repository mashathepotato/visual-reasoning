from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: opencv-python. Install with `pip install opencv-python`.") from e

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_baselines import (  # noqa: E402
    DEFAULT_MAZE_CELLS,
    DEFAULT_UPSCALE,
    MAZE_SOLVE_PROMPT,
    MAZE_TRACE_PROMPT,
    ROTATION_PROMPT,
    build_3d_blocks_samples,
    build_colored_shapes_samples,
    build_maze_solve_instances,
    build_maze_trace_samples,
    build_tetris_samples,
    eval_maze_solve,
    eval_maze_trace,
    eval_rotation,
)


def _segments_from_indices(idxs: np.ndarray) -> List[Tuple[int, int]]:
    if idxs.size == 0:
        return []
    segs: List[Tuple[int, int]] = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for x in idxs[1:]:
        xi = int(x)
        if xi != prev + 1:
            segs.append((start, prev))
            start = xi
        prev = xi
    segs.append((start, prev))
    return segs


def _crop_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def split_pair_image(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    arr = np.asarray(img.convert("RGB"))
    non_white = np.any(arr < 250, axis=-1)
    h, w = non_white.shape

    col_sum = non_white.sum(axis=0)
    active_cols = np.where(col_sum > (0.2 * h))[0]
    segs = _segments_from_indices(active_cols)
    if len(segs) < 2:
        mid = w // 2
        left = img.crop((0, 0, mid, img.height))
        right = img.crop((mid, 0, img.width, img.height))
        return left, right

    segs = sorted(segs, key=lambda s: (s[1] - s[0]), reverse=True)[:2]
    segs = sorted(segs, key=lambda s: s[0])

    crops: List[Image.Image] = []
    for x0, x1 in segs:
        sub = non_white[:, x0 : x1 + 1]
        row_sum = sub.sum(axis=1)
        active_rows = np.where(row_sum > (0.2 * (x1 - x0 + 1)))[0]
        if active_rows.size == 0:
            y0, y1 = 0, h - 1
        else:
            y0, y1 = int(active_rows.min()), int(active_rows.max())
        crops.append(img.crop((int(x0), y0, int(x1) + 1, y1 + 1)))

    return crops[0], crops[1]


def _mask_nonblack(img: Image.Image, *, thr: int = 8) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr > thr, axis=-1).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _hu_vec(mask_u8: np.ndarray) -> np.ndarray:
    m = cv2.moments(mask_u8, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.astype(np.float32)


def predict_same_different(pair_img: Image.Image) -> str:
    a, b = split_pair_image(pair_img)
    a_mask = _mask_nonblack(a)
    b_mask = _mask_nonblack(b)

    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return "DIFFERENT"

    hu_a = _hu_vec(a_mask)
    hu_b = _hu_vec(b_mask)

    a_flip = a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    hu_a_flip = _hu_vec(_mask_nonblack(a_flip))

    d_same = float(np.linalg.norm(hu_a - hu_b))
    d_flip = float(np.linalg.norm(hu_a_flip - hu_b))
    return "SAME" if d_same <= d_flip else "DIFFERENT"


def _downsample_nearest(img: Image.Image, *, upscale: int) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    if h % upscale != 0 or w % upscale != 0:
        raise ValueError(f"Image shape {arr.shape} not divisible by upscale={upscale}")
    oy = upscale // 2
    ox = upscale // 2
    return arr[oy::upscale, ox::upscale, :]


def _find_single_color(grid_rgb: np.ndarray, color: Sequence[int], *, name: str) -> Tuple[int, int]:
    target = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask = np.all(grid_rgb == target, axis=-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"Could not find {name} color {list(color)} in grid")
    return int(ys[0]), int(xs[0])


def predict_maze_trace_yes_no(img: Image.Image, *, upscale: int) -> str:
    grid = _downsample_nearest(img, upscale=upscale)

    wall = np.all(grid < 10, axis=-1)
    start = _find_single_color(grid, (50, 220, 50), name="start")
    goal = _find_single_color(grid, (50, 50, 220), name="goal")

    is_red = np.all(grid == np.array([255, 50, 50], dtype=np.uint8), axis=-1)
    trace = is_red.copy()
    trace[start] = True
    trace[goal] = True

    if np.any(trace & wall):
        return "NO"

    def neigh(y: int, x: int) -> List[Tuple[int, int]]:
        out = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < trace.shape[0] and 0 <= nx < trace.shape[1] and trace[ny, nx]:
                out.append((ny, nx))
        return out

    start_deg = len(neigh(*start))
    goal_deg = len(neigh(*goal))
    if start_deg != 1 or goal_deg != 1:
        return "NO"

    ys, xs = np.where(trace)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if (y, x) in (start, goal):
            continue
        if len(neigh(int(y), int(x))) != 2:
            return "NO"

    q: deque[Tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            break
        for ny, nx in neigh(y, x):
            if (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))
    return "YES" if goal in seen else "NO"


def predict_maze_moves(img: Image.Image, *, upscale: int) -> str:
    grid = _downsample_nearest(img, upscale=upscale)
    wall = np.all(grid < 10, axis=-1)
    free = ~wall
    start = _find_single_color(grid, (50, 220, 50), name="start")
    goal = _find_single_color(grid, (50, 50, 220), name="goal")

    q: deque[Tuple[int, int]] = deque([start])
    prev: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}

    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            break
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < free.shape[0] and 0 <= nx < free.shape[1] and free[ny, nx] and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                q.append((ny, nx))

    if goal not in prev:
        return ""

    path: List[Tuple[int, int]] = []
    cur: Tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    moves = []
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        dy, dx = y1 - y0, x1 - x0
        if (dy, dx) == (-1, 0):
            moves.append("U")
        elif (dy, dx) == (1, 0):
            moves.append("D")
        elif (dy, dx) == (0, -1):
            moves.append("L")
        elif (dy, dx) == (0, 1):
            moves.append("R")
        else:
            return ""
    return "".join(moves)


class ViperGPTToolsBaseline:
    """
    A ViperGPT-style baseline: answers are produced by executing a small set of
    vision tools (no learned weights, no fine-tuning).

    It plugs into the existing `utils.llm_baselines` eval loops by implementing:
        fn(prompt: str, image: PIL.Image, max_output_tokens: int) -> str
    """

    def __init__(self, *, upscale: int = DEFAULT_UPSCALE):
        self.upscale = int(upscale)

    def __call__(self, prompt: str, image: Image.Image, *, max_output_tokens: int) -> str:  # noqa: ARG002
        p = (prompt or "").strip()
        if p == ROTATION_PROMPT:
            return predict_same_different(image)
        if p == MAZE_TRACE_PROMPT:
            return predict_maze_trace_yes_no(image, upscale=self.upscale)
        if p == MAZE_SOLVE_PROMPT:
            return predict_maze_moves(image, upscale=self.upscale)
        raise ValueError(f"Unknown prompt (len={len(p)}). Pass one of the known prompts from utils.llm_baselines.")


def _parse_tasks(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a ViperGPT-style tools baseline on FoT synthetic benchmarks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rotation", type=int, default=400, help="Samples per rotation dataset (tetris/colors/3d).")
    ap.add_argument("--n-maze-trace", type=int, default=300)
    ap.add_argument("--n-maze-solve", type=int, default=60)
    ap.add_argument("--maze-cells", type=int, default=DEFAULT_MAZE_CELLS)
    ap.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE)
    ap.add_argument(
        "--tasks",
        type=str,
        default="tetris,colors,3d,maze-trace,maze-solve",
        help="Comma-separated: tetris, colors, 3d, maze-trace, maze-solve",
    )
    ap.add_argument("--out", type=str, default="", help="Optional path to write JSON results.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tasks = set(_parse_tasks(args.tasks))
    baseline = ViperGPTToolsBaseline(upscale=args.upscale)

    out: Dict[str, Any] = {"seed": args.seed, "tasks": sorted(tasks)}

    if "tetris" in tasks:
        samples = build_tetris_samples(rng, args.n_rotation, upscale=args.upscale)
        out["tetris"] = eval_rotation("tetris", samples, llm_vision=baseline)

    if "colors" in tasks:
        samples = build_colored_shapes_samples(rng, args.n_rotation, upscale=args.upscale)
        out["colored_shapes"] = eval_rotation("colored_shapes", samples, llm_vision=baseline)

    if "3d" in tasks or "3d-blocks" in tasks or "blocks3d" in tasks:
        try:
            samples = build_3d_blocks_samples(REPO_ROOT, rng, args.n_rotation, upscale=args.upscale)
        except FileNotFoundError as e:
            out["3d_blocks_error"] = str(e)
        else:
            out["3d_blocks"] = eval_rotation("3d_blocks", samples, llm_vision=baseline)

    if "maze-trace" in tasks:
        samples = build_maze_trace_samples(rng, args.n_maze_trace, maze_cells=args.maze_cells, upscale=args.upscale)
        out["maze_trace"] = eval_maze_trace(samples, llm_vision=baseline)

    if "maze-solve" in tasks:
        instances = build_maze_solve_instances(rng, args.n_maze_solve, maze_cells=args.maze_cells)
        out["maze_solve"] = eval_maze_solve(instances, llm_vision=baseline, upscale=args.upscale)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    summary = {
        "tetris_acc": out.get("tetris", {}).get("accuracy"),
        "colors_acc": out.get("colored_shapes", {}).get("accuracy"),
        "3d_acc": out.get("3d_blocks", {}).get("accuracy"),
        "maze_trace_acc": out.get("maze_trace", {}).get("accuracy"),
        "maze_solve_success": out.get("maze_solve", {}).get("success_rate"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
