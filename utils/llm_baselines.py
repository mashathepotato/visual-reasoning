from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore[no-redef]
        return x


DEFAULT_IMG_SIZE = 64
DEFAULT_MAZE_CELLS = 9
DEFAULT_UPSCALE = 4
DEFAULT_PAIR_PAD = 12


# --- Prompts (use these verbatim for evaluation) ---

ROTATION_PROMPT = (
    "Analyze the two objects in this image. Are they the exact same object just rotated, "
    "or are they different/mirrored? Answer strictly with one word: 'SAME' or 'DIFFERENT'."
)

MAZE_TRACE_PROMPT = (
    "Analyze this maze. Does the highlighted trace represent a valid, continuous path "
    "from the start to the goal without crossing any walls? Answer strictly with one word: 'YES' or 'NO'."
)

MAZE_SOLVE_PROMPT = (
    "Analyze this maze. Provide a valid path from the start to the goal without crossing any walls. "
    "Return ONLY a sequence of moves using the letters U, D, L, R (no spaces, no punctuation)."
)


# --- Small parsing helpers ---


def parse_choice(text: str, choices: Sequence[str]) -> Optional[str]:
    t = (text or "").strip().upper()
    m = re.search(r"[A-Z]+", t)
    if not m:
        return None
    tok = m.group(0)
    return tok if tok in choices else None


def extract_moves(text: str, max_len: int = 512) -> str:
    moves = re.findall(r"[UDLR]", (text or "").upper())
    return "".join(moves)[:max_len]


# --- Image helpers ---


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def cache_key(
    *,
    model: str,
    prompt: str,
    image_bytes: bytes,
    max_output_tokens: int,
    temperature: float,
) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\n")
    h.update(prompt.encode("utf-8"))
    h.update(b"\n")
    h.update(str(max_output_tokens).encode("utf-8"))
    h.update(b"\n")
    h.update(repr(float(temperature)).encode("utf-8"))
    h.update(b"\n")
    h.update(image_bytes)
    return h.hexdigest()


def load_jsonl_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            if isinstance(key, str):
                cache[key] = obj
    return cache


class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self._cache = load_jsonl_cache(path)

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def get_response(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if not entry:
            return None
        resp = entry.get("response")
        return resp if isinstance(resp, str) else None

    def put(self, *, key: str, model: str, prompt: str, response: str, meta: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {
            "key": key,
            "ts": time.time(),
            "model": model,
            "prompt": prompt,
            "response": response,
        }
        if meta:
            entry["meta"] = meta

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._cache[key] = entry


def to_u8_from_minus1_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = (x + 1.0) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


def gray_to_rgb(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 3 and img_u8.shape[0] == 1:
        img_u8 = img_u8[0]
    if img_u8.ndim != 2:
        raise ValueError(f"Expected HxW (or 1xHxW), got shape={img_u8.shape}")
    return np.stack([img_u8, img_u8, img_u8], axis=-1)


def resize_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def make_pair_image(left_rgb_u8: np.ndarray, right_rgb_u8: np.ndarray, pad: int = DEFAULT_PAIR_PAD) -> Image.Image:
    left = Image.fromarray(left_rgb_u8.astype(np.uint8), mode="RGB")
    right = Image.fromarray(right_rgb_u8.astype(np.uint8), mode="RGB")

    h = max(left.height, right.height)
    w = left.width + right.width + (3 * pad)
    out = Image.new("RGB", (w, h + 2 * pad), color=(255, 255, 255))
    out.paste(left, (pad, pad))
    out.paste(right, (2 * pad + left.width, pad))

    draw = ImageDraw.Draw(out)
    draw.text((pad + 2, 2), "A", fill=(0, 0, 0))
    draw.text((2 * pad + left.width + 2, 2), "B", fill=(0, 0, 0))
    return out


def pair_image_from_gray_arrays(
    x0_u8: np.ndarray,
    x1_u8: np.ndarray,
    *,
    upscale: int = DEFAULT_UPSCALE,
    pad: int = DEFAULT_PAIR_PAD,
) -> Image.Image:
    left = resize_nearest(gray_to_rgb(x0_u8), upscale)
    right = resize_nearest(gray_to_rgb(x1_u8), upscale)
    return make_pair_image(left, right, pad=pad)


def pair_image_from_rgb_float01(
    img0: np.ndarray,
    img1: np.ndarray,
    *,
    upscale: int = DEFAULT_UPSCALE,
    pad: int = DEFAULT_PAIR_PAD,
) -> Image.Image:
    x0_u8 = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
    x1_u8 = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
    left = resize_nearest(x0_u8, upscale)
    right = resize_nearest(x1_u8, upscale)
    return make_pair_image(left, right, pad=pad)


# --- Rotation datasets ---


CHIRAL_TETRIS_SHAPES = {
    "L": [(0, -1), (0, 0), (0, 1), (1, 1)],
    "J": [(0, -1), (0, 0), (0, 1), (-1, 1)],
    "S": [(0, 0), (1, 0), (0, 1), (-1, 1)],
    "Z": [(0, 0), (-1, 0), (0, 1), (1, 1)],
    "F": [(0, 0), (0, -1), (1, -1), (-1, 0), (0, 1)],
    "P": [(0, 0), (0, -1), (1, -1), (1, 0), (0, 1)],
}


def draw_tetris_shape_np(name: str, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    block_size = size // 8

    for dx, dy in CHIRAL_TETRIS_SHAPES[name]:
        x = center + (dx * block_size) - (block_size // 2)
        y = center + (dy * block_size) - (block_size // 2)
        cv2.rectangle(img, (x, y), (x + block_size, y + block_size), 255, -1)
    return img


def rotate_gray_pad_to_diag(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    diag = int(math.ceil(math.sqrt(h * h + w * w)))
    pad_h = max(0, diag - h)
    pad_w = max(0, diag - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_p = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    hp, wp = img_p.shape
    center = (wp / 2.0, hp / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(
        img_p,
        M,
        (wp, hp),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    y0 = (hp - h) // 2
    x0 = (wp - w) // 2
    return rot[y0 : y0 + h, x0 : x0 + w]


def sample_tetris_pair(
    rng: random.Random,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    mirror_prob: float = 0.5,
    angle_step: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    key = rng.choice(list(CHIRAL_TETRIS_SHAPES.keys()))
    target = draw_tetris_shape_np(key, img_size)

    mirrored = rng.random() < mirror_prob
    source = cv2.flip(target, 1) if mirrored else target.copy()

    angles = np.arange(0.0, 360.0, angle_step, dtype=np.float32)
    angle = float(rng.choice(list(angles)))
    source = rotate_gray_pad_to_diag(source, angle)

    gt = "DIFFERENT" if mirrored else "SAME"
    return source, target, gt


def random_colored_rectangles(h: int, w: int, num_shapes: int, rng: random.Random) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(num_shapes):
        color = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
        y0 = rng.randint(0, h - 8)
        x0 = rng.randint(0, w - 8)
        y1 = min(h, y0 + rng.randint(6, max(7, h // 2)))
        x1 = min(w, x0 + rng.randint(6, max(7, w // 2)))
        img[y0:y1, x0:x1, :] = color
    return img


def rotate_rgb(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w, _ = img.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )


def sample_colored_shapes_pair(
    rng: random.Random,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    num_shapes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, str]:
    img_a = random_colored_rectangles(img_size, img_size, num_shapes, rng)
    angle = float(rng.randint(0, 359))
    img_b = rotate_rgb(img_a, angle)

    same = rng.random() > 0.5
    if not same:
        img_b = cv2.flip(img_b, 1)

    gt = "SAME" if same else "DIFFERENT"
    return img_a, img_b, gt


def load_3d_blocks_pairs(repo_root: Path) -> List[Dict[str, Any]]:
    path = repo_root / "data" / "test_balanced.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it by running notebooks/pipeline_3d.ipynb or benchmarks/dinov3_3d_baseline.ipynb."
        )
    raw = np.load(str(path), allow_pickle=True)
    return list(raw)


# --- Maze dataset + scoring ---


ACTION_TO_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def generate_maze(cells_w: int, cells_h: int, rng: random.Random) -> np.ndarray:
    # Perfect maze with DFS backtracking. Returns grid with 1=wall, 0=free.
    grid = np.ones((cells_h * 2 + 1, cells_w * 2 + 1), dtype=np.uint8)
    visited = np.zeros((cells_h, cells_w), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    grid[1, 1] = 0

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cells_w and 0 <= ny < cells_h and not visited[ny, nx]:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = rng.choice(neighbors)
            grid[y * 2 + 1 + dy, x * 2 + 1 + dx] = 0
            grid[ny * 2 + 1, nx * 2 + 1] = 0
            visited[ny, nx] = True
            stack.append((nx, ny))
        else:
            stack.pop()
    return grid


def bfs_shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    prev = {start: None}

    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            break
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0 and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                q.append((ny, nx))

    if goal not in prev:
        return []

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def is_trace_valid(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    trace: List[Tuple[int, int]],
) -> bool:
    if not trace:
        return False
    if trace[0] != start or trace[-1] != goal:
        return False
    h, w = grid.shape
    for (y, x) in trace:
        if not (0 <= y < h and 0 <= x < w):
            return False
        if grid[y, x] != 0:
            return False
    for (y0, x0), (y1, x1) in zip(trace, trace[1:]):
        if abs(y0 - y1) + abs(x0 - x1) != 1:
            return False
    return True


def render_maze(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    trace: Optional[List[Tuple[int, int]]] = None,
    upscale: int = DEFAULT_UPSCALE,
) -> Image.Image:
    h, w = grid.shape
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    img[grid == 1] = 0

    if trace is not None:
        for (y, x) in trace:
            if 0 <= y < h and 0 <= x < w:
                img[y, x] = np.array([255, 50, 50], dtype=np.uint8)

    sy, sx = start
    gy, gx = goal
    img[sy, sx] = np.array([50, 220, 50], dtype=np.uint8)
    img[gy, gx] = np.array([50, 50, 220], dtype=np.uint8)

    img_big = resize_nearest(img, upscale)
    return Image.fromarray(img_big, mode="RGB")


def verify_moves(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], moves: str) -> bool:
    pos = start
    h, w = grid.shape
    for m in moves:
        if m not in ACTION_TO_DELTA:
            return False
        dy, dx = ACTION_TO_DELTA[m]
        ny, nx = pos[0] + dy, pos[1] + dx
        if not (0 <= ny < h and 0 <= nx < w):
            return False
        if grid[ny, nx] != 0:
            return False
        pos = (ny, nx)
    return pos == goal


@dataclass
class MazeInstance:
    grid: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]


def make_maze_instance(rng: random.Random, *, maze_cells: int = DEFAULT_MAZE_CELLS) -> MazeInstance:
    grid = generate_maze(maze_cells, maze_cells, rng)
    start = (1, 1)
    goal = (grid.shape[0] - 2, grid.shape[1] - 2)
    return MazeInstance(grid=grid, start=start, goal=goal)


def make_trace_sample(
    rng: random.Random,
    *,
    maze_cells: int = DEFAULT_MAZE_CELLS,
    upscale: int = DEFAULT_UPSCALE,
) -> Tuple[Image.Image, str]:
    inst = make_maze_instance(rng, maze_cells=maze_cells)
    path = bfs_shortest_path(inst.grid, inst.start, inst.goal)
    if not path:
        # Very rare; regenerate.
        return make_trace_sample(rng, maze_cells=maze_cells, upscale=upscale)

    valid = rng.random() < 0.5
    if valid:
        trace = path
        gt = "YES"
    else:
        mode = rng.choice(["prefix", "gap", "wall"])
        trace = list(path)
        if mode == "prefix":
            k = rng.randint(2, max(3, len(trace) - 2))
            trace = trace[:k]
        elif mode == "gap":
            if len(trace) > 6:
                i = rng.randint(2, len(trace) - 4)
                trace = trace[:i] + trace[i + 2 :]
            else:
                trace = trace[: max(2, len(trace) // 2)]
        else:  # wall
            inserted = False
            for _ in range(50):
                i = rng.randint(1, len(trace) - 2)
                y, x = trace[i]
                candidates = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                rng.shuffle(candidates)
                for ny, nx in candidates:
                    if 0 <= ny < inst.grid.shape[0] and 0 <= nx < inst.grid.shape[1] and inst.grid[ny, nx] == 1:
                        trace.insert(i + 1, (ny, nx))
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                trace = trace[: max(2, len(trace) // 2)]

        gt = "NO"

    # Sanity check labeling.
    if is_trace_valid(inst.grid, inst.start, inst.goal, trace):
        gt = "YES"

    img = render_maze(inst.grid, inst.start, inst.goal, trace=trace, upscale=upscale)
    return img, gt


# --- Sample builders (deterministic given RNG) ---


def build_tetris_samples(
    rng: random.Random,
    n: int,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    upscale: int = DEFAULT_UPSCALE,
    pad: int = DEFAULT_PAIR_PAD,
) -> List[Tuple[Image.Image, str]]:
    samples: List[Tuple[Image.Image, str]] = []
    for _ in range(n):
        src, tgt, gt = sample_tetris_pair(rng, img_size=img_size)
        img = pair_image_from_gray_arrays(src, tgt, upscale=upscale, pad=pad)
        samples.append((img, gt))
    return samples


def build_colored_shapes_samples(
    rng: random.Random,
    n: int,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    upscale: int = DEFAULT_UPSCALE,
    pad: int = DEFAULT_PAIR_PAD,
) -> List[Tuple[Image.Image, str]]:
    samples: List[Tuple[Image.Image, str]] = []
    for _ in range(n):
        a, b, gt = sample_colored_shapes_pair(rng, img_size=img_size)
        img = pair_image_from_rgb_float01(a, b, upscale=upscale, pad=pad)
        samples.append((img, gt))
    return samples


def build_3d_blocks_samples(
    repo_root: Path,
    rng: random.Random,
    n: int,
    *,
    upscale: int = DEFAULT_UPSCALE,
    pad: int = DEFAULT_PAIR_PAD,
) -> List[Tuple[Image.Image, str]]:
    raw = load_3d_blocks_pairs(repo_root)
    rng.shuffle(raw)
    raw = raw[: min(n, len(raw))]

    samples: List[Tuple[Image.Image, str]] = []
    for s in raw:
        x0_u8 = to_u8_from_minus1_1(np.asarray(s["x0"]))
        x1_u8 = to_u8_from_minus1_1(np.asarray(s["x1"]))
        gt = "SAME" if s.get("label") == "same" else "DIFFERENT"
        img = pair_image_from_gray_arrays(x0_u8, x1_u8, upscale=upscale, pad=pad)
        samples.append((img, gt))
    return samples


def build_maze_trace_samples(
    rng: random.Random,
    n: int,
    *,
    maze_cells: int = DEFAULT_MAZE_CELLS,
    upscale: int = DEFAULT_UPSCALE,
) -> List[Tuple[Image.Image, str]]:
    return [make_trace_sample(rng, maze_cells=maze_cells, upscale=upscale) for _ in range(n)]


def build_maze_solve_instances(
    rng: random.Random,
    n: int,
    *,
    maze_cells: int = DEFAULT_MAZE_CELLS,
) -> List[MazeInstance]:
    return [make_maze_instance(rng, maze_cells=maze_cells) for _ in range(n)]


# --- Evaluation loops ---


class VisionFn(Protocol):
    def __call__(self, prompt: str, image: Image.Image, *, max_output_tokens: int) -> str: ...


def eval_rotation(
    name: str,
    images_and_labels: List[Tuple[Image.Image, str]],
    *,
    llm_vision: VisionFn,
    prompt: str = ROTATION_PROMPT,
    max_output_tokens: int = 8,
) -> Dict[str, Any]:
    results = []
    for img, gt in tqdm(images_and_labels, desc=f"{name} (rotation)"):
        raw = llm_vision(prompt, img, max_output_tokens=max_output_tokens)
        pred = parse_choice(raw, ["SAME", "DIFFERENT"])
        correct = pred == gt
        results.append({"gt": gt, "pred": pred, "raw": raw, "correct": correct})
    acc = float(np.mean([r["correct"] for r in results])) if results else 0.0
    return {"name": name, "n": len(results), "accuracy": acc, "results": results}


def eval_maze_trace(
    images_and_labels: List[Tuple[Image.Image, str]],
    *,
    llm_vision: VisionFn,
    prompt: str = MAZE_TRACE_PROMPT,
) -> Dict[str, Any]:
    results = []
    for img, gt in tqdm(images_and_labels, desc="Maze trace (YES/NO)"):
        raw = llm_vision(prompt, img, max_output_tokens=8)
        pred = parse_choice(raw, ["YES", "NO"])
        correct = pred == gt
        results.append({"gt": gt, "pred": pred, "raw": raw, "correct": correct})
    acc = float(np.mean([r["correct"] for r in results])) if results else 0.0
    return {"name": "maze_trace", "n": len(results), "accuracy": acc, "results": results}


def eval_maze_solve(
    instances: List[MazeInstance],
    *,
    llm_vision: VisionFn,
    prompt: str = MAZE_SOLVE_PROMPT,
    upscale: int = DEFAULT_UPSCALE,
) -> Dict[str, Any]:
    results = []
    for inst in tqdm(instances, desc="Maze solve (path)"):
        img = render_maze(inst.grid, inst.start, inst.goal, trace=None, upscale=upscale)
        raw = llm_vision(prompt, img, max_output_tokens=512)
        moves = extract_moves(raw, max_len=512)
        ok = verify_moves(inst.grid, inst.start, inst.goal, moves)
        results.append({"moves": moves, "raw": raw, "success": ok, "len": len(moves)})
    success = float(np.mean([r["success"] for r in results])) if results else 0.0
    return {"name": "maze_solve", "n": len(results), "success_rate": success, "results": results}

