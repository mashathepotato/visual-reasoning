from __future__ import annotations

"""
ViperGPT pipeline baseline for FoT synthetic benchmarks.

This script runs a *program-of-tools* baseline: an LLM first synthesizes a small Python
program (cached on disk) that can answer a given benchmark prompt by calling a restricted
set of vision tools implemented below. The synthesized program is then executed on many
images without further API calls.

Usage (requires an API key):
  export OPENAI_API_KEY=...
  .venv/bin/python benchmarks/vipergpt_pipeline_tests.py --seed 0 --out benchmarks/vipergpt_pipeline_seed0.json

If you already have cached programs, you can run without API access:
  .venv/bin/python benchmarks/vipergpt_pipeline_tests.py --no-api
"""

import argparse
import ast
import builtins
import json
import os
import random
import re
import signal
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: opencv-python. Install with `pip install opencv-python`.") from e

import numpy as np
from PIL import Image

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_baselines import (  # noqa: E402
    DEFAULT_MAZE_CELLS,
    DEFAULT_UPSCALE,
    MAZE_SOLVE_PROMPT,
    MAZE_TRACE_PROMPT,
    ROTATION_PROMPT,
    JsonlCache,
    build_3d_blocks_samples,
    build_colored_shapes_samples,
    build_maze_solve_instances,
    build_maze_trace_samples,
    build_tetris_samples,
    cache_key,
    eval_maze_solve,
    eval_maze_trace,
    eval_rotation,
)


# -----------------------
# Vision tool primitives
# -----------------------


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


def split_pair_image(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Split a side-by-side pair image into (left, right) crops.

    Works for `utils.llm_baselines.make_pair_image` style renders.
    """

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


def mask_nonblack(img: Image.Image, *, thr: int = 8) -> np.ndarray:
    """Binary mask of pixels that are not near-black."""
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr > thr, axis=-1).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def hu_moments(mask_u8: np.ndarray) -> np.ndarray:
    """Rotation-invariant Hu moments (log-scaled), shape=(7,)."""
    m = cv2.moments(mask_u8.astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.astype(np.float32)


def flip_lr(img: Image.Image) -> Image.Image:
    """Left-right mirror of an image."""
    op = getattr(Image, "FLIP_LEFT_RIGHT", Image.Transpose.FLIP_LEFT_RIGHT)
    return img.transpose(op)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two arrays after flattening."""
    a1 = np.asarray(a, dtype=np.float32).ravel()
    b1 = np.asarray(b, dtype=np.float32).ravel()
    if a1.shape != b1.shape:
        raise ValueError(f"Shape mismatch: {a1.shape} vs {b1.shape}")
    d = a1 - b1
    return float(np.sqrt(np.dot(d, d)))


def downsample_to_grid(img: Image.Image, *, upscale: int) -> np.ndarray:
    """
    Inverse of nearest-neighbor upscale used in `utils.llm_baselines.render_maze`.
    Returns an HxWx3 uint8 grid.
    """
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    if h % upscale != 0 or w % upscale != 0:
        raise ValueError(f"Image shape {arr.shape} not divisible by upscale={upscale}")
    oy = upscale // 2
    ox = upscale // 2
    return arr[oy::upscale, ox::upscale, :].copy()


def find_color(grid_rgb: np.ndarray, color: Sequence[int]) -> Tuple[int, int]:
    """Find the first (y,x) location of an exact RGB value in a grid image."""
    target = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask = np.all(grid_rgb == target, axis=-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"Could not find color={list(color)} in grid")
    return int(ys[0]), int(xs[0])


def wall_mask(grid_rgb: np.ndarray, *, thr: int = 10) -> np.ndarray:
    """Boolean mask of walls (near-black pixels)."""
    return np.all(grid_rgb < thr, axis=-1)


def bfs_path_4n(free: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """BFS shortest path in a 4-neighborhood grid. `free` is a boolean HxW mask."""
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
        return []

    path: List[Tuple[int, int]] = []
    cur: Tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def path_to_moves(path: Sequence[Tuple[int, int]]) -> str:
    """Convert a (y,x) path to a UDLR string."""
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


START_RGB: Tuple[int, int, int] = (50, 220, 50)
GOAL_RGB: Tuple[int, int, int] = (50, 50, 220)
TRACE_RGB: Tuple[int, int, int] = (255, 50, 50)


TOOL_DOCS_TEMPLATE = """
Available tools (call directly in Python):

Globals/constants:
- UPSCALE (int) = {upscale}
- START_RGB = {start_rgb}
- GOAL_RGB = {goal_rgb}
- TRACE_RGB = {trace_rgb}

Pair/shape tools:
- split_pair_image(image) -> (image_a, image_b)
- mask_nonblack(image, thr=8) -> np.ndarray uint8 mask (H,W) in {{0,1}}
- hu_moments(mask) -> np.ndarray float32 shape=(7,) (rotation-invariant moments; compare with L2 distance, not equality)
- l2(a, b) -> float (L2 distance between arrays)
- flip_lr(image) -> mirrored image

Maze/grid tools:
- downsample_to_grid(image, upscale=UPSCALE) -> np.ndarray uint8 (H,W,3)
- find_color(grid_rgb, [R,G,B]) -> (y,x) tuple
- wall_mask(grid_rgb, thr=10) -> np.ndarray bool (H,W) where True=wall (black)
- bfs_path_4n(free_mask, start, goal) -> list[(y,x)] shortest path (4-neighborhood)
- path_to_moves(path) -> UDLR string
""".strip()


# -----------------------
# Program synthesis + run
# -----------------------


SYSTEM_PROMPT = (
    "You are ViperGPT: you solve visual questions by writing a short Python program that calls tools.\n"
    "Rules:\n"
    "- Output ONLY valid Python code (no markdown, no backticks).\n"
    "- Do NOT import anything.\n"
    "- Do NOT access files or the network.\n"
    "- Define: def solve(image, question): ... and return a string answer.\n"
    "- For rotation questions return exactly 'SAME' or 'DIFFERENT'.\n"
    "- For maze trace validity return exactly 'YES' or 'NO'.\n"
    "- For maze solving return ONLY a string of moves using letters U,D,L,R.\n"
    "Tips:\n"
    "- Hu moments are floats: use a distance (e.g., L2) rather than exact equality.\n"
    "- Mirroring changes shapes: compare A vs B and flip_lr(A) vs B.\n"
    "- For mazes, use the provided START_RGB/GOAL_RGB/TRACE_RGB and UPSCALE.\n"
)


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", t)
        t = re.sub(r"\n```$", "", t).strip()
    return t


def _assert_safe_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Generated code must not import.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Generated code must not access dunder attributes.")
        if isinstance(node, ast.Name) and node.id in {"__import__", "__builtins__", "eval", "exec", "open"}:
            raise ValueError("Generated code uses a forbidden name.")


@contextmanager
def _time_limit(seconds: float):
    if seconds <= 0:
        yield
        return

    def handler(_signum, _frame):  # type: ignore[no-untyped-def]
        raise TimeoutError("Timed out")

    prev = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _restricted_builtins() -> Dict[str, Any]:
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        root = str(name).split(".", 1)[0]
        if root not in {"numpy", "cv2", "PIL"}:
            raise ImportError(f"Import blocked: {name!r}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    allowed = {
        "__import__": safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "Exception": Exception,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "ValueError": ValueError,
        "zip": zip,
    }
    return allowed


def _tool_globals(*, upscale: int) -> Dict[str, Any]:
    return {
        "np": np,
        "UPSCALE": int(upscale),
        "START_RGB": START_RGB,
        "GOAL_RGB": GOAL_RGB,
        "TRACE_RGB": TRACE_RGB,
        # tools
        "split_pair_image": split_pair_image,
        "mask_nonblack": mask_nonblack,
        "hu_moments": hu_moments,
        "l2": l2,
        "flip_lr": flip_lr,
        "downsample_to_grid": downsample_to_grid,
        "find_color": find_color,
        "wall_mask": wall_mask,
        "bfs_path_4n": bfs_path_4n,
        "path_to_moves": path_to_moves,
    }


def _compile_solve_fn(code: str, *, tool_globals: Dict[str, Any]) -> Callable[[Image.Image, str], str]:
    code = _strip_code_fences(code)
    _assert_safe_ast(code)

    g: Dict[str, Any] = {"__builtins__": _restricted_builtins()}
    g.update(tool_globals)
    l: Dict[str, Any] = {}
    exec(compile(code, "<vipergpt_program>", "exec"), g, l)  # noqa: S102
    solve = l.get("solve") or g.get("solve")
    if not callable(solve):
        raise ValueError("Generated code must define a callable `solve(image, question)`.")
    return solve  # type: ignore[return-value]


def _tool_docs(*, upscale: int) -> str:
    return TOOL_DOCS_TEMPLATE.format(
        upscale=int(upscale),
        start_rgb=START_RGB,
        goal_rgb=GOAL_RGB,
        trace_rgb=TRACE_RGB,
    )


def _program_prompt(*, question: str, tool_docs: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{tool_docs}\n\nQuestion:\n{question}\n"


class ViperGPTPipeline:
    def __init__(
        self,
        *,
        provider: str,
        upscale: int,
        model: str,
        temperature: float,
        max_output_tokens: int,
        program_cache_path: Path,
        timeout_s: float,
        no_api: bool,
        force_regen: bool,
    ):
        self.provider = provider.strip().lower()
        if self.provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider={provider!r}. Use 'openai' or 'anthropic'.")

        self.upscale = int(upscale)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_s = float(timeout_s)
        self.no_api = bool(no_api)
        self.force_regen = bool(force_regen)

        self.cache = JsonlCache(program_cache_path)
        self._solve_fns: Dict[str, Callable[[Image.Image, str], str]] = {}

        if self.no_api:
            self.client = None
            return

        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Set OPENAI_API_KEY (or pass --no-api if you have cached programs).")
            self.client = OpenAI()
        else:
            if Anthropic is None:
                raise RuntimeError("Missing dependency: anthropic. Install with `pip install anthropic`.")
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY (or pass --no-api if you have cached programs).")
            self.client = Anthropic()

    def _synth_program(self, question: str) -> str:
        if not self.client:
            raise RuntimeError(
                "No API client available (set OPENAI_API_KEY / ANTHROPIC_API_KEY, or run with --no-api + cached programs)."
            )

        tool_docs = _tool_docs(upscale=self.upscale)
        prompt = _program_prompt(question=question, tool_docs=tool_docs)
        key = cache_key(
            model=f"{self.provider}:{self.model}",
            prompt=prompt,
            image_bytes=b"",
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

        if not self.force_regen:
            cached = self.cache.get_response(key)
            if cached is not None:
                return cached

        if self.provider == "openai":
            resp = self.client.responses.create(  # type: ignore[union-attr]
                model=self.model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            text = getattr(resp, "output_text", None)
            if not isinstance(text, str) or not text.strip():
                text = str(resp)
        else:
            msg = self.client.messages.create(  # type: ignore[union-attr]
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            blocks = getattr(msg, "content", None)
            if isinstance(blocks, list):
                text_parts = [getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"]
                text = "\n".join([t for t in text_parts if isinstance(t, str) and t.strip()]).strip()
            else:
                text = ""
            if not text:
                text = str(msg)
        code = _strip_code_fences(text)

        self.cache.put(
            key=key,
            model=f"{self.provider}:{self.model}",
            prompt=prompt,
            response=code,
            meta={"max_output_tokens": self.max_output_tokens, "temperature": self.temperature},
        )
        return code

    def _get_solve(self, question: str) -> Callable[[Image.Image, str], str]:
        if question in self._solve_fns:
            return self._solve_fns[question]

        tool_docs = _tool_docs(upscale=self.upscale)
        prompt = _program_prompt(question=question, tool_docs=tool_docs)
        key = cache_key(
            model=f"{self.provider}:{self.model}",
            prompt=prompt,
            image_bytes=b"",
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

        code: Optional[str] = None
        if not self.force_regen:
            code = self.cache.get_response(key)

        if code is None:
            if self.no_api:
                raise RuntimeError(
                    f"Missing cached program for provider={self.provider} model={self.model}. "
                    "Re-run without --no-api to generate it."
                )
            code = self._synth_program(question)

        solve = _compile_solve_fn(code, tool_globals=_tool_globals(upscale=self.upscale))
        self._solve_fns[question] = solve
        return solve

    def __call__(self, prompt: str, image: Image.Image, *, max_output_tokens: int) -> str:  # noqa: ARG002
        solve = self._get_solve(prompt)
        try:
            with _time_limit(self.timeout_s):
                ans = solve(image, prompt)
        except Exception:
            return ""
        return str(ans).strip()


def _parse_tasks(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a ViperGPT (program-of-tools) baseline on FoT synthetic benchmarks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rotation", type=int, default=200)
    ap.add_argument("--n-maze-trace", type=int, default=200)
    ap.add_argument("--n-maze-solve", type=int, default=40)
    ap.add_argument("--maze-cells", type=int, default=DEFAULT_MAZE_CELLS)
    ap.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE)
    ap.add_argument(
        "--tasks",
        type=str,
        default="tetris,colors,3d,maze-trace,maze-solve",
        help="Comma-separated: tetris, colors, 3d, maze-trace, maze-solve",
    )
    ap.add_argument("--provider", type=str, default="openai", help="openai|anthropic")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=900, help="Max tokens for program synthesis.")
    ap.add_argument("--timeout-s", type=float, default=0.25, help="Per-sample execution timeout.")
    ap.add_argument("--program-cache", type=str, default="benchmarks/vipergpt_program_cache.jsonl")
    ap.add_argument("--force-regen", action="store_true", help="Regenerate programs even if cached.")
    ap.add_argument("--no-api", action="store_true", help="Do not call the API; require cached programs.")
    ap.add_argument("--out", type=str, default="", help="Optional path to write JSON results.")
    args = ap.parse_args()

    provider = args.provider.strip().lower()
    if not args.model:
        if provider == "anthropic":
            args.model = "claude-3-5-sonnet-20241022"
        else:
            args.model = "gpt-4o-mini"

    rng = random.Random(args.seed)
    tasks = set(_parse_tasks(args.tasks))

    vipergpt = ViperGPTPipeline(
        provider=provider,
        upscale=args.upscale,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        program_cache_path=Path(args.program_cache),
        timeout_s=args.timeout_s,
        no_api=args.no_api,
        force_regen=args.force_regen,
    )

    out: Dict[str, Any] = {"seed": args.seed, "tasks": sorted(tasks), "provider": provider, "model": args.model}

    if "tetris" in tasks:
        samples = build_tetris_samples(rng, args.n_rotation, upscale=args.upscale)
        out["tetris"] = eval_rotation("tetris", samples, llm_vision=vipergpt, prompt=ROTATION_PROMPT)

    if "colors" in tasks:
        samples = build_colored_shapes_samples(rng, args.n_rotation, upscale=args.upscale)
        out["colored_shapes"] = eval_rotation("colored_shapes", samples, llm_vision=vipergpt, prompt=ROTATION_PROMPT)

    if "3d" in tasks or "3d-blocks" in tasks:
        try:
            samples = build_3d_blocks_samples(REPO_ROOT, rng, args.n_rotation, upscale=args.upscale)
        except FileNotFoundError as e:
            out["3d_blocks_error"] = str(e)
        else:
            out["3d_blocks"] = eval_rotation("3d_blocks", samples, llm_vision=vipergpt, prompt=ROTATION_PROMPT)

    if "maze-trace" in tasks:
        samples = build_maze_trace_samples(rng, args.n_maze_trace, maze_cells=args.maze_cells, upscale=args.upscale)
        out["maze_trace"] = eval_maze_trace(samples, llm_vision=vipergpt, prompt=MAZE_TRACE_PROMPT)

    if "maze-solve" in tasks:
        instances = build_maze_solve_instances(rng, args.n_maze_solve, maze_cells=args.maze_cells)
        out["maze_solve"] = eval_maze_solve(instances, llm_vision=vipergpt, prompt=MAZE_SOLVE_PROMPT, upscale=args.upscale)

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
