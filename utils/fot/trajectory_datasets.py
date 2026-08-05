"""Deterministic task trajectories for the NeurReps flow rebuild."""

from __future__ import annotations

import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

from utils.llm_baselines import CHIRAL_TETRIS_SHAPES

from .colored_shapes_ops import random_colored_rectangles
from .maze_ops import bfs_shortest_path, build_cond, generate_maze, nodes_to_trace, resize_nn
from .rotation_ops import rotate_tensor
from .tetris_ops import get_tetris_tensor


class RotationTrajectoryDataset(Dataset):
    """Base images plus deterministic start/action parameters for orbit rendering."""

    def __init__(
        self,
        *,
        task: str,
        n_samples: int,
        image_size: int = 64,
        seed: int = 0,
        num_shapes: int = 4,
        shape_keys: Tuple[str, ...] | None = None,
        split_label: str = "unspecified",
    ):
        if task not in {"tetris", "colored"}:
            raise ValueError(f"Unsupported rotation task: {task}")
        self.task = task
        self.n_samples = int(n_samples)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.num_shapes = int(num_shapes)
        self.tetris_keys = tuple(sorted(CHIRAL_TETRIS_SHAPES)) if shape_keys is None else tuple(shape_keys)
        if task == "tetris" and not self.tetris_keys:
            raise ValueError("Tetris shape_keys must not be empty")
        self.split_label = split_label

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + 1_000_003 * int(index))
        if self.task == "tetris":
            key = rng.choice(self.tetris_keys)
            base = get_tetris_tensor(key, self.image_size, channels=1).clone()
        else:
            base = random_colored_rectangles(
                self.image_size, self.image_size, num_shapes=self.num_shapes, rng=rng
            )
        if rng.random() < 0.5:
            base = torch.flip(base, dims=[2])
        # Source pose and requested motion are both seen during training. This
        # removes the legacy canonical-condition/inference-condition mismatch.
        start = torch.tensor(rng.uniform(-180.0, 180.0), dtype=torch.float32)
        delta = torch.tensor(rng.uniform(-180.0, 180.0), dtype=torch.float32)
        return base, start, delta


def render_rotation_state(
    base: torch.Tensor,
    start_degrees: torch.Tensor,
    delta_degrees: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Render x(t) directly from the base, avoiding accumulated resampling."""
    if t.ndim > 1:
        t = t.reshape(-1)
    angles = start_degrees.reshape(-1) + t * delta_degrees.reshape(-1)
    return rotate_tensor(base, angles, pad_to_diag=True, pad_value=0.0).clamp(0.0, 1.0)


def render_rotation_frames(
    base: torch.Tensor,
    start_degrees: torch.Tensor,
    delta_degrees: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Return (B,T,C,H,W) states, each rendered once from the same base."""
    batch, channels, height, width = base.shape
    frame_count = int(times.numel())
    expanded_base = base[:, None].expand(-1, frame_count, -1, -1, -1).reshape(
        batch * frame_count, channels, height, width
    )
    angles = start_degrees[:, None] + delta_degrees[:, None] * times[None, :].to(base.device)
    rendered = rotate_tensor(expanded_base, angles.reshape(-1), pad_to_diag=True, pad_value=0.0)
    return rendered.reshape(batch, frame_count, channels, height, width).clamp(0.0, 1.0)


class MazeTrajectoryDataset(Dataset):
    """Maze conditions and cumulative ground-truth shortest-path trajectories."""

    def __init__(
        self,
        *,
        n_samples: int,
        maze_cells: int = 9,
        image_size: int = 64,
        trajectory_steps: int = 8,
        seed: int = 0,
    ):
        self.n_samples = int(n_samples)
        self.maze_cells = int(maze_cells)
        self.image_size = int(image_size)
        self.trajectory_steps = int(trajectory_steps)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + 1_000_003 * int(index))
        grid = generate_maze(self.maze_cells, self.maze_cells, rng)
        start = (1, 1)
        goal = (grid.shape[0] - 2, grid.shape[1] - 2)
        path = bfs_shortest_path(grid, start, goal)
        if not path:
            raise RuntimeError("Perfect-maze generator produced no path")
        frames = []
        for step in range(self.trajectory_steps + 1):
            fraction = step / float(self.trajectory_steps)
            nodes = max(1, round(1 + fraction * (len(path) - 1)))
            trace = torch.tensor(nodes_to_trace(grid.shape, path[:nodes]), dtype=torch.float32)
            frames.append(resize_nn(trace, self.image_size))
        return build_cond(grid, start, goal, self.image_size), torch.stack(frames, dim=0)
