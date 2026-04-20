from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_maze(cells_w: int, cells_h: int, rng: random.Random) -> np.ndarray:
    """Perfect maze with DFS backtracking. Returns grid with 1=wall, 0=free."""
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


def path_to_segments(path: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    if not path:
        return []
    if len(path) == 1:
        return [[path[0]]]

    segments: List[List[Tuple[int, int]]] = []
    curr = [path[0]]
    curr_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    for i in range(1, len(path)):
        step = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if step != curr_dir:
            segments.append(curr)
            curr = [path[i - 1], path[i]]
            curr_dir = step
        else:
            curr.append(path[i])
    segments.append(curr)
    return segments


def segment_frames_from_path(path: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    segments = path_to_segments(path)
    frames: List[List[Tuple[int, int]]] = [[]]
    current: List[Tuple[int, int]] = []
    for seg in segments:
        if current and seg and current[-1] == seg[0]:
            current.extend(seg[1:])
        else:
            current.extend(seg)
        frames.append(list(current))
    return frames


def one_hot_point(shape: Tuple[int, int], pos: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.float32)
    m[pos[0], pos[1]] = 1.0
    return m


def nodes_to_trace(shape: Tuple[int, int], nodes: List[Tuple[int, int]]) -> np.ndarray:
    trace = np.zeros(shape, dtype=np.float32)
    for y, x in nodes:
        trace[y, x] = 1.0
    return trace


def resize_nn(t: torch.Tensor, size: int) -> torch.Tensor:
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    out = F.interpolate(t, size=(size, size), mode="nearest")
    return out.squeeze(0)


def build_cond(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], img_size: int) -> torch.Tensor:
    walls = grid.astype(np.float32)
    start_ch = one_hot_point(grid.shape, start)
    goal_ch = one_hot_point(grid.shape, goal)
    cond = np.stack([walls, start_ch, goal_ch], axis=0)
    cond_t = torch.tensor(cond).float()
    return resize_nn(cond_t, img_size)


class MazeTraceDataset(Dataset):
    """Time-conditioned sketch steps from segment frames."""

    def __init__(self, *, n_samples: int, maze_cells: int, img_size: int, seed: int = 0):
        self.n = int(n_samples)
        self.maze_cells = int(maze_cells)
        self.img_size = int(img_size)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        grid = generate_maze(self.maze_cells, self.maze_cells, self.rng)
        start = (1, 1)
        goal = (grid.shape[0] - 2, grid.shape[1] - 2)
        path = bfs_shortest_path(grid, start, goal)
        frames = segment_frames_from_path(path)
        if len(frames) < 2:
            return self.__getitem__(idx + 1)

        k = self.rng.randint(0, len(frames) - 2)
        trace_t_np = nodes_to_trace(grid.shape, frames[k])
        trace_next_np = nodes_to_trace(grid.shape, frames[k + 1])
        delta_np = trace_next_np - trace_t_np
        t = float(k) / float(max(1, len(frames) - 1))

        cond = build_cond(grid, start, goal, self.img_size)
        trace_t = resize_nn(torch.tensor(trace_t_np).float(), self.img_size)
        delta = resize_nn(torch.tensor(delta_np).float(), self.img_size)

        # Return:
        # - cond: (3, H, W)
        # - trace_t: (1, H, W)
        # - t: (1,)
        # - delta: (1, H, W)
        trace_t = trace_t.unsqueeze(0)
        delta = delta.unsqueeze(0)
        t_t = torch.tensor([t], dtype=torch.float32)
        return cond, trace_t, t_t, delta

