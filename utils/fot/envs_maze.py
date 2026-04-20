from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .maze_ops import (
    bfs_shortest_path,
    build_cond,
    generate_maze,
    one_hot_point,
    resize_nn,
    segment_frames_from_path,
)
from .torch_utils import get_device


class MazeEnvFMProgress(gym.Env):
    """PPO environment where actions control progress along an FM trace."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        sketcher: nn.Module,
        maze_cells: int = 9,
        img_size: int = 64,
        max_steps: int = 180,
        device: Optional[torch.device] = None,
        seed: int = 0,
        goal_reward: float = 10.0,
        progress_reward: float = 1.0,
        step_penalty: float = 0.01,
    ):
        super().__init__()
        self.sketcher = sketcher
        self.maze_cells = int(maze_cells)
        self.img_size = int(img_size)
        self.max_steps = int(max_steps)
        self.device = device or get_device()
        self.rng = random.Random(seed)

        self.goal_reward = float(goal_reward)
        self.progress_reward = float(progress_reward)
        self.step_penalty = float(step_penalty)

        # channels: walls, start, goal, trace
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, img_size, img_size), dtype=np.float32)
        # actions: advance by 1, 2, 4, or hold
        self.action_space = gym.spaces.Discrete(4)
        self.action_steps = {0: 1, 1: 2, 2: 4, 3: 0}

        self.grid = None
        self.start = None
        self.goal = None
        self.frames_len = None
        self.progress = 0
        self.trace = None
        self.cond = None
        self.goal_mask = None
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        trace_np = self.trace[0].detach().cpu().numpy()  # (1, H, W)
        cond_np = self.cond[0].detach().cpu().numpy()  # (3, H, W)
        obs = np.concatenate([cond_np, trace_np], axis=0)
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        grid = generate_maze(self.maze_cells, self.maze_cells, self.rng)
        start = (1, 1)
        goal = (grid.shape[0] - 2, grid.shape[1] - 2)
        path = bfs_shortest_path(grid, start, goal)
        frames = segment_frames_from_path(path)

        self.grid = grid
        self.start = start
        self.goal = goal
        self.frames_len = max(2, len(frames))
        self.progress = 0
        self.step_count = 0

        cond = build_cond(grid, start, goal, self.img_size)
        self.cond = cond.unsqueeze(0).to(self.device)
        goal_ch = one_hot_point(grid.shape, goal)
        self.goal_mask = resize_nn(torch.tensor(goal_ch).float(), self.img_size).to(self.device)
        self.trace = torch.zeros((1, 1, self.img_size, self.img_size), device=self.device)

        return self._obs(), {}

    def step(self, action: int):
        n = self.action_steps.get(int(action), 1)
        prev_progress = self.progress

        # advance FM sketch n steps
        for _ in range(int(n)):
            if self.progress >= self.frames_len - 1:
                break
            t = torch.tensor([[self.progress / max(1, self.frames_len - 1)]], device=self.device)
            with torch.no_grad():
                delta = self.sketcher(self.trace, self.cond, t)
            self.trace = (self.trace + delta).clamp(0.0, 1.0)
            self.progress += 1

        progress_delta = self.progress - prev_progress
        reward = -self.step_penalty + self.progress_reward * (progress_delta / max(1, self.frames_len - 1))

        terminated = False
        if self.progress >= self.frames_len - 1:
            reward += self.goal_reward
            terminated = True

        goal_on_trace = (self.trace[0, 0] * self.goal_mask[0]).sum().item() > 0.5
        if goal_on_trace and not terminated:
            reward += self.goal_reward * 0.5
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, {}

