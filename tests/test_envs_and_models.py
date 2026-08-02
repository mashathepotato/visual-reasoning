from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from utils.fot.envs_maze import MazeEnvFMProgress
from utils.fot.envs_rotation import RotationEnv
from utils.fot.models import FastRotator


class ZeroSketcher(nn.Module):
    def forward(self, trace: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del cond, t
        return torch.zeros_like(trace)


class EnvironmentAndModelTests(unittest.TestCase):
    def test_rotation_action_and_same_commit(self) -> None:
        image = torch.zeros((3, 16, 16), dtype=torch.float32)
        image[:, 5:10, 6:9] = 1.0
        env = RotationEnv(image_shape=(3, 16, 16), device=torch.device("cpu"), max_steps=3)
        observation, _ = env.reset(options={"pair": (image, image.clone(), False)})
        self.assertEqual(observation.shape, (7, 16, 16))
        _, reward, terminated, truncated, _ = env.step(6)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 120.0)

    def test_rotation_invalid_action(self) -> None:
        image = torch.zeros((3, 8, 8), dtype=torch.float32)
        env = RotationEnv(image_shape=(3, 8, 8), device=torch.device("cpu"))
        env.reset(options={"pair": (image, image, False)})
        with self.assertRaises(ValueError):
            env.step(9)

    def test_maze_hold_reward_and_invalid_action(self) -> None:
        env = MazeEnvFMProgress(
            sketcher=ZeroSketcher(),
            maze_cells=3,
            img_size=16,
            max_steps=2,
            device=torch.device("cpu"),
            seed=0,
        )
        observation, _ = env.reset(seed=4)
        self.assertEqual(observation.shape, (4, 16, 16))
        _, reward, terminated, truncated, _ = env.step(3)
        self.assertAlmostEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        with self.assertRaises(ValueError):
            env.step(4)

    def test_flow_shape(self) -> None:
        flow = FastRotator(in_ch=1, out_ch=1, backbone_dim=8, flow_dim=4)
        output = flow(
            torch.zeros((2, 1, 16, 16)),
            torch.zeros((2, 1)),
            torch.zeros((2, 8)),
            torch.zeros((2, 1)),
        )
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
