from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from utils.fot.envs_maze import MazeEnvFMProgress
from utils.fot.envs_rotation import RotationEnv
from utils.fot.models import FastRotator
from utils.fot.maze_moe import MazeExpertMixtureFlow, integrate_maze_moe
from utils.fot.trajectory_flow import (
    TrajectoryFlowField,
    integrate_deformation_times,
    integrate_trajectory,
    rotation_action,
)


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

    def test_trajectory_flow_rollout_is_differentiable(self) -> None:
        flow = TrajectoryFlowField(
            state_channels=1, condition_channels=3, width=8, context_dim=32
        )
        initial = torch.zeros((2, 1, 16, 16))
        condition = torch.zeros((2, 3, 16, 16))
        endpoint = integrate_trajectory(
            flow,
            initial,
            condition,
            rotation_action(torch.tensor([30.0, -45.0])),
            steps=2,
        )
        endpoint.square().mean().backward()
        self.assertEqual(tuple(endpoint.shape), (2, 1, 16, 16))
        self.assertTrue(all(parameter.grad is not None for parameter in flow.parameters()))

    def test_transport_flow_renders_arbitrary_times_from_source(self) -> None:
        flow = TrajectoryFlowField(
            state_channels=1, condition_channels=1, width=8, context_dim=32,
            dynamics_mode="transport",
        )
        source = torch.rand((2, 1, 16, 16))
        frames = integrate_deformation_times(
            flow, source, source, torch.zeros(2, 3), [0.0, 0.37, 1.0], max_step=0.1
        )
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(torch.allclose(frame, source, atol=2e-5) for frame in frames))

    def test_maze_moe_routes_frozen_rotation_features(self) -> None:
        tetris = TrajectoryFlowField(
            state_channels=1, condition_channels=1, width=8, context_dim=32,
            dynamics_mode="transport",
        )
        colored = TrajectoryFlowField(
            state_channels=3, condition_channels=3, width=8, context_dim=32,
            dynamics_mode="transport",
        )
        model = MazeExpertMixtureFlow(
            tetris, colored, width=8, context_dim=32, expert_dim=8, router_width=8
        )
        condition = torch.zeros((2, 3, 16, 16))
        condition[:, 0, 4:12, 7:9] = 1
        initial = torch.zeros((2, 1, 16, 16))
        prepared = model.prepare_experts(condition)
        velocity, weights = model(
            initial, condition, torch.zeros(2, 1),
            prepared_experts=prepared, return_router=True,
        )
        self.assertEqual(tuple(velocity.shape), (2, 1, 16, 16))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0])))
        self.assertTrue(all(not parameter.requires_grad for parameter in tetris.parameters()))
        endpoint = integrate_maze_moe(model, initial, condition, None, steps=2)
        endpoint.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if not name.startswith(("tetris_expert.", "colored_expert."))
        ))


if __name__ == "__main__":
    unittest.main()
