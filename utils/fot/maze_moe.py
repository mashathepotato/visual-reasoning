"""Maze flow with frozen Tetris and colored-shape spatial experts."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trajectory_flow import FiLMResidualBlock, TrajectoryFlowField, sinusoidal_time_embedding


PreparedExperts = Tuple[torch.Tensor, torch.Tensor]


class MazeExpertMixtureFlow(nn.Module):
    """Additive maze dynamics routed over two frozen rotation representations.

    The rotation checkpoints are used only as static spatial feature extractors.
    A maze-specific U-Net and a pixelwise router are trained from scratch.
    """

    state_channels = 1
    condition_channels = 3
    action_dim = 3
    dynamics_mode = "additive"

    def __init__(
        self,
        tetris_expert: TrajectoryFlowField,
        colored_expert: TrajectoryFlowField,
        *,
        width: int = 32,
        context_dim: int = 128,
        expert_dim: int = 16,
        router_width: int = 16,
        router_temperature: float = 0.5,
        router_mode: str = "learned",
    ):
        super().__init__()
        if tetris_expert.dynamics_mode != "transport" or colored_expert.dynamics_mode != "transport":
            raise ValueError("Maze MoE requires transport-flow rotation experts")
        if tetris_expert.state_channels != 1 or colored_expert.state_channels != 3:
            raise ValueError("Expected one-channel Tetris and three-channel colored experts")
        if router_mode not in {"learned", "uniform", "tetris_only", "colored_only"}:
            raise ValueError(f"Unknown router mode: {router_mode}")
        self.tetris_expert = tetris_expert
        self.colored_expert = colored_expert
        self.width = int(width)
        self.context_dim = int(context_dim)
        self.expert_dim = int(expert_dim)
        self.router_width = int(router_width)
        if router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        self.router_temperature = float(router_temperature)
        self.router_mode = router_mode
        for expert in (self.tetris_expert, self.colored_expert):
            expert.requires_grad_(False)
            expert.eval()

        self.tetris_projection = nn.Sequential(
            nn.Conv2d(tetris_expert.width, expert_dim, 1),
            nn.GroupNorm(self._groups(expert_dim), expert_dim),
            nn.SiLU(),
        )
        self.colored_projection = nn.Sequential(
            nn.Conv2d(colored_expert.width, expert_dim, 1),
            nn.GroupNorm(self._groups(expert_dim), expert_dim),
            nn.SiLU(),
        )
        router_inputs = 1 + 3 + 2 + 2 * expert_dim
        self.router = nn.Sequential(
            nn.Conv2d(router_inputs, router_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(router_width, 2, 1),
        )
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.router[-1].bias)

        self.time_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(3, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.stem = nn.Conv2d(1 + 3 + 2 + expert_dim, width, 3, padding=1)
        self.enc1 = FiLMResidualBlock(width, width, context_dim)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc2 = FiLMResidualBlock(width * 2, width * 2, context_dim)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = FiLMResidualBlock(width * 4, width * 4, context_dim)
        self.dec2 = FiLMResidualBlock(width * 6, width * 2, context_dim)
        self.dec1 = FiLMResidualBlock(width * 3, width, context_dim)
        self.out_norm = nn.GroupNorm(self._groups(width), width)
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _groups(channels: int) -> int:
        groups = min(8, int(channels))
        while channels % groups:
            groups -= 1
        return groups

    @staticmethod
    def coordinates(reference: torch.Tensor) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, reference.shape[-2], device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, reference.shape[-1], device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(reference.shape[0], -1, -1, -1)

    @staticmethod
    def expert_inputs(condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wall, start, goal = condition[:, :1], condition[:, 1:2], condition[:, 2:3]
        gray = torch.maximum(0.35 * wall, torch.maximum(start, goal)).clamp(0.0, 1.0)
        colored = torch.cat(
            [0.35 * wall + goal, 0.35 * wall + start, 0.35 * wall], dim=1
        ).clamp(0.0, 1.0)
        return gray, colored

    @torch.no_grad()
    def prepare_experts(self, condition: torch.Tensor) -> PreparedExperts:
        """Compute static frozen-expert features once for an entire rollout."""
        self.tetris_expert.eval()
        self.colored_expert.eval()
        gray, colored = self.expert_inputs(condition)
        batch = condition.shape[0]
        time = torch.zeros(batch, 1, device=condition.device, dtype=condition.dtype)
        # rotation_action(0 degrees) = [0, 0, 1]
        action = torch.zeros(batch, 3, device=condition.device, dtype=condition.dtype)
        action[:, 2] = 1.0
        return (
            self.tetris_expert.spatial_features(gray, gray, time, action).detach(),
            self.colored_expert.spatial_features(colored, colored, time, action).detach(),
        )

    def routing(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        prepared_experts: PreparedExperts,
        *,
        router_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self.router_mode if router_mode is None else router_mode
        if mode not in {"learned", "uniform", "tetris_only", "colored_only"}:
            raise ValueError(f"Unknown router mode: {mode}")
        tetris = self.tetris_projection(prepared_experts[0])
        colored = self.colored_projection(prepared_experts[1])
        if mode == "learned":
            logits = self.router(
                torch.cat([state, condition, self.coordinates(state), tetris, colored], dim=1)
            )
            weights = (logits / self.router_temperature).softmax(dim=1)
        else:
            values = {
                "uniform": (0.5, 0.5),
                "tetris_only": (1.0, 0.0),
                "colored_only": (0.0, 1.0),
            }[mode]
            weights = torch.empty(
                state.shape[0], 2, state.shape[-2], state.shape[-1],
                device=state.device, dtype=state.dtype,
            )
            weights[:, 0].fill_(values[0])
            weights[:, 1].fill_(values[1])
        mixture = weights[:, :1] * tetris + weights[:, 1:2] * colored
        return mixture, weights

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        prepared_experts: Optional[PreparedExperts] = None,
        router_mode: Optional[str] = None,
        return_router: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        if t.ndim == 1:
            t = t[:, None]
        if action is None:
            action = torch.zeros(batch, 3, device=state.device, dtype=state.dtype)
        if prepared_experts is None:
            prepared_experts = self.prepare_experts(condition)
        mixture, weights = self.routing(
            state, condition, prepared_experts, router_mode=router_mode
        )
        context = self.time_mlp(sinusoidal_time_embedding(t, self.context_dim)) + self.action_mlp(action)
        inputs = torch.cat([state, condition, self.coordinates(state), mixture], dim=1)
        x1 = self.enc1(self.stem(inputs), context)
        x2 = self.enc2(self.down1(x1), context)
        middle = self.middle(self.down2(x2), context)
        up2 = F.interpolate(middle, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.dec2(torch.cat([up2, x2], dim=1), context)
        up1 = F.interpolate(up2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(torch.cat([up1, x1], dim=1), context)
        velocity = self.out(F.silu(self.out_norm(up1)))
        return (velocity, weights) if return_router else velocity

    def apply_field(self, state: torch.Tensor, field: torch.Tensor, dt: float) -> torch.Tensor:
        return state + float(dt) * field

    def state_velocity(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        prepared_experts: Optional[PreparedExperts] = None,
        router_mode: Optional[str] = None,
    ) -> torch.Tensor:
        return self(
            state, condition, t, action,
            prepared_experts=prepared_experts, router_mode=router_mode,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.tetris_expert.eval()
        self.colored_expert.eval()
        return self

    def config(self) -> dict:
        return {
            "width": self.width,
            "context_dim": self.context_dim,
            "expert_dim": self.expert_dim,
            "router_width": self.router_width,
            "router_temperature": self.router_temperature,
            "router_mode": self.router_mode,
        }


def integrate_maze_moe(
    model: MazeExpertMixtureFlow,
    initial: torch.Tensor,
    condition: torch.Tensor,
    action: Optional[torch.Tensor],
    *,
    steps: int,
    method: str = "heun",
    clamp: Optional[Tuple[float, float]] = None,
    return_frames: bool = False,
    prepared_experts: Optional[PreparedExperts] = None,
    router_mode: Optional[str] = None,
) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
    """Integrate maze dynamics while reusing frozen features at every step."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unknown integration method: {method}")
    if prepared_experts is None:
        prepared_experts = model.prepare_experts(condition)
    state = initial
    frames = [state]
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full((state.shape[0], 1), index * dt, device=state.device, dtype=state.dtype)
        field = model(
            state, condition, t0, action,
            prepared_experts=prepared_experts, router_mode=router_mode,
        )
        if method == "heun":
            proposal = model.apply_field(state, field, dt)
            t1 = torch.full(
                (state.shape[0], 1), (index + 1) * dt,
                device=state.device, dtype=state.dtype,
            )
            field_next = model(
                proposal, condition, t1, action,
                prepared_experts=prepared_experts, router_mode=router_mode,
            )
            state = model.apply_field(state, 0.5 * (field + field_next), dt)
        else:
            state = model.apply_field(state, field, dt)
        if clamp is not None:
            state = state.clamp(*clamp)
        if return_frames:
            frames.append(state)
    return (state, frames) if return_frames else state
