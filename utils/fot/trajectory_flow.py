"""Trajectory-supervised continuous spatial flow fields.

Unlike the legacy FoT models, this module trains the vector field on states that
lie on an explicit task trajectory and evaluates the same multi-step integration
used at inference time. It intentionally has no PPO dependency.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10_000.0) / max(1, half - 1))
    )
    angles = t * frequencies[None, :] * (2.0 * math.pi)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.context = nn.Linear(context_dim, 2 * out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.context(context).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return residual + self.conv2(F.silu(hidden))


class TrajectoryFlowField(nn.Module):
    """A time/action-conditioned U-Net vector field over a spatial neural state."""

    def __init__(
        self,
        *,
        state_channels: int,
        condition_channels: int,
        action_dim: int = 3,
        width: int = 32,
        context_dim: int = 128,
        dynamics_mode: str = "additive",
    ):
        super().__init__()
        if dynamics_mode not in {"additive", "transport"}:
            raise ValueError(f"Unknown dynamics mode: {dynamics_mode}")
        self.state_channels = int(state_channels)
        self.condition_channels = int(condition_channels)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.context_dim = int(context_dim)
        self.dynamics_mode = dynamics_mode

        self.time_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        # Explicit coordinates let the field represent spatial generators (for
        # example rotations) without baking the renderer into the architecture.
        self.stem = nn.Conv2d(state_channels + condition_channels + 2, width, 3, padding=1)
        self.enc1 = FiLMResidualBlock(width, width, context_dim)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc2 = FiLMResidualBlock(width * 2, width * 2, context_dim)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = FiLMResidualBlock(width * 4, width * 4, context_dim)
        self.dec2 = FiLMResidualBlock(width * 6, width * 2, context_dim)
        self.dec1 = FiLMResidualBlock(width * 3, width, context_dim)
        self.out_norm = nn.GroupNorm(min(8, width), width)
        output_channels = state_channels if dynamics_mode == "additive" else 2
        self.out = nn.Conv2d(width, output_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = state.shape[0]
        if t.ndim == 1:
            t = t[:, None]
        if action is None:
            action = torch.zeros(batch, self.action_dim, device=state.device, dtype=state.dtype)
        context = self.time_mlp(sinusoidal_time_embedding(t, self.context_dim)) + self.action_mlp(action)
        y_coordinates = torch.linspace(-1.0, 1.0, state.shape[-2], device=state.device, dtype=state.dtype)
        x_coordinates = torch.linspace(-1.0, 1.0, state.shape[-1], device=state.device, dtype=state.dtype)
        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        coordinates = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        x1 = self.enc1(self.stem(torch.cat([state, condition, coordinates], dim=1)), context)
        x2 = self.enc2(self.down1(x1), context)
        middle = self.middle(self.down2(x2), context)
        up2 = F.interpolate(middle, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.dec2(torch.cat([up2, x2], dim=1), context)
        up1 = F.interpolate(up2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(torch.cat([up1, x1], dim=1), context)
        return self.out(F.silu(self.out_norm(up1)))

    def config(self) -> dict:
        return {
            "state_channels": self.state_channels,
            "condition_channels": self.condition_channels,
            "action_dim": self.action_dim,
            "width": self.width,
            "context_dim": self.context_dim,
            "dynamics_mode": self.dynamics_mode,
        }

    @staticmethod
    def sampling_grid(state: torch.Tensor) -> torch.Tensor:
        y_coordinates = torch.linspace(-1.0, 1.0, state.shape[-2], device=state.device, dtype=state.dtype)
        x_coordinates = torch.linspace(-1.0, 1.0, state.shape[-1], device=state.device, dtype=state.dtype)
        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(state.shape[0], -1, -1, -1)

    # Compatibility for the v1 training code and frozen checkpoints.
    _sampling_grid = sampling_grid

    def apply_field(self, state: torch.Tensor, field: torch.Tensor, dt: float) -> torch.Tensor:
        if self.dynamics_mode == "additive":
            return state + float(dt) * field
        grid = self.sampling_grid(state) + float(dt) * field.permute(0, 2, 3, 1)
        return F.grid_sample(
            state, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

    def state_velocity(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        epsilon: float = 0.01,
    ) -> torch.Tensor:
        field = self(state, condition, t, action)
        if self.dynamics_mode == "additive":
            return field
        before = self.apply_field(state, field, -0.5 * epsilon)
        after = self.apply_field(state, field, 0.5 * epsilon)
        return (after - before) / float(epsilon)


def rotation_action(delta_degrees: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(delta_degrees)
    return torch.stack([delta_degrees / 180.0, radians.sin(), radians.cos()], dim=1)


def integrate_trajectory(
    model: TrajectoryFlowField,
    initial: torch.Tensor,
    condition: torch.Tensor,
    action: Optional[torch.Tensor],
    *,
    steps: int,
    method: str = "heun",
    clamp: Optional[Tuple[float, float]] = None,
    return_frames: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
    """Differentiable fixed-step ODE integration used in both train and audit."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unknown integration method: {method}")
    state = initial
    frames = [state]
    dt = 1.0 / float(steps)
    for index in range(steps):
        t0 = torch.full((state.shape[0], 1), index * dt, device=state.device, dtype=state.dtype)
        field = model(state, condition, t0, action)
        if method == "heun":
            proposal = model.apply_field(state, field, dt)
            t1 = torch.full((state.shape[0], 1), (index + 1) * dt, device=state.device, dtype=state.dtype)
            field_next = model(proposal, condition, t1, action)
            state = model.apply_field(state, 0.5 * (field + field_next), dt)
        else:
            state = model.apply_field(state, field, dt)
        if clamp is not None:
            state = state.clamp(*clamp)
        if return_frames:
            frames.append(state)
    return (state, frames) if return_frames else state


def render_from_sampling_map(source: torch.Tensor, sampling_map: torch.Tensor) -> torch.Tensor:
    """Render from the original source exactly once for the requested state."""
    return F.grid_sample(
        source, sampling_map, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def sample_vector_field(field: torch.Tensor, sampling_map: torch.Tensor) -> torch.Tensor:
    """Evaluate an Eulerian vector field along the current characteristics."""
    return F.grid_sample(
        field, sampling_map, mode="bilinear", padding_mode="zeros", align_corners=True
    ).permute(0, 2, 3, 1)


def integrate_deformation_times(
    model: TrajectoryFlowField,
    source: torch.Tensor,
    condition: torch.Tensor,
    action: Optional[torch.Tensor],
    times: Sequence[float] | torch.Tensor,
    *,
    max_step: float = 1.0 / 12.0,
    method: str = "heun",
    clamp: Optional[Tuple[float, float]] = None,
    return_maps: bool = False,
) -> List[torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate transport-flow states at arbitrary continuous times.

    A deformation map is integrated continuously, while every returned image is
    sampled directly from ``source``. This avoids the progressive blur caused by
    repeatedly resampling the previous image. Additive dynamics should continue
    to use :func:`integrate_trajectory`.
    """
    if model.dynamics_mode != "transport":
        raise ValueError("Single-resampling deformation integration requires transport dynamics")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unknown integration method: {method}")
    requested = [float(value) for value in (times.detach().cpu().tolist() if torch.is_tensor(times) else times)]
    if any(value < 0.0 or value > 1.0 for value in requested):
        raise ValueError("Requested times must lie in [0, 1]")
    if any(right < left for left, right in zip(requested, requested[1:])):
        raise ValueError("Requested times must be sorted")

    sampling_map = model.sampling_grid(source).clone()
    # Preserve the frozen model's native recurrent state as the field input.
    # The parallel map is a decoder state: it records the same transport while
    # allowing requested images to be sampled once from the original source.
    shadow_state = source
    current_time = 0.0
    frames: List[torch.Tensor] = []
    maps: List[torch.Tensor] = []
    for target_time in requested:
        while current_time + 1e-10 < target_time:
            dt = min(float(max_step), target_time - current_time)
            t0 = torch.full(
                (source.shape[0], 1), current_time, device=source.device, dtype=source.dtype
            )
            field = model(shadow_state, condition, t0, action)
            if method == "heun":
                proposal_state = model.apply_field(shadow_state, field, dt)
                map_velocity = sample_vector_field(field, sampling_map)
                proposal_map = sampling_map + dt * map_velocity
                t1 = torch.full(
                    (source.shape[0], 1), current_time + dt, device=source.device, dtype=source.dtype
                )
                next_field = model(proposal_state, condition, t1, action)
                average_field = 0.5 * (field + next_field)
                shadow_state = model.apply_field(shadow_state, average_field, dt)
                next_map_velocity = sample_vector_field(next_field, proposal_map)
                sampling_map = sampling_map + 0.5 * dt * (map_velocity + next_map_velocity)
            else:
                shadow_state = model.apply_field(shadow_state, field, dt)
                sampling_map = sampling_map + dt * sample_vector_field(field, sampling_map)
            if clamp is not None:
                shadow_state = shadow_state.clamp(*clamp)
            current_time += dt
        frame = render_from_sampling_map(source, sampling_map)
        if clamp is not None:
            frame = frame.clamp(*clamp)
        frames.append(frame)
        if return_maps:
            maps.append(sampling_map.clone())
    return (frames, maps) if return_maps else frames


def weighted_image_loss(prediction: torch.Tensor, target: torch.Tensor, *, foreground_weight: float = 4.0) -> torch.Tensor:
    foreground = target.detach().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
    weight = 1.0 + foreground_weight * foreground
    return ((prediction - target).abs() * weight).mean()


def soft_dice_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    dims: Sequence[int] = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
