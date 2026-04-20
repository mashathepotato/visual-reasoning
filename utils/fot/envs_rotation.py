from __future__ import annotations

from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .dino_utils import dino_embed_fm_gray64
from .integrators import apply_heun_steps
from .rotation_ops import build_state, rotate_tensor, wrap_angle_deg
from .tetris_ops import get_tetris_tensor, infer_tetris_shape_key, to_fm_tensor, to_obs_tensor
from .torch_utils import get_device


class RotationEnv(gym.Env):
    """Rotation env with deterministic kornia dynamics (used in ppo.ipynb)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int, int],
        max_steps: int = 180,
        max_total_rotation: float = 360.0,
        epsilon: float = 0.03,
        commit_bonus: float = 20.0,
        commit_bonus_eps: Optional[float] = None,
        mismatch_patience: int = 15,
        improve_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        pair_sampler: Optional[Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor, bool]]] = None,
    ):
        super().__init__()
        self.image_shape = image_shape  # (3, H, W)
        self.max_steps = int(max_steps)
        self.max_total_rotation = float(max_total_rotation)
        self.epsilon = float(epsilon)
        self.commit_bonus = float(commit_bonus)
        self.commit_bonus_eps = float(commit_bonus_eps) if commit_bonus_eps is not None else float(0.5 * epsilon)
        self.mismatch_patience = int(mismatch_patience)
        self.improve_eps = float(improve_eps)
        self.device = device or get_device()
        self.pair_sampler = pair_sampler

        _, h, w = image_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7, h, w), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

        self.base_source = None
        self.target = None
        self.current_source = None
        self.current_angle = 0.0
        self.is_mirrored = False
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = None
        self.best_error = None
        self.best_angle = 0.0
        self.no_improve_steps = 0

    def _set_pair(self, source: torch.Tensor, target: torch.Tensor, is_mirrored: bool) -> None:
        self.base_source = source.to(self.device).float().clamp(0, 1)
        self.target = target.to(self.device).float().clamp(0, 1)
        self.is_mirrored = bool(is_mirrored)
        self.current_angle = 0.0
        self.current_source = self.base_source
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = self._alignment_error().detach()
        self.best_error = self.prev_error
        self.best_angle = self.current_angle
        self.no_improve_steps = 0

    def _obs(self) -> np.ndarray:
        state = build_state(self.current_source, self.target, self.current_angle)[0]
        return state.detach().cpu().numpy().astype(np.float32)

    def _alignment_error(self) -> torch.Tensor:
        return torch.mean((self.current_source - self.target) ** 2)

    def _update_no_improve(self, err: torch.Tensor) -> None:
        if err < (self.prev_error - self.improve_eps):
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1
        self.prev_error = err.detach()

    def _update_best(self, err: torch.Tensor) -> None:
        if self.best_error is None or err < self.best_error:
            self.best_error = err.detach()
            self.best_angle = self.current_angle

    def _apply_rotation(self, delta: float) -> None:
        self.current_angle = (self.current_angle + float(delta)) % 360.0
        self.total_rotation += abs(float(delta))
        self.current_source = rotate_tensor(self.base_source, self.current_angle)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        if "pair" in options:
            source, target, is_mirrored = options["pair"]
        elif "source" in options and "target" in options:
            source = options["source"]
            target = options["target"]
            is_mirrored = bool(options.get("is_mirrored", False))
        elif self.pair_sampler is not None:
            source, target, is_mirrored = self.pair_sampler(self.device)
        else:
            raise ValueError("No pair provided. Pass options={pair: (...)} or set pair_sampler.")

        self._set_pair(source, target, is_mirrored)
        info = {"angle": self.current_angle}
        return self._obs(), info

    def step(self, action: int):
        terminated = False
        reward = 0.0
        err = self._alignment_error()

        if action == 0 or action == 1:
            delta = -30.0 if action == 0 else 30.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 2 or action == 3:
            delta = -15.0 if action == 2 else 15.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 4 or action == 5:
            delta = -2.0 if action == 4 else 2.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 8:
            delta = 180.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 6:
            best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
            if best_err < self.epsilon:
                is_match = not self.is_mirrored
                reward = 100.0 if is_match else -100.0
                if is_match and best_err < self.commit_bonus_eps:
                    reward += self.commit_bonus
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        elif action == 7:
            if self.no_improve_steps >= self.mismatch_patience:
                best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
                is_mismatch = self.is_mirrored and (best_err >= self.epsilon)
                reward = 100.0 if is_mismatch else -100.0
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        truncated = (self.step_count >= self.max_steps) and (not terminated)
        if self.total_rotation >= self.max_total_rotation and not terminated:
            truncated = True

        info = {
            "angle": float(self.current_angle),
            "error": float(err.detach().cpu()),
            "best_error": float(self.best_error.detach().cpu()) if self.best_error is not None else float(err.detach().cpu()),
            "best_angle": float(self.best_angle),
            "is_mirrored": bool(self.is_mirrored),
            "no_improve_steps": int(self.no_improve_steps),
            "total_rotation": float(self.total_rotation),
        }
        return self._obs(), float(reward), terminated, truncated, info


class RotationEnvColorsFM(gym.Env):
    """Rotation env where dynamics are produced by a frozen FM rotator + learned encoder (colors)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int, int],
        fm_model: nn.Module,
        encoder: nn.Module,
        fm_steps: int = 10,
        max_steps: int = 180,
        max_total_rotation: float = 360.0,
        epsilon: float = 0.03,
        commit_bonus: float = 20.0,
        commit_bonus_eps: Optional[float] = None,
        mismatch_patience: int = 15,
        improve_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        pair_sampler: Optional[Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor, bool]]] = None,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.fm_model = fm_model
        self.encoder = encoder
        self.fm_steps = int(fm_steps)
        self.max_steps = int(max_steps)
        self.max_total_rotation = float(max_total_rotation)
        self.epsilon = float(epsilon)
        self.commit_bonus = float(commit_bonus)
        self.commit_bonus_eps = float(commit_bonus_eps) if commit_bonus_eps is not None else float(0.5 * epsilon)
        self.mismatch_patience = int(mismatch_patience)
        self.improve_eps = float(improve_eps)
        self.device = device or get_device()
        self.pair_sampler = pair_sampler

        _, h, w = image_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7, h, w), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

        self.base_source = None
        self.base_emb = None
        self.target = None
        self.current_source = None
        self.current_angle = 0.0
        self.is_mirrored = False
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = None
        self.best_error = None
        self.best_angle = 0.0
        self.no_improve_steps = 0

    def _set_pair(self, source: torch.Tensor, target: torch.Tensor, is_mirrored: bool) -> None:
        src = source.to(self.device).float().clamp(0, 1)
        tgt = target.to(self.device).float().clamp(0, 1)
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(0)
        self.base_source = src
        self.target = tgt
        with torch.no_grad():
            self.base_emb = self.encoder(self.base_source)

        self.is_mirrored = bool(is_mirrored)
        self.current_angle = 0.0
        self.current_source = self.base_source
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = self._alignment_error().detach()
        self.best_error = self.prev_error
        self.best_angle = self.current_angle
        self.no_improve_steps = 0

    def _obs(self) -> np.ndarray:
        state = build_state(self.current_source, self.target, self.current_angle)[0]
        return state.detach().cpu().numpy().astype(np.float32)

    def _alignment_error(self) -> torch.Tensor:
        return torch.mean((self.current_source - self.target) ** 2)

    def _update_no_improve(self, err: torch.Tensor) -> None:
        if err < (self.prev_error - self.improve_eps):
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1
        self.prev_error = err.detach()

    def _update_best(self, err: torch.Tensor) -> None:
        if self.best_error is None or err < self.best_error:
            self.best_error = err.detach()
            self.best_angle = self.current_angle

    def _apply_rotation(self, delta: float) -> None:
        self.current_angle = wrap_angle_deg(self.current_angle + float(delta))
        self.total_rotation += abs(float(delta))
        b = int(self.base_source.shape[0])
        target_ang = torch.full((b, 1), float(self.current_angle), device=self.base_source.device, dtype=self.base_source.dtype)
        self.current_source = apply_heun_steps(
            model=self.fm_model,
            x0=self.base_source,
            cond_emb=self.base_emb,
            target_angle_deg=target_ang,
            steps=self.fm_steps,
            clamp_min=0.0,
            clamp_max=1.0,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        if "pair" in options:
            source, target, is_mirrored = options["pair"]
        elif "source" in options and "target" in options:
            source = options["source"]
            target = options["target"]
            is_mirrored = bool(options.get("is_mirrored", False))
        elif self.pair_sampler is not None:
            source, target, is_mirrored = self.pair_sampler(self.device)
        else:
            raise ValueError("No pair provided. Pass options={pair: (...)} or set pair_sampler.")

        self._set_pair(source, target, is_mirrored)
        info = {"angle": self.current_angle}
        return self._obs(), info

    def step(self, action: int):
        terminated = False
        reward = 0.0
        err = self._alignment_error()

        if action == 0 or action == 1:
            delta = -30.0 if action == 0 else 30.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 2 or action == 3:
            delta = -15.0 if action == 2 else 15.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 4 or action == 5:
            delta = -2.0 if action == 4 else 2.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 8:
            delta = 180.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 6:
            best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
            if best_err < self.epsilon:
                is_match = not self.is_mirrored
                reward = 100.0 if is_match else -100.0
                if is_match and best_err < self.commit_bonus_eps:
                    reward += self.commit_bonus
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        elif action == 7:
            if self.no_improve_steps >= self.mismatch_patience:
                best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
                is_mismatch = self.is_mirrored and (best_err >= self.epsilon)
                reward = 100.0 if is_mismatch else -100.0
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        truncated = (self.step_count >= self.max_steps) and (not terminated)
        if self.total_rotation >= self.max_total_rotation and not terminated:
            truncated = True

        info = {
            "angle": float(self.current_angle),
            "error": float(err.detach().cpu()),
            "best_error": float(self.best_error.detach().cpu()) if self.best_error is not None else float(err.detach().cpu()),
            "best_angle": float(self.best_angle),
            "is_mirrored": bool(self.is_mirrored),
            "no_improve_steps": int(self.no_improve_steps),
            "total_rotation": float(self.total_rotation),
        }
        return self._obs(), float(reward), terminated, truncated, info


class RotationEnvFM(gym.Env):
    """Rotation env where dynamics are produced by a frozen FM rotator conditioned on DINOv3 (tetris/gray)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int, int],
        fm_model: nn.Module,
        dino_model: nn.Module,
        corrector: Optional[nn.Module] = None,
        fm_steps: int = 10,
        max_steps: int = 180,
        max_total_rotation: float = 360.0,
        epsilon: float = 0.03,
        commit_bonus: float = 20.0,
        commit_bonus_eps: Optional[float] = None,
        mismatch_patience: int = 15,
        improve_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        pair_sampler: Optional[Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor, bool]]] = None,
        corrector_weight: float = 1.0,
        corrector_each_step: bool = False,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.fm_model = fm_model
        self.corrector = corrector
        self.corrector_weight = float(corrector_weight)
        self.corrector_each_step = bool(corrector_each_step)
        self.dino_model = dino_model
        self.fm_steps = int(fm_steps)
        self.max_steps = int(max_steps)
        self.max_total_rotation = float(max_total_rotation)
        self.epsilon = float(epsilon)
        self.commit_bonus = float(commit_bonus)
        self.commit_bonus_eps = float(commit_bonus_eps) if commit_bonus_eps is not None else float(0.5 * epsilon)
        self.mismatch_patience = int(mismatch_patience)
        self.improve_eps = float(improve_eps)
        self.device = device or get_device()
        self.pair_sampler = pair_sampler

        _, h, w = image_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7, h, w), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

        self.base_source_obs = None
        self.base_source_fm = None
        self.base_emb = None
        self.target_obs = None
        self.current_source_obs = None
        self.current_source_fm = None
        self.current_angle = 0.0
        self.is_mirrored = False
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = None
        self.best_error = None
        self.best_angle = 0.0
        self.no_improve_steps = 0

    def _set_pair(self, source: torch.Tensor, target: torch.Tensor, is_mirrored: bool) -> None:
        self.base_source_obs = source.to(self.device).float().clamp(0, 1)
        self.target_obs = target.to(self.device).float().clamp(0, 1)
        if self.base_source_obs.dim() == 3:
            self.base_source_obs = self.base_source_obs.unsqueeze(0)
        if self.target_obs.dim() == 3:
            self.target_obs = self.target_obs.unsqueeze(0)

        self.base_source_fm = to_fm_tensor(self.base_source_obs)

        shape_key = infer_tetris_shape_key(self.target_obs)
        if shape_key is not None:
            big = get_tetris_tensor(shape_key, 224, channels=1).to(self.device).float().clamp(0, 1)
            big_fm = to_fm_tensor(big)  # (1,1,224,224) in [-1,1]
            emb = dino_embed_fm_gray64(big_fm, self.dino_model)
            if emb.shape[0] == 1 and self.base_source_fm.shape[0] > 1:
                emb = emb.expand(self.base_source_fm.shape[0], -1)
            self.base_emb = emb
        else:
            self.base_emb = dino_embed_fm_gray64(to_fm_tensor(self.target_obs), self.dino_model)

        self.is_mirrored = bool(is_mirrored)
        self.current_angle = 0.0
        self.current_source_fm = self.base_source_fm.clamp(-1.0, 1.0)
        self.current_source_obs = to_obs_tensor(self.current_source_fm)
        self.step_count = 0
        self.total_rotation = 0.0
        self.prev_error = self._alignment_error().detach()
        self.best_error = self.prev_error
        self.best_angle = self.current_angle
        self.no_improve_steps = 0

    def _obs(self) -> np.ndarray:
        state = build_state(self.current_source_obs, self.target_obs, self.current_angle)[0]
        return state.detach().cpu().numpy().astype(np.float32)

    def _alignment_error(self) -> torch.Tensor:
        return torch.mean((self.current_source_obs - self.target_obs) ** 2)

    def _update_no_improve(self, err: torch.Tensor) -> None:
        if err < (self.prev_error - self.improve_eps):
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1
        self.prev_error = err.detach()

    def _update_best(self, err: torch.Tensor) -> None:
        if self.best_error is None or err < self.best_error:
            self.best_error = err.detach()
            self.best_angle = self.current_angle

    def _apply_rotation(self, delta: float) -> None:
        self.current_angle = wrap_angle_deg(self.current_angle + float(delta))
        self.total_rotation += abs(float(delta))
        b = int(self.base_source_fm.shape[0])
        target_ang = torch.full((b, 1), float(self.current_angle), device=self.base_source_fm.device, dtype=self.base_source_fm.dtype)
        self.current_source_fm = apply_heun_steps(
            model=self.fm_model,
            x0=self.base_source_fm,
            cond_emb=self.base_emb,
            target_angle_deg=target_ang,
            steps=self.fm_steps,
            clamp_min=-1.0,
            clamp_max=1.0,
            corrector=self.corrector,
            corrector_weight=self.corrector_weight,
            corrector_each_step=self.corrector_each_step,
        )
        self.current_source_obs = to_obs_tensor(self.current_source_fm)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        if "pair" in options:
            source, target, is_mirrored = options["pair"]
        elif "source" in options and "target" in options:
            source = options["source"]
            target = options["target"]
            is_mirrored = bool(options.get("is_mirrored", False))
        elif self.pair_sampler is not None:
            source, target, is_mirrored = self.pair_sampler(self.device)
        else:
            raise ValueError("No pair provided. Pass options={pair: (...)} or set pair_sampler.")

        self._set_pair(source, target, is_mirrored)
        info = {"angle": self.current_angle}
        return self._obs(), info

    def step(self, action: int):
        terminated = False
        reward = 0.0
        err = self._alignment_error()

        if action == 0 or action == 1:
            delta = -30.0 if action == 0 else 30.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 2 or action == 3:
            delta = -15.0 if action == 2 else 15.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 4 or action == 5:
            delta = -2.0 if action == 4 else 2.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 8:
            delta = 180.0
            self._apply_rotation(delta)
            err = self._alignment_error()
            reward = -float(err.detach())
            self._update_best(err)
            self._update_no_improve(err)
        elif action == 6:
            best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
            if best_err < self.epsilon:
                is_match = not self.is_mirrored
                reward = 100.0 if is_match else -100.0
                if is_match and best_err < self.commit_bonus_eps:
                    reward += self.commit_bonus
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        elif action == 7:
            if self.no_improve_steps >= self.mismatch_patience:
                best_err = float(self.best_error.detach()) if self.best_error is not None else float(err.detach())
                is_mismatch = self.is_mirrored and (best_err >= self.epsilon)
                reward = 100.0 if is_mismatch else -100.0
                terminated = True
            else:
                reward = -float(err.detach())
                self._update_no_improve(err)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        truncated = (self.step_count >= self.max_steps) and (not terminated)
        if self.total_rotation >= self.max_total_rotation and not terminated:
            truncated = True

        info = {
            "angle": float(self.current_angle),
            "error": float(err.detach().cpu()),
            "best_error": float(self.best_error.detach().cpu()) if self.best_error is not None else float(err.detach().cpu()),
            "best_angle": float(self.best_angle),
            "is_mirrored": bool(self.is_mirrored),
            "no_improve_steps": int(self.no_improve_steps),
            "total_rotation": float(self.total_rotation),
        }
        return self._obs(), float(reward), terminated, truncated, info
