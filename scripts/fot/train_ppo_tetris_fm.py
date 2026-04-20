from __future__ import annotations

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.fot.checkpoint_utils import load_state_dict
from utils.fot.dino_utils import create_dinov3
from utils.fot.envs_rotation import RotationEnvFM
from utils.fot.models import CorrectorUNet, FastRotator
from utils.fot.tetris_ops import make_tetris_pair_sampler
from utils.fot.torch_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO controller for Tetris rotation using frozen FM dynamics.")
    p.add_argument("--rotator", type=str, default="models/rotator_fm_tetris_with_corrector.pth")
    p.add_argument("--corrector", type=str, default="models/rotator_fm_tetris_corrector.pth")
    p.add_argument("--fm-steps", type=int, default=10)
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=str, default="models/ppo_tetris_fm_controller")
    p.add_argument("--log-dir", type=str, default="logs/ppo_tetris_fm")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    rotator_sd = load_state_dict(args.rotator, device)
    flow_dim = int(rotator_sd["inc.net.0.weight"].shape[0])
    backbone_dim = int(rotator_sd["cond_proj.weight"].shape[1])
    fm_model = FastRotator(in_ch=1, out_ch=1, backbone_dim=backbone_dim, flow_dim=flow_dim).to(device)
    fm_model.load_state_dict(rotator_sd)
    fm_model.eval()
    for p in fm_model.parameters():
        p.requires_grad = False

    corr_sd = load_state_dict(args.corrector, device)
    corr_base = int(corr_sd["inc.net.0.weight"].shape[0])
    corrector = CorrectorUNet(in_ch=1, out_ch=1, base_ch=corr_base).to(device)
    corrector.load_state_dict(corr_sd)
    corrector.eval()
    for p in corrector.parameters():
        p.requires_grad = False

    dino = create_dinov3(device=device)

    image_shape = (3, 64, 64)
    sampler = make_tetris_pair_sampler(image_shape=image_shape, mirror_prob=0.5, angle_step=5.0)

    def make_env():
        env = RotationEnvFM(
            image_shape=image_shape,
            fm_model=fm_model,
            dino_model=dino,
            corrector=corrector,
            fm_steps=args.fm_steps,
            max_steps=180,
            device=device,
            pair_sampler=sampler,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(log_dir), ["stdout", "csv"])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=None,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        policy_kwargs={"normalize_images": False},
        seed=args.seed,
    )
    model.set_logger(logger)
    model.learn(total_timesteps=int(args.total_timesteps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print("Saved PPO controller:", out)


if __name__ == "__main__":
    main()
