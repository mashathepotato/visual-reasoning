from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.fot.checkpoint_utils import load_state_dict
from utils.fot.envs_maze import MazeEnvFMProgress
from utils.fot.models import MazeSketcher
from utils.fot.torch_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO controller for maze progress using frozen FM sketcher dynamics.")
    p.add_argument("--sketcher", type=str, default="models/maze_sketcher_fm.pth")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=str, default="models/ppo_maze_fm_progress")
    p.add_argument("--log-dir", type=str, default="logs/ppo_maze_fm_progress")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    sd = load_state_dict(args.sketcher, device)
    flow_dim = int(sd["inc.net.0.weight"].shape[0])
    sketcher = MazeSketcher(cond_ch=3, flow_dim=flow_dim).to(device)
    sketcher.load_state_dict(sd)
    sketcher.eval()
    for p in sketcher.parameters():
        p.requires_grad = False

    def make_env():
        env = MazeEnvFMProgress(sketcher=sketcher, device=device, seed=args.seed)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        policy_kwargs={"normalize_images": False},
        seed=args.seed,
    )
    model.learn(total_timesteps=int(args.total_timesteps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print("Saved PPO controller:", out)


if __name__ == "__main__":
    main()

