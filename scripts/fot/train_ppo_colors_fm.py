from __future__ import annotations

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.fot.colored_shapes_ops import make_colored_shapes_pair_sampler
from utils.fot.envs_rotation import RotationEnvColorsFM
from utils.fot.models import CondEncoder, FastRotator
from utils.fot.torch_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO controller for colored-rectangles rotation using frozen FM dynamics.")
    p.add_argument("--ckpt", type=str, default="models/rotator_colors_500e_checkpoint.pth")
    p.add_argument("--fm-steps", type=int, default=10)
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=str, default="models/ppo_colors_fm_controller")
    p.add_argument("--log-dir", type=str, default="logs/ppo_colors_fm")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    emb_dim = int(ckpt.get("emb_dim", 256))
    flow_dim = int(ckpt.get("flow_dim", 64))

    encoder = CondEncoder(in_ch=3, emb_dim=emb_dim).to(device)
    fm_model = FastRotator(in_ch=3, out_ch=3, backbone_dim=emb_dim, flow_dim=flow_dim).to(device)

    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    else:
        raise KeyError("Missing encoder_state_dict in checkpoint.")
    if "model_state_dict" in ckpt:
        fm_model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError("Missing model_state_dict in checkpoint.")

    encoder.eval()
    fm_model.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in fm_model.parameters():
        p.requires_grad = False

    image_shape = (3, 64, 64)
    sampler = make_colored_shapes_pair_sampler(image_shape=image_shape, mirror_prob=0.5, angle_step=5.0, num_shapes=4, seed=args.seed)

    def make_env():
        env = RotationEnvColorsFM(
            image_shape=image_shape,
            fm_model=fm_model,
            encoder=encoder,
            fm_steps=args.fm_steps,
            max_steps=180,
            device=device,
            pair_sampler=sampler,
        )
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
