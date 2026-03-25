from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def _configure_matplotlib() -> None:
	# Avoid matplotlib/fontconfig trying to write under user home dirs.
	mpl_cfg = Path(tempfile.gettempdir()) / "mplconfig"
	mpl_cfg.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
	os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

	import matplotlib  # noqa: WPS433

	matplotlib.use("Agg")
	import matplotlib as mpl  # noqa: WPS433

	mpl.rcParams.update(
		{
			"figure.dpi": 150,
			"savefig.dpi": 300,
			"font.size": 10,
			"axes.labelsize": 10,
			"axes.titlesize": 10,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
			"legend.fontsize": 9,
			"axes.grid": True,
			"grid.alpha": 0.25,
			"grid.linestyle": "--",
			"grid.linewidth": 0.6,
			"axes.spines.top": False,
			"axes.spines.right": False,
		}
	)


def _moving_average(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
	if window <= 1:
		epochs = np.arange(1, len(x) + 1, dtype=np.float64)
		return epochs, x

	k = int(window)
	k = min(k, len(x))
	kernel = np.ones(k, dtype=np.float64) / float(k)
	y = np.convolve(x, kernel, mode="valid")
	epochs = np.arange(k, len(x) + 1, dtype=np.float64)
	return epochs, y


def _extract_train_losses(ckpt: object) -> Iterable[float]:
	if isinstance(ckpt, dict):
		if "train_losses" in ckpt and isinstance(ckpt["train_losses"], (list, tuple)):
			return ckpt["train_losses"]
		# Many of our `.pth` files are plain `state_dict`s (OrderedDict of tensors).
		# In that case there is no training history to plot.
		if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
			raise KeyError(
				"Checkpoint looks like a weights-only `state_dict` (e.g. `models/rotator_l1_500e_10k.pth`) "
				"and does not include training history. Provide a checkpoint containing a `train_losses` list, "
				"or pass `--losses` pointing to a saved loss array (e.g. a `.npy`)."
			)
	raise KeyError("Checkpoint does not contain a `train_losses` list.")


def _load_losses_file(path: Path) -> np.ndarray:
	if not path.exists():
		raise FileNotFoundError(path)

	if path.suffix.lower() == ".npy":
		arr = np.load(path)
		return np.asarray(arr, dtype=np.float64).reshape(-1)

	if path.suffix.lower() == ".json":
		import json  # noqa: WPS433

		payload = json.loads(path.read_text(encoding="utf-8"))
		if isinstance(payload, list):
			return np.asarray(payload, dtype=np.float64).reshape(-1)
		if isinstance(payload, dict) and "train_losses" in payload:
			return np.asarray(payload["train_losses"], dtype=np.float64).reshape(-1)
		raise ValueError(f"Unsupported JSON format in: {path} (expected list or dict with `train_losses`).")

	raise ValueError(f"Unsupported losses file type: {path} (supported: .npy, .json)")


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot flow-matching training loss from a checkpoint with `train_losses`.")
	parser.add_argument(
		"--ckpt",
		type=Path,
		default=Path("models/rotator_colors_500e_checkpoint.pth"),
		help="Path to FM checkpoint containing `train_losses` (default: models/rotator_colors_500e_checkpoint.pth).",
	)
	parser.add_argument(
		"--losses",
		type=Path,
		default=None,
		help="Optional: path to a saved loss array (.npy) or JSON list/dict with `train_losses`.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Flow Matching Training Loss",
		help="Plot title (default: Flow Matching Training Loss).",
	)
	parser.add_argument(
		"--fig-width",
		type=float,
		default=7.0,
		help="Figure width in inches (default: 7.0).",
	)
	parser.add_argument(
		"--fig-height",
		type=float,
		default=3.6,
		help="Figure height in inches (default: 3.6).",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path("diagrams"),
		help="Output directory (default: diagrams).",
	)
	parser.add_argument(
		"--stem",
		type=str,
		default="fm_training_tetris",
		help="Output filename stem (default: fm_training_tetris).",
	)
	parser.add_argument(
		"--total-epochs",
		type=int,
		default=500,
		help="Optional x-axis max epoch for paper plots (default: 500).",
	)
	parser.add_argument(
		"--smooth",
		type=int,
		default=10,
		help="Moving-average window for smoothing (default: 10).",
	)
	args = parser.parse_args()

	_configure_matplotlib()
	import matplotlib.pyplot as plt  # noqa: WPS433

	if args.losses is not None:
		losses = _load_losses_file(args.losses)
	else:
		ckpt = torch.load(args.ckpt, map_location="cpu")
		losses = np.asarray(list(_extract_train_losses(ckpt)), dtype=np.float64)
	if losses.size == 0:
		raise RuntimeError("Loss list is empty.")

	epochs = np.arange(1, losses.size + 1, dtype=np.float64)
	sm_epochs, sm = _moving_average(losses, args.smooth)

	fig, ax = plt.subplots(1, 1, figsize=(args.fig_width, args.fig_height))
	ax.plot(epochs, losses, label="epoch loss", linewidth=1.5, alpha=0.35)
	ax.plot(sm_epochs, sm, label=f"{args.smooth}-epoch moving avg", linewidth=2.2)

	ax.set_title(args.title)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Velocity regression loss")

	if args.total_epochs and args.total_epochs > 0:
		ax.set_xlim(1, max(args.total_epochs, int(epochs.max())))
		if losses.size < args.total_epochs:
			ax.axvline(losses.size, color="black", linewidth=1.0, alpha=0.4)
			ax.text(
				losses.size,
				float(np.nanmin(losses)),
				f"  recorded={losses.size}",
				rotation=90,
				va="bottom",
				ha="left",
				alpha=0.7,
			)

	ax.legend(frameon=False, loc="upper right")
	fig.tight_layout()

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pdf_path = args.out_dir / f"{args.stem}.pdf"
	png_path = args.out_dir / f"{args.stem}.png"
	fig.savefig(pdf_path, bbox_inches="tight")
	fig.savefig(png_path, bbox_inches="tight")
	plt.close(fig)

	if args.losses is not None:
		print(f"Losses: {args.losses}")
	else:
		print(f"Checkpoint: {args.ckpt}")
	print(f"Epochs recorded: {losses.size}")
	print(f"Saved: {pdf_path}")
	print(f"Saved: {png_path}")


if __name__ == "__main__":
	main()
