from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np


def _configure_matplotlib() -> None:
	# Avoid matplotlib trying to write under ~/.matplotlib in restricted environments.
	mpl_cfg = Path(tempfile.gettempdir()) / "mplconfig"
	mpl_cfg.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

	# Also steer fontconfig cache away from user dirs that may not be writable.
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


def _read_progress_csv(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def _col_as_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
	out: list[float] = []
	for r in rows:
		v = r.get(key, "")
		if v is None or v == "":
			out.append(np.nan)
		else:
			out.append(float(v))
	return np.asarray(out, dtype=np.float64)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot Stable-Baselines3 PPO training curves from progress.csv.")
	parser.add_argument(
		"--log",
		type=Path,
		default=Path("logs/ppo_tetris_fm/progress.csv"),
		help="Path to Stable-Baselines3 progress.csv (default: logs/ppo_tetris_fm/progress.csv).",
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
		default="ppo_training_tetris",
		help="Output filename stem (default: ppo_training_tetris).",
	)
	args = parser.parse_args()

	_configure_matplotlib()
	import matplotlib.pyplot as plt  # noqa: WPS433

	rows = _read_progress_csv(args.log)
	if not rows:
		raise RuntimeError(f"No rows found in: {args.log}")

	# x-axis: PPO update/iteration index (Stable-Baselines3 uses time/iterations).
	x = _col_as_float(rows, "time/iterations")
	if np.all(np.isnan(x)):
		# Fall back: use row index.
		x = np.arange(len(rows), dtype=np.float64)

	metrics = {
		"policy": _col_as_float(rows, "train/policy_gradient_loss"),
		"entropy": _col_as_float(rows, "train/entropy_loss"),
		"value": _col_as_float(rows, "train/value_loss"),
		"total": _col_as_float(rows, "train/loss"),
	}

	fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6), sharex=True)

	# Left: policy + entropy
	ax = axes[0]
	ax.plot(x, metrics["policy"], label="policy", linewidth=2.0)
	ax.plot(x, metrics["entropy"], label="entropy", linewidth=2.0)
	ax.set_title("Policy/Entropy")
	ax.set_xlabel("Update")
	ax.set_ylabel("Loss")
	ax.legend(frameon=False, loc="lower right")

	# Right: value + total
	ax = axes[1]
	ax.plot(x, metrics["value"], label="value", linewidth=2.0)
	ax.plot(x, metrics["total"], label="total", linewidth=2.0)
	ax.set_title("Value/Total")
	ax.set_xlabel("Update")
	ax.set_ylabel("Loss")
	ax.legend(frameon=False, loc="upper right")

	fig.tight_layout()

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pdf_path = args.out_dir / f"{args.stem}.pdf"
	png_path = args.out_dir / f"{args.stem}.png"
	fig.savefig(pdf_path, bbox_inches="tight")
	fig.savefig(png_path, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved: {pdf_path}")
	print(f"Saved: {png_path}")


if __name__ == "__main__":
	main()
