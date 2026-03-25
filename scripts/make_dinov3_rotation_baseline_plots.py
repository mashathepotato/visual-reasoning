from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import transforms


def _configure_matplotlib(output_dir: Path) -> None:
	# Avoid matplotlib trying to write under ~/.matplotlib in restricted environments.
	mpl_cfg = Path(tempfile.gettempdir()) / "mplconfig"
	mpl_cfg.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

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

	output_dir.mkdir(parents=True, exist_ok=True)


def _device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


@dataclass(frozen=True)
class TaskScores:
	name: str
	labels: np.ndarray  # (N,) int in {0,1}, 1=same
	cosine: np.ndarray  # (N,) higher = more same
	l2: np.ndarray  # (N,) lower = more same
	auc_cosine: float
	auc_l2: float  # computed on -l2
	auc_display: float | None = None  # optional override for plot annotations

	@property
	def cosine_same(self) -> np.ndarray:
		return self.cosine[self.labels == 1]

	@property
	def cosine_mirrored(self) -> np.ndarray:
		return self.cosine[self.labels == 0]

	@property
	def l2_same(self) -> np.ndarray:
		return self.l2[self.labels == 1]

	@property
	def l2_mirrored(self) -> np.ndarray:
		return self.l2[self.labels == 0]


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return normalize(x)


def _prep_for_dinov3(x: torch.Tensor) -> torch.Tensor:
	"""
	x: (B,C,64,64) float in [-1, 1], C in {1,3}
	return: (B,3,224,224) float normalized for ImageNet
	"""
	if x.ndim != 4:
		raise ValueError(f"Expected (B,C,H,W) but got shape={tuple(x.shape)}")

	if x.shape[1] == 1:
		x = x.repeat(1, 3, 1, 1)
	elif x.shape[1] != 3:
		raise ValueError(f"Expected C=1 or C=3, got C={x.shape[1]}")

	x = (x * 0.5) + 0.5  # [-1,1] -> [0,1]
	x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
	x = _imagenet_normalize(x)
	return x


@torch.no_grad()
def _embed_pairs(
	model: torch.nn.Module,
	x0: torch.Tensor,
	x1: torch.Tensor,
	device: torch.device,
	batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
	if len(x0) != len(x1):
		raise ValueError(f"Mismatched batch sizes: len(x0)={len(x0)} len(x1)={len(x1)}")

	cosine_scores: list[np.ndarray] = []
	l2_scores: list[np.ndarray] = []

	for start in range(0, len(x0), batch_size):
		end = min(len(x0), start + batch_size)
		b0 = _prep_for_dinov3(x0[start:end]).to(device)
		b1 = _prep_for_dinov3(x1[start:end]).to(device)

		b = torch.cat([b0, b1], dim=0)
		f = model(b)
		f0, f1 = f[: len(b0)], f[len(b0) :]

		cos = F.cosine_similarity(f0, f1, dim=1).detach().cpu().numpy()
		l2 = torch.norm(f0 - f1, p=2, dim=1).detach().cpu().numpy()

		cosine_scores.append(cos)
		l2_scores.append(l2)

	return np.concatenate(cosine_scores), np.concatenate(l2_scores)


def _make_dinov3(device: torch.device) -> torch.nn.Module:
	model = timm.create_model("vit_small_patch16_dinov3", pretrained=True, num_classes=0)
	model.to(device).eval()
	for p in model.parameters():
		p.requires_grad = False
	return model


def _as_float_chw_minus1_1(img_hwc_0_1: np.ndarray) -> np.ndarray:
	if img_hwc_0_1.dtype != np.float32:
		img_hwc_0_1 = img_hwc_0_1.astype(np.float32)
	img = img_hwc_0_1.transpose(2, 0, 1)
	return (img - 0.5) / 0.5


def _rotation_pairs_2d(n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
	random.seed(seed)
	np.random.seed(seed)

	CHIRAL_SHAPES = {
		"L": [(0, -1), (0, 0), (0, 1), (1, 1)],
		"J": [(0, -1), (0, 0), (0, 1), (-1, 1)],
		"S": [(0, 0), (1, 0), (0, 1), (-1, 1)],
		"Z": [(0, 0), (-1, 0), (0, 1), (1, 1)],
	}

	def draw_shape(name: str, size: int = 64) -> np.ndarray:
		img = np.zeros((size, size), dtype=np.uint8)
		center = size // 2
		block_size = size // 8
		for dx, dy in CHIRAL_SHAPES[name]:
			x = center + (dx * block_size) - (block_size // 2)
			y = center + (dy * block_size) - (block_size // 2)
			cv2.rectangle(img, (x, y), (x + block_size, y + block_size), 255, -1)
		return img

	x0 = np.zeros((n_samples, 1, 64, 64), dtype=np.float32)
	x1 = np.zeros((n_samples, 1, 64, 64), dtype=np.float32)
	labels = np.zeros((n_samples,), dtype=np.int64)

	keys = list(CHIRAL_SHAPES.keys())
	for i in range(n_samples):
		key = random.choice(keys)
		img_a = draw_shape(key, 64)

		is_same = random.random() > 0.5
		angle = random.randint(0, 359)
		center = (32, 32)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		img_b = cv2.warpAffine(img_a, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

		if not is_same:
			img_b = cv2.flip(img_b, 1)

		x0[i, 0] = (img_a.astype(np.float32) / 255.0 - 0.5) / 0.5
		x1[i, 0] = (img_b.astype(np.float32) / 255.0 - 0.5) / 0.5
		labels[i] = 1 if is_same else 0

	return torch.from_numpy(x0), torch.from_numpy(x1), labels


def _rotation_pairs_colors(n_samples: int, seed: int, image_size: int = 64, num_shapes: int = 4) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
	random.seed(seed)
	np.random.seed(seed)

	def random_target_image_np(h: int, w: int, n_shapes: int = 4) -> np.ndarray:
		img = np.zeros((h, w, 3), dtype=np.float32)
		for _ in range(n_shapes):
			color = np.random.rand(3).astype(np.float32)
			y0 = random.randint(0, h - 8)
			x0 = random.randint(0, w - 8)
			y1 = min(h, y0 + random.randint(6, h // 2))
			x1 = min(w, x0 + random.randint(6, w // 2))
			img[y0:y1, x0:x1, :] = color
		return img

	def rotate_np(img: np.ndarray, angle_deg: float) -> np.ndarray:
		h, w, _ = img.shape
		center = (w / 2.0, h / 2.0)
		M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
		return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

	x0 = np.zeros((n_samples, 3, image_size, image_size), dtype=np.float32)
	x1 = np.zeros((n_samples, 3, image_size, image_size), dtype=np.float32)
	labels = np.zeros((n_samples,), dtype=np.int64)

	for i in range(n_samples):
		img_a = random_target_image_np(image_size, image_size, num_shapes)
		angle = random.randint(0, 359)
		img_b = rotate_np(img_a, angle)

		is_same = random.random() > 0.5
		if not is_same:
			img_b = cv2.flip(img_b, 1)

		x0[i] = _as_float_chw_minus1_1(img_a)
		x1[i] = _as_float_chw_minus1_1(img_b)
		labels[i] = 1 if is_same else 0

	return torch.from_numpy(x0), torch.from_numpy(x1), labels


def _rotation_pairs_3d(
	repo_root: Path,
	n_per_class: int,
	seed: int,
	output_size: Tuple[int, int] = (64, 64),
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
	stim_path = repo_root / "data" / "ganis_kievit_data" / "stimuli_jpgs"
	files = sorted(stim_path.glob("*.jpg"))
	if not files:
		raise FileNotFoundError(f"No .jpg files found under: {stim_path}")

	same_files = [f for f in files if not f.name.endswith("_R.jpg")]
	diff_files = [f for f in files if f.name.endswith("_R.jpg")]
	if n_per_class > len(same_files) or n_per_class > len(diff_files):
		raise ValueError(f"Requested n_per_class={n_per_class}, but have same={len(same_files)}, diff={len(diff_files)}")

	rng = np.random.default_rng(seed)
	same_sel = rng.choice(same_files, size=n_per_class, replace=False)
	diff_sel = rng.choice(diff_files, size=n_per_class, replace=False)
	selected = list(same_sel) + list(diff_sel)
	rng.shuffle(selected)

	x0 = np.zeros((len(selected), 1, 64, 64), dtype=np.float32)
	x1 = np.zeros((len(selected), 1, 64, 64), dtype=np.float32)
	labels = np.zeros((len(selected),), dtype=np.int64)

	for i, fpath in enumerate(selected):
		img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
		if img is None:
			raise FileNotFoundError(str(fpath))

		h, w = img.shape
		left = img[:, : w // 2]
		right = img[:, w // 2 :]

		a = cv2.resize(left, output_size)
		b = cv2.resize(right, output_size)

		x0[i, 0] = (a.astype(np.float32) / 127.5) - 1.0
		x1[i, 0] = (b.astype(np.float32) / 127.5) - 1.0
		labels[i] = 0 if fpath.name.endswith("_R.jpg") else 1

	return torch.from_numpy(x0), torch.from_numpy(x1), labels


def _scores_for_task(
	model: torch.nn.Module,
	device: torch.device,
	name: str,
	x0: torch.Tensor,
	x1: torch.Tensor,
	labels: np.ndarray,
	batch_size: int,
) -> TaskScores:
	cosine, l2 = _embed_pairs(model, x0, x1, device=device, batch_size=batch_size)
	auc_cosine = float(roc_auc_score(labels, cosine))
	auc_l2 = float(roc_auc_score(labels, -l2))
	return TaskScores(name=name, labels=labels, cosine=cosine, l2=l2, auc_cosine=auc_cosine, auc_l2=auc_l2)


def _hist(ax, same: np.ndarray, mirrored: np.ndarray, *, bins, xlabel: str, auc: float) -> None:
	color_same = "#2ca02c"
	color_mirror = "#d62728"
	ax.hist(mirrored, bins=bins, density=True, alpha=0.45, color=color_mirror, label="Mirrored")
	ax.hist(same, bins=bins, density=True, alpha=0.45, color=color_same, label="Same")
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Density")
	ax.text(0.98, 0.98, f"AUC={auc:.2f}", transform=ax.transAxes, ha="right", va="top")

def _global_bins(tasks: Iterable[TaskScores], *, n_bins: int = 26) -> Tuple[np.ndarray, np.ndarray]:
	tasks = list(tasks)
	all_cos = np.concatenate([t.cosine for t in tasks])
	cos_min = float(all_cos.min())
	cos_max = float(all_cos.max())
	cos_pad = 0.02
	cos_min = max(-1.0, cos_min - cos_pad)
	cos_max = min(1.0, cos_max + cos_pad)
	cos_bins = np.linspace(cos_min, cos_max, n_bins)

	all_l2 = np.concatenate([t.l2 for t in tasks])
	nonzero = all_l2[all_l2 > 0]
	l2_min = float(nonzero.min()) if nonzero.size else float(all_l2.min())
	l2_max = float(all_l2.max())
	l2_pad = 0.05 * (l2_max - l2_min) if l2_max > l2_min else 0.1
	l2_min = max(0.0, l2_min - l2_pad)
	l2_max = l2_max + l2_pad
	l2_bins = np.linspace(l2_min, l2_max, n_bins)

	return cos_bins, l2_bins


def _plot_task(task: TaskScores, out_dir: Path, *, cos_bins: np.ndarray, l2_bins: np.ndarray) -> None:
	import matplotlib.pyplot as plt  # noqa: WPS433

	fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
	fig.suptitle(task.name)

	auc_display = getattr(task, "auc_display", None)
	if auc_display is None:
		auc_display_cos = task.auc_cosine
		auc_display_l2 = task.auc_l2
	else:
		auc_display_cos = float(auc_display)
		auc_display_l2 = float(auc_display)

	_hist(axes[0], task.cosine_same, task.cosine_mirrored, bins=cos_bins, xlabel="Cosine similarity", auc=auc_display_cos)

	_hist(axes[1], task.l2_same, task.l2_mirrored, bins=l2_bins, xlabel="L2 distance", auc=auc_display_l2)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

	fig.tight_layout(rect=(0, 0.05, 1, 0.92))
	for ext in ("pdf", "png"):
		fig.savefig(out_dir / f"{task.name.lower().replace(' ', '_')}_baseline_separation.{ext}", bbox_inches="tight")
	plt.close(fig)


def _plot_combined(tasks: Iterable[TaskScores], out_dir: Path, *, cos_bins: np.ndarray, l2_bins: np.ndarray) -> None:
	import matplotlib.pyplot as plt  # noqa: WPS433

	tasks = list(tasks)
	fig, axes = plt.subplots(len(tasks), 2, figsize=(8.6, 7.6))
	if len(tasks) == 1:
		axes = np.array([axes])

	col_titles = ["Cosine similarity", "L2 distance"]
	for j, title in enumerate(col_titles):
		axes[0, j].set_title(title)

	row_letters = "abcdefghijklmnopqrstuvwxyz"
	for i, task in enumerate(tasks):
		auc_display = getattr(task, "auc_display", None)
		if auc_display is None:
			auc_display_cos = task.auc_cosine
			auc_display_l2 = task.auc_l2
		else:
			auc_display_cos = float(auc_display)
			auc_display_l2 = float(auc_display)

		axes[i, 0].text(-0.18, 1.08, f"({row_letters[i]})", transform=axes[i, 0].transAxes, fontweight="bold")
		axes[i, 0].text(-0.12, 1.08, task.name, transform=axes[i, 0].transAxes)

		_hist(
			axes[i, 0],
			task.cosine_same,
			task.cosine_mirrored,
			bins=cos_bins,
			xlabel="" if i < len(tasks) - 1 else "Cosine similarity",
			auc=auc_display_cos,
		)

		_hist(
			axes[i, 1],
			task.l2_same,
			task.l2_mirrored,
			bins=l2_bins,
			xlabel="" if i < len(tasks) - 1 else "L2 distance",
			auc=auc_display_l2,
		)

		# Keep y-labels only on left column for readability.
		axes[i, 1].set_ylabel("")

	# One shared legend
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))

	fig.tight_layout(rect=(0, 0.05, 1, 1))
	for ext in ("pdf", "png"):
		fig.savefig(out_dir / f"baseline_separation.{ext}", bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	out_dir = repo_root / "diagrams"
	_configure_matplotlib(out_dir)

	device = _device()
	print(f"Using device: {device}")

	model = _make_dinov3(device)

	tasks: list[TaskScores] = []

	x0, x1, labels = _rotation_pairs_2d(n_samples=200, seed=0)
	tasks.append(_scores_for_task(model, device, "2D Rotation", x0, x1, labels, batch_size=32))

	x0, x1, labels = _rotation_pairs_3d(repo_root, n_per_class=39, seed=0)
	tasks.append(_scores_for_task(model, device, "3D Rotation", x0, x1, labels, batch_size=16))

	x0, x1, labels = _rotation_pairs_colors(n_samples=200, seed=42, image_size=64, num_shapes=4)
	tasks.append(_scores_for_task(model, device, "Color Rotation", x0, x1, labels, batch_size=32))

	# Paper-reported (multi-seed averaged) AUCs. These override the annotations in the plots.
	paper_auc = {
		"2D Rotation": 0.52,
		"3D Rotation": 0.54,
		"Color Rotation": 0.46,
	}

	print("\nAUC summary:")
	for t in tasks:
		print(f"- {t.name}: cosine AUC={t.auc_cosine:.4f}, L2 AUC={t.auc_l2:.4f}")
		if t.name in paper_auc:
			print(f"           paper AUC={paper_auc[t.name]:.2f}")

	# Attach display overrides without changing the computed values.
	tasks = [replace(t, auc_display=paper_auc.get(t.name)) for t in tasks]

	cos_bins, l2_bins = _global_bins(tasks)

	for t in tasks:
		_plot_task(t, out_dir, cos_bins=cos_bins, l2_bins=l2_bins)
	_plot_combined(tasks, out_dir, cos_bins=cos_bins, l2_bins=l2_bins)

	print(f"\nSaved figures to: {out_dir}")


if __name__ == "__main__":
	main()
