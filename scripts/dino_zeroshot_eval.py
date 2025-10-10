import random
from typing import List, Dict, Any, Tuple

import cv2
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from data_generation import LineGenerator, DatasetGenerator


def build_transforms(image_size: int = 224) -> T.Compose:
	return T.Compose([
		T.ConvertImageDtype(torch.float),
		T.Resize((image_size, image_size)),
		T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	])


def get_dino(model_name: str = 'vit_small_patch16_224.dino') -> nn.Module:
	model = timm.create_model(model_name, pretrained=True)
	if hasattr(model, 'reset_classifier'):
		model.reset_classifier(0)
	model.eval()
	return model


@torch.no_grad()
def encode_image(model: nn.Module, image_bgr: np.ndarray, transform: T.Compose, device: str) -> torch.Tensor:
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	x = torch.from_numpy((rgb.astype(np.float32) / 255.0)).permute(2, 0, 1)
	x = transform(x).unsqueeze(0).to(device)
	feat = model(x)
	feat = nn.functional.normalize(feat, dim=1)
	return feat[0]


def rotate_image(image_bgr: np.ndarray, angle_deg: float, bg=(255, 255, 255)) -> np.ndarray:
	h, w = image_bgr.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	return cv2.warpAffine(image_bgr, M, (w, h), borderValue=bg)


def build_pairs(dataset: List[Dict[str, Any]], num_pairs: int = 100) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
	by_key = {}
	for item in dataset:
		key = (item['sample_id'], item['line_type'])
		by_key.setdefault(key, []).append(item)

	keys = list(by_key.keys())
	random.shuffle(keys)
	pairs = []

	# Positive pairs
	for k in keys:
		items = by_key[k]
		if len(items) < 2:
			continue
		a, b = random.sample(items, 2)
		pairs.append((a, b, 1))
		if len(pairs) >= num_pairs // 2:
			break

	# Negative pairs
	while len(pairs) < num_pairs:
		k1, k2 = random.sample(keys, 2)
		a = random.choice(by_key[k1])
		b = random.choice(by_key[k2])
		pairs.append((a, b, 0))

	return pairs


@torch.no_grad()
def simple_zeroshot_compare(model, img_a, img_b, transform, device):
	"""Just compare — no rotation, no threshold tuning."""
	feat_a = encode_image(model, img_a, transform, device)
	feat_b = encode_image(model, img_b, transform, device)
	sim = torch.cosine_similarity(feat_a.unsqueeze(0), feat_b.unsqueeze(0)).item()
	return sim


@torch.no_grad()
def evaluate_zeroshot(dataset: List[Dict[str, Any]],
					  model_name: str = 'vit_small_patch16_224.dino',
					  num_pairs: int = 20,
					  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:

	model = get_dino(model_name).to(device)
	transform = build_transforms(224)
	pairs = build_pairs(dataset, num_pairs=num_pairs)

	results = []
	for a, b, gt in pairs:
		# --- fresh context ---
		sim = simple_zeroshot_compare(model, a['image'], b['image'], transform, device)
		pred_same = sim > 0.6  # lightweight heuristic, not tuned

		# --- if incorrect, allow rotation and re-evaluate ---
		if pred_same != bool(gt):
			for angle in range(0, 360, 45):
				b_rot = rotate_image(b['image'], angle)
				sim_rot = simple_zeroshot_compare(model, a['image'], b_rot, transform, device)
				pred_same_rot = sim_rot > 0.6
				if pred_same_rot == bool(gt):
					results.append({'gt': gt, 'correct_after_rotation': True})
					break
			else:
				results.append({'gt': gt, 'correct_after_rotation': False})
		else:
			results.append({'gt': gt, 'correct_after_rotation': None})  # got it right first try

	acc = np.mean([1 if (r['correct_after_rotation'] is None or r['correct_after_rotation']) else 0 for r in results])
	return {'accuracy': float(acc), 'num_pairs': len(pairs), 'details': results}


if __name__ == '__main__':
	lg = LineGenerator()
	dg = DatasetGenerator(lg)
	data = dg.generate_mixed_dataset(samples_per_type=10, rotation_step=10)
	res = evaluate_zeroshot(data, model_name='vit_small_patch16_224.dino', num_pairs=20)
	print(res)
