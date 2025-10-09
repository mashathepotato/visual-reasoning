import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2


class AngleDataset(Dataset):

	def __init__(self, images: List[np.ndarray], angles: List[int], transform=None):
		self.images = images
		self.angles = angles
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img = self.images[idx]
		angle = self.angles[idx]
		# BGR to RGB
		if len(img.shape) == 3 and img.shape[2] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
		if self.transform:
			img = self.transform(img)
		return img, angle


def get_dino_model(model_name: str = 'vit_small_patch16_224.dino'):
	# timm provides dino-pretrained weights with suffix .dino or .dinov2 when available
	model = timm.create_model(model_name, pretrained=True)
	model.eval()
	# Remove classifier, use global pooled features
	if hasattr(model, 'reset_classifier'):
		model.reset_classifier(0)
	return model


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[np.ndarray, np.ndarray]:
	model = model.to(device)
	feats = []
	labels = []
	for images, angles in loader:
		images = images.to(device)
		outputs = model(images)
		# outputs is (B, D) after reset_classifier(0)
		feats.append(outputs.detach().cpu().numpy())
		labels.append(angles.numpy())
	features = np.concatenate(feats, axis=0)
	labels_np = np.concatenate(labels, axis=0)
	return features, labels_np


def angle_to_class_indices(angles: List[int]) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
	unique = sorted(set(angles))
	mapping = {a: i for i, a in enumerate(unique)}
	reverse = {i: a for a, i in mapping.items()}
	classes = np.array([mapping[a] for a in angles], dtype=np.int64)
	return classes, mapping, reverse


def build_transforms(image_size: int = 224) -> T.Compose:
	# For ViT/DINO expected input size
	return T.Compose([
		T.ConvertImageDtype(torch.float),
		T.Resize((image_size, image_size)),
		T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	])


def run_linear_probe(images: List[np.ndarray], angles: List[int],
					 model_name: str = 'vit_small_patch16_224.dino',
					 batch_size: int = 32, image_size: int = 224,
					 test_size: float = 0.2, val_size: float = 0.2,
					 random_state: int = 42) -> Dict[str, Any]:
	# Map to class indices
	class_indices, angle_to_cls, cls_to_angle = angle_to_class_indices(angles)

	# Split
	X_temp, X_test, y_temp, y_test = train_test_split(images, class_indices, test_size=test_size, random_state=random_state, stratify=class_indices)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp)

	# Datasets
	transform = build_transforms(image_size)
	train_ds = AngleDataset(X_train, y_train, transform=transform)
	val_ds = AngleDataset(X_val, y_val, transform=transform)
	test_ds = AngleDataset(X_test, y_test, transform=transform)

	# Loaders
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

	# Model and feature extraction
	model = get_dino_model(model_name)
	train_feats, train_labels = extract_features(model, train_loader)
	val_feats, val_labels = extract_features(model, val_loader)
	test_feats, test_labels = extract_features(model, test_loader)

	# Linear probe (L2-regularized logistic regression)
	clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class='multinomial')
	clf.fit(train_feats, train_labels)
	val_acc = accuracy_score(val_labels, clf.predict(val_feats))
	test_acc = accuracy_score(test_labels, clf.predict(test_feats))

	return {
		'model_name': model_name,
		'num_classes': len(set(class_indices)),
		'train_samples': len(train_ds),
		'val_samples': len(val_ds),
		'test_samples': len(test_ds),
		'val_accuracy': float(val_acc),
		'test_accuracy': float(test_acc),
		'angle_to_class': angle_to_cls,
		'class_to_angle': cls_to_angle,
	}


def prepare_from_generators(dataset_generator, samples_per_type: int = 20, rotation_step: int = 5) -> Tuple[List[np.ndarray], List[int]]:
	# Use existing generation utilities to create mixed dataset
	dataset = dataset_generator.generate_mixed_dataset(samples_per_type=samples_per_type, rotation_step=rotation_step)
	images, angles = [], []
	for item in dataset:
		images.append(item['image'])
		angles.append(item['angle'])
	return images, angles


if __name__ == "__main__":
	from data_generation import LineGenerator, DatasetGenerator
	print("Preparing dataset...")
	lg = LineGenerator()
	dg = DatasetGenerator(lg)
	images, angles = prepare_from_generators(dg, samples_per_type=20, rotation_step=5)
	print(f"Total samples: {len(images)}")
	print("Running DINO evaluation (linear probe)...")
	results = run_linear_probe(images, angles, model_name='vit_small_patch16_224.dino')
	print(results)

