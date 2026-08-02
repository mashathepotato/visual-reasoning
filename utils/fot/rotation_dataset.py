from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from .colored_shapes_ops import random_colored_rectangles
from .rotation_ops import rotate_tensor
from .tetris_ops import get_tetris_tensor
from .reproducibility import sha256_file


VALID_SPLITS = ("train", "validation", "test_id", "test_ood_angle")


def load_rotation_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported rotation manifest: {path}")
    if manifest.get("task") not in {"tetris_rotation", "colored_rotation"}:
        raise ValueError(f"Unknown rotation task in {path}: {manifest.get('task')!r}")
    split_descriptors = manifest.get("splits")
    if not isinstance(split_descriptors, dict):
        raise ValueError(f"Manifest {path} has no splits object")

    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in VALID_SPLITS:
        descriptor = split_descriptors.get(split_name)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise ValueError(f"Manifest {path} is missing descriptor for split {split_name!r}")
        split_path = path.parent / descriptor["path"]
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        if sha256_file(split_path) != descriptor.get("sha256"):
            raise ValueError(f"Split hash mismatch: {split_path}")
        rows: List[Dict[str, Any]] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {split_path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Split row in {split_path}:{line_number} must be an object")
                rows.append(row)
        if len(rows) != descriptor.get("n"):
            raise ValueError(f"Split count mismatch: {split_path}")
        splits[split_name] = rows
    manifest["splits"] = splits

    seen_ids = set()
    for split_name in VALID_SPLITS:
        rows = splits[split_name]
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row in {split_name!r} must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid sample_id in split {split_name!r}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r}")
            seen_ids.add(sample_id)
            if row.get("label") not in (0, 1):
                raise ValueError(f"Invalid binary label for {sample_id!r}")
    return manifest


class RotationPairDataset(Dataset):
    """Stateless pair renderer backed by an immutable JSON manifest.

    Label 1 means SAME (rotation only); label 0 means DIFFERENT (mirrored then
    rotated). Every image is a pure function of the manifest row.
    """

    def __init__(self, manifest_path: Path, split: str):
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_rotation_manifest(self.manifest_path)
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}")
        self.split = split
        self.rows: List[Mapping[str, Any]] = self.manifest["splits"][split]
        self.task = str(self.manifest["task"])
        self.image_size = int(self.manifest["image_size"])
        self.num_shapes = int(self.manifest.get("num_shapes", 4))

    def __len__(self) -> int:
        return len(self.rows)

    def _target(self, row: Mapping[str, Any]) -> torch.Tensor:
        if self.task == "tetris_rotation":
            return get_tetris_tensor(str(row["base_id"]), self.image_size, channels=3).clone()
        generator = random.Random(int(row["render_seed"]))
        return random_colored_rectangles(
            self.image_size,
            self.image_size,
            num_shapes=self.num_shapes,
            rng=generator,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        target = self._target(row)
        label = int(row["label"])
        source = target.clone() if label == 1 else torch.flip(target, dims=[2])
        source = rotate_tensor(source, float(row["angle_deg"]), pad_value=0.0).clamp(0.0, 1.0)
        pair = torch.cat([source, target], dim=0)
        return {
            "pair": pair,
            "source": source,
            "target": target,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "base_id": str(row["base_id"]),
            "angle_deg": torch.tensor(float(row["angle_deg"]), dtype=torch.float32),
        }


def nested_fraction(dataset: Dataset, fraction: float) -> Dataset:
    """Return the deterministic leading subset used by all methods."""
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if float(fraction) == 1.0:
        return dataset
    count = max(1, int(len(dataset) * float(fraction)))
    return Subset(dataset, list(range(count)))


def split_sample_ids(manifest: Mapping[str, Any], split: str) -> Sequence[str]:
    return [str(row["sample_id"]) for row in manifest["splits"][split]]
