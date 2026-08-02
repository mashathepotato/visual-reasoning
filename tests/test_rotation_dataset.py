from __future__ import annotations

import unittest
from pathlib import Path

import torch

from utils.fot.rotation_dataset import RotationPairDataset, load_rotation_manifest, split_sample_ids


REPO_ROOT = Path(__file__).resolve().parents[1]


class RotationDatasetTests(unittest.TestCase):
    def test_manifest_splits_and_angle_sets_are_disjoint(self) -> None:
        path = REPO_ROOT / "data" / "splits" / "colored_rotation_v1.json"
        manifest = load_rotation_manifest(path)
        split_sets = {name: set(split_sample_ids(manifest, name)) for name in manifest["splits"]}
        names = list(split_sets)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertTrue(split_sets[left].isdisjoint(split_sets[right]))
        id_angles = {row["angle_deg"] for row in manifest["splits"]["test_id"]}
        ood_angles = {row["angle_deg"] for row in manifest["splits"]["test_ood_angle"]}
        self.assertTrue(id_angles.isdisjoint(ood_angles))

    def test_colored_samples_are_stateless(self) -> None:
        path = REPO_ROOT / "data" / "splits" / "colored_rotation_v1.json"
        dataset = RotationPairDataset(path, "train")
        first = dataset[7]
        _ = dataset[2]
        repeated = dataset[7]
        self.assertEqual(first["sample_id"], repeated["sample_id"])
        self.assertTrue(torch.equal(first["pair"], repeated["pair"]))
        self.assertEqual(int(first["label"]), int(repeated["label"]))

    def test_labels_are_balanced(self) -> None:
        path = REPO_ROOT / "data" / "splits" / "tetris_rotation_v1.json"
        manifest = load_rotation_manifest(path)
        for rows in manifest["splits"].values():
            labels = [row["label"] for row in rows]
            self.assertEqual(labels.count(0), labels.count(1))


if __name__ == "__main__":
    unittest.main()
