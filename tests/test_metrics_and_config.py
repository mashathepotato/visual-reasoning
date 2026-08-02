from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.fot.aggregation import aggregate_run_directories
from utils.fot.experiment_config import validate_experiment_config
from utils.fot.metrics import binary_auc, binary_classification_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


class MetricsAndConfigTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        metrics = binary_classification_metrics(labels, probabilities)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(binary_auc(labels, probabilities), 1.0)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tn"], 2)

    def test_minimal_config_validates(self) -> None:
        config = {
            "schema_version": 1,
            "experiment": {"name": "test", "seed": 0, "device": "cpu"},
            "dataset": {
                "task": "colored_rotation",
                "manifest": "manifest.json",
                "image_size": 64,
                "train_fraction": 1.0,
            },
            "model": {"type": "vit", "input_channels": 6, "num_classes": 2},
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "num_workers": 0,
            },
            "evaluation": {"checkpoint_selection": "validation_accuracy", "splits": ["test_id"]},
            "transition": {"type": "none"},
            "controller": {"type": "direct_supervised_classifier"},
        }
        validate_experiment_config(config)

    def test_aggregation_marks_missing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "seed0"
            missing = root / "seed1"
            complete.mkdir()
            missing.mkdir()
            config = {
                "schema_version": 1,
                "experiment": {"name": "test", "seed": 0, "device": "cpu"},
                "dataset": {"task": "colored_rotation"},
                "model": {"type": "vit"},
            }
            (complete / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
            summary = {
                "experiment_name": "colored_rotation_vit_smoke",
                "seed": 0,
                "task": "colored_rotation",
                "model": "vit",
                "metrics": {"test_id": {"accuracy": 0.5, "n": 8}},
            }
            (complete / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            aggregate = aggregate_run_directories([complete, missing])
            self.assertEqual(aggregate["seeds"], [0])
            self.assertEqual(len(aggregate["missing_or_failed_runs"]), 1)


if __name__ == "__main__":
    unittest.main()
