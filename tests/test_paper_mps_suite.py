from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fot.train_dinov3_pair_baseline import PairHead
from scripts.run_paper_mps_suite import stage_plan
from utils.fot.external_datasets import SATv2Dataset
from utils.fot.maze_ops import MazeTraceDataset
from utils.fot.metrics import mean_t_ci, wilson_accuracy_ci


class PaperSuiteTests(unittest.TestCase):
    def test_smoke_and_overnight_stage_counts(self) -> None:
        common = dict(seeds=[0, 1, 2], run_root=Path("/tmp/paper-suite"), results_dir=Path("/tmp/results"),
                      categories=None, num_workers=0)
        smoke = stage_plan(SimpleNamespace(profile="smoke", **common))
        overnight = stage_plan(SimpleNamespace(profile="overnight", **common))
        self.assertEqual(len(smoke), 25)
        self.assertEqual(len(overnight), 65)
        self.assertFalse(any("partial" in stage.name for stage in overnight))

    def test_pair_head_backward_is_finite(self) -> None:
        torch.manual_seed(0)
        model = PairHead(384)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 384), torch.randn(8, 384)),
                                                 torch.tensor([0, 1] * 4))
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_sat_multi_image_grid_preserves_all_images(self) -> None:
        dataset = SATv2Dataset.__new__(SATv2Dataset)
        dataset.ds = [{"images": [Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)) for value in (32, 96, 160, 224)],
                       "question": "q", "answers": ["a", "b"], "correct_answer": "b", "question_type": "rotation"}]
        dataset.image_size = 32; dataset.image_index = None; dataset.max_images = 9
        row = dataset[0]
        self.assertEqual(tuple(row["image"].shape), (3, 32, 32))
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["meta"]["num_images"], 4)
        means = [row["image"][:, :16, :16].mean(), row["image"][:, :16, 16:].mean(),
                 row["image"][:, 16:, :16].mean(), row["image"][:, 16:, 16:].mean()]
        self.assertTrue(all(float(left) < float(right) for left, right in zip(means, means[1:])))

    def test_maze_dataset_is_stateless(self) -> None:
        dataset = MazeTraceDataset(n_samples=4, maze_cells=5, img_size=32, seed=7)
        first = dataset[2]
        _ = dataset[0]
        second = dataset[2]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_confidence_intervals(self) -> None:
        low, high = wilson_accuracy_ci(80, 100)
        self.assertLess(low, 0.8); self.assertGreater(high, 0.8)
        low, high = mean_t_ci([0.7, 0.8, 0.9])
        self.assertLess(low, 0.8); self.assertGreater(high, 0.8)

    def test_aggregator_writes_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run_root = root / "runs"; results = root / "results"
            for seed, accuracy in enumerate((0.7, 0.8, 0.9)):
                directory = run_root / "x" / f"seed{seed}"; directory.mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps({"experiment_name": "x", "task": "t",
                    "model": "m", "seed": seed, "metrics": {"test": {"accuracy": accuracy}}}), encoding="utf-8")
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "aggregate_benchmark_suite.py"),
                            "--run-root", str(run_root), "--results-dir", str(results)], cwd=REPO_ROOT, check=True)
            self.assertTrue((results / "audit.json").exists())
            self.assertTrue((results / "metrics.csv").exists())
            self.assertTrue((results / "REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
