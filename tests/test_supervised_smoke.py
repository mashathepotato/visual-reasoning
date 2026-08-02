from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SupervisedSmokeTests(unittest.TestCase):
    def test_tiny_training_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "fot" / "train_supervised_baseline.py"),
                    "--config",
                    str(REPO_ROOT / "configs" / "baselines" / "rotation_vit_smoke.json"),
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["preliminary"])
            self.assertEqual(summary["train_samples"], 10)
            self.assertEqual(summary["metrics"]["test_id"]["n"], 8)
            self.assertEqual(summary["checkpoint_selection"], "validation_accuracy")
            self.assertIn("commit", metadata["git"])
            self.assertIn("torch", metadata["dependencies"])


if __name__ == "__main__":
    unittest.main()
