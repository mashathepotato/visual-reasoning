from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MPSOvernightTests(unittest.TestCase):
    def test_dry_run_plans_all_models_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_mps_overnight.py"),
                    "--dry-run",
                    "--output-root",
                    str(root / "runs"),
                    "--results-dir",
                    str(root / "results"),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            state = json.loads((root / "results" / "overnight_status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "dry_run")
            self.assertEqual(len(state["runs"]), 6)
            self.assertEqual({row["model"] for row in state["runs"]}, {"cnn", "vit"})
            self.assertTrue(all(row["status"] == "dry_run" for row in state["runs"]))
            self.assertEqual(result.stdout.count("--device mps"), 6)


if __name__ == "__main__":
    unittest.main()
