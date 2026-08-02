from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.aggregation import aggregate_run_directories
from utils.fot.reproducibility import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compatible experiment runs across seeds.")
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate = aggregate_run_directories(args.run_directories)
    write_json(args.output, aggregate)
    missing = aggregate["missing_or_failed_runs"]
    print(f"Wrote {args.output} from {len(aggregate['completed_runs'])} complete run(s)")
    if missing:
        print(f"Marked {len(missing)} missing or failed run(s); see output JSON")


if __name__ == "__main__":
    main()
