from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def compatibility_key(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    experiment = normalized.get("experiment", {})
    if isinstance(experiment, dict):
        experiment.pop("seed", None)
        experiment.pop("resolved_device", None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def aggregate_run_directories(run_directories: Iterable[Path]) -> Dict[str, Any]:
    runs: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    missing: List[Dict[str, str]] = []
    for directory in run_directories:
        run_dir = directory.resolve()
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "resolved_config.json"
        absent = [str(path.name) for path in (summary_path, config_path) if not path.exists()]
        if absent:
            missing.append({"run_directory": str(run_dir), "missing": ",".join(absent)})
            continue
        runs.append((run_dir, _read_json(summary_path), _read_json(config_path)))
    if not runs:
        raise ValueError("No complete runs were provided")

    expected_key = compatibility_key(runs[0][2])
    incompatible = [str(directory) for directory, _, config in runs[1:] if compatibility_key(config) != expected_key]
    if incompatible:
        raise ValueError(f"Incompatible run configurations: {incompatible}")

    seeds = [int(summary["seed"]) for _, summary, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds in run set: {seeds}")

    split_names = set(runs[0][1]["metrics"])
    for _, summary, _ in runs[1:]:
        if set(summary["metrics"]) != split_names:
            raise ValueError("Runs contain different evaluation splits")

    aggregated: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for split in sorted(split_names):
        metric_names = set(runs[0][1]["metrics"][split])
        aggregated[split] = {}
        for metric in sorted(metric_names):
            values = [summary["metrics"][split][metric] for _, summary, _ in runs]
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            numeric = [float(value) for value in values]
            aggregated[split][metric] = {
                "mean": statistics.fmean(numeric),
                "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
                "n_seeds": len(numeric),
            }

    return {
        "experiment_name": runs[0][1]["experiment_name"],
        "task": runs[0][1]["task"],
        "model": runs[0][1]["model"],
        "seeds": sorted(seeds),
        "completed_runs": [str(directory) for directory, _, _ in runs],
        "missing_or_failed_runs": missing,
        "metrics": aggregated,
    }
