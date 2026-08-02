from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "experiment",
    "dataset",
    "model",
    "training",
    "evaluation",
    "controller",
    "transition",
}


def _require(mapping: Mapping[str, Any], key: str, expected_type: type, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field {context}.{key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise ValueError(f"{context}.{key} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL.difference(config)
    if missing:
        raise ValueError(f"Missing top-level config fields: {sorted(missing)}")
    unknown = set(config).difference(REQUIRED_TOP_LEVEL)
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {sorted(unknown)}")
    if config["schema_version"] != 1:
        raise ValueError(f"Unsupported schema_version: {config['schema_version']!r}")

    experiment = _require(config, "experiment", dict, "config")
    _require(experiment, "name", str, "experiment")
    _require(experiment, "seed", int, "experiment")
    device = _require(experiment, "device", str, "experiment")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"experiment.device must be auto/cpu/cuda/mps, got {device!r}")

    dataset = _require(config, "dataset", dict, "config")
    if _require(dataset, "task", str, "dataset") not in {"tetris_rotation", "colored_rotation"}:
        raise ValueError("dataset.task must be tetris_rotation or colored_rotation")
    _require(dataset, "manifest", str, "dataset")
    image_size = _require(dataset, "image_size", int, "dataset")
    if image_size <= 0:
        raise ValueError("dataset.image_size must be positive")
    fraction = dataset.get("train_fraction", 1.0)
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise ValueError("dataset.train_fraction must be in (0, 1]")
    max_eval = dataset.get("max_eval_samples")
    if max_eval is not None and (not isinstance(max_eval, int) or max_eval <= 0):
        raise ValueError("dataset.max_eval_samples must be null or a positive integer")

    model = _require(config, "model", dict, "config")
    if _require(model, "type", str, "model") not in {"cnn", "vit"}:
        raise ValueError("model.type must be cnn or vit")
    _require(model, "input_channels", int, "model")
    _require(model, "num_classes", int, "model")

    training = _require(config, "training", dict, "config")
    for field in ("epochs", "batch_size", "num_workers"):
        value = _require(training, field, int, "training")
        if value < 0 or (field != "num_workers" and value == 0):
            raise ValueError(f"training.{field} has invalid value {value}")
    for field in ("learning_rate", "weight_decay"):
        value = training.get(field)
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"training.{field} must be non-negative")
    if training.get("optimizer") != "adamw":
        raise ValueError("Only training.optimizer='adamw' is currently supported")

    evaluation = _require(config, "evaluation", dict, "config")
    if evaluation.get("checkpoint_selection") != "validation_accuracy":
        raise ValueError("evaluation.checkpoint_selection must be validation_accuracy")
    _require(evaluation, "splits", list, "evaluation")

    _require(config, "controller", dict, "config")
    _require(config, "transition", dict, "config")


def load_experiment_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config root must be a JSON object")
    validate_experiment_config(config)
    return config
