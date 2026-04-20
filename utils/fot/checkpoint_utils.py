from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch


def _coerce_state_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "rotator_state_dict", "rotator", "rotator_sd"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not any(str(k).startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {str(k)[len(prefix) :]: v for k, v in state_dict.items()}


def load_state_dict(path: Union[str, Path], device: torch.device) -> Dict[str, Any]:
    p = Path(path)
    obj = torch.load(str(p), map_location=device)
    sd = _coerce_state_dict(obj)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict-like dict in {p}, got {type(sd)}")
    return _strip_prefix(sd, "module.")

