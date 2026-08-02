from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "kornia",
    "gymnasium",
    "stable-baselines3",
    "scikit-learn",
    "timm",
    "datasets",
    "pillow",
)


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def dependency_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_run_metadata(
    *,
    repo_root: Path,
    config_path: Optional[Path] = None,
    command: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Collect portable provenance for a training or evaluation run."""
    status = _git(repo_root, "status", "--porcelain")
    metadata: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command if command is not None else sys.argv),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines() if status else [],
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        },
    }
    if config_path is not None:
        resolved = config_path.resolve()
        metadata["config"] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    return metadata


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
