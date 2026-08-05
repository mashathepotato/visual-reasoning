#!/usr/bin/env python3
"""Create an auditable, non-overwriting snapshot of a completed benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_below(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_manifest(root: Path, destination: Path) -> str:
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files_below(root)]
    text = "\n".join(lines) + ("\n" if lines else "")
    destination.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clone_tree(source: Path, destination: Path) -> str:
    """Prefer APFS copy-on-write clones; fall back to ordinary copies."""
    try:
        subprocess.run(["/bin/cp", "-cR", str(source), str(destination)], check=True, capture_output=True)
        return "apfs_clone"
    except (FileNotFoundError, subprocess.CalledProcessError):
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        return "copy"


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mps_paper_suite_2026-08-03")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "models/runs/mps_paper_suite")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results/mps_paper_suite")
    parser.add_argument("--archive-root", type=Path, default=REPO_ROOT / "models/archives")
    parser.add_argument("--snapshot-root", type=Path, default=REPO_ROOT / "results/baseline_archives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    results_dir = args.results_dir.resolve()
    raw_archive = args.archive_root.resolve() / args.name
    snapshot = args.snapshot_root.resolve() / args.name
    for source in (run_root, results_dir):
        if not source.is_dir():
            raise SystemExit(f"Missing source directory: {source}")
    for destination in (raw_archive, snapshot):
        if destination.exists():
            raise SystemExit(f"Refusing to overwrite existing snapshot: {destination}")

    raw_archive.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    method = clone_tree(run_root, raw_archive)
    shutil.copytree(results_dir, snapshot)

    raw_manifest_hash = write_manifest(raw_archive, raw_archive / "MANIFEST.sha256")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "source": {"run_root": str(run_root), "results_dir": str(results_dir)},
        "archive": {
            "raw_path": str(raw_archive),
            "results_path": str(snapshot),
            "copy_method": method,
            "read_only": True,
            "raw_file_count": sum(1 for _ in files_below(run_root)),
            "results_file_count": sum(1 for _ in files_below(results_dir)) + 2,
            "raw_manifest_sha256": raw_manifest_hash,
            "results_manifest": str(snapshot / "MANIFEST.sha256"),
        },
        "git_commit_at_archive": git_commit,
        "notes": [
            "This preserves the completed 65-stage MPS paper suite before flow-v1 development.",
            "Raw checkpoints are stored outside Git under models/archives; compact reports are tracked under results.",
            "The archive utility refuses to overwrite a snapshot with the same name.",
        ],
    }
    (snapshot / "SNAPSHOT.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (snapshot / "README.md").write_text(
        "# Frozen MPS paper-suite baseline\n\n"
        "This directory is the compact, auditable snapshot of the completed 65-stage overnight suite. "
        "The full checkpoint tree is preserved at the `raw_path` recorded in `SNAPSHOT.json`. "
        "Verify either tree with `shasum -a 256 -c MANIFEST.sha256`. New flow experiments must use "
        "`models/runs/neurreps_flow_v1` and must not write into this snapshot.\n",
        encoding="utf-8",
    )
    # A manifest cannot checksum itself, so it contains every other snapshot file.
    lines = [
        f"{sha256(path)}  {path.relative_to(snapshot).as_posix()}"
        for path in files_below(snapshot)
        if path.name != "MANIFEST.sha256"
    ]
    (snapshot / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_read_only(raw_archive)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
