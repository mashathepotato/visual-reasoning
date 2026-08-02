from __future__ import annotations

import argparse
import hashlib
import random
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.reproducibility import sha256_file, write_json
from utils.llm_baselines import CHIRAL_TETRIS_SHAPES


SPLIT_OFFSETS = {
    "train": 0,
    "validation": 10_000,
    "test_id": 20_000,
    "test_ood_angle": 30_000,
}


def _derived_seed(base_seed: int, split: str, index: int) -> int:
    payload = f"{base_seed}:{split}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**32)


def _rows(
    *,
    task: str,
    split: str,
    count: int,
    base_seed: int,
    angles: Sequence[int],
) -> List[Dict[str, Any]]:
    shape_keys = sorted(CHIRAL_TETRIS_SHAPES)
    rows: List[Dict[str, Any]] = []
    for index in range(int(count)):
        render_seed = _derived_seed(base_seed, split, index)
        rng = random.Random(render_seed)
        label = index % 2
        if task == "tetris_rotation":
            base_id = shape_keys[rng.randrange(len(shape_keys))]
        else:
            base_id = f"scene-{render_seed:010d}"
        rows.append(
            {
                "sample_id": f"{task}:{split}:{index:06d}",
                "base_id": base_id,
                "render_seed": render_seed,
                "angle_deg": int(rng.choice(list(angles))),
                "is_mirrored": label == 0,
                "label": label,
                "label_name": "SAME" if label == 1 else "DIFFERENT",
            }
        )
    # Keep every prefix nearly balanced while decorrelating angles and base IDs.
    pair_rng = random.Random(base_seed + SPLIT_OFFSETS[split])
    pairs = [rows[i : i + 2] for i in range(0, len(rows), 2)]
    pair_rng.shuffle(pairs)
    return [row for pair in pairs for row in pair]


def build_manifest(
    *,
    task: str,
    image_size: int,
    base_seed: int,
    counts: Dict[str, int],
    num_shapes: int,
) -> Dict[str, Any]:
    id_angles = list(range(0, 360, 10))
    ood_angles = list(range(5, 360, 10))
    splits = {
        split: _rows(
            task=task,
            split=split,
            count=count,
            base_seed=base_seed + SPLIT_OFFSETS[split],
            angles=ood_angles if split == "test_ood_angle" else id_angles,
        )
        for split, count in counts.items()
    }
    return {
        "schema_version": 1,
        "task": task,
        "generator": "scripts/generate_rotation_manifests.py",
        "generator_version": 1,
        "base_seed": int(base_seed),
        "image_size": int(image_size),
        "num_shapes": int(num_shapes),
        "label_semantics": {"0": "DIFFERENT", "1": "SAME"},
        "id_angles_deg": id_angles,
        "ood_angles_deg": ood_angles,
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate immutable rotation split manifests.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test-id", type=int, default=1000)
    parser.add_argument("--test-ood-angle", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=20260802)
    parser.add_argument("--num-shapes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "train": args.train,
        "validation": args.validation,
        "test_id": args.test_id,
        "test_ood_angle": args.test_ood_angle,
    }
    if any(value <= 0 or value % 2 for value in counts.values()):
        raise ValueError("All split sizes must be positive even integers")
    for task in ("tetris_rotation", "colored_rotation"):
        manifest = build_manifest(
            task=task,
            image_size=args.image_size,
            base_seed=args.base_seed,
            counts=counts,
            num_shapes=args.num_shapes,
        )
        split_rows = manifest.pop("splits")
        split_descriptors: Dict[str, Any] = {}
        for split, rows in split_rows.items():
            split_path = args.output_dir / f"{task}_v1_{split}.jsonl"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with split_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            split_descriptors[split] = {
                "path": split_path.name,
                "n": len(rows),
                "sha256": sha256_file(split_path),
            }
        manifest["splits"] = split_descriptors
        output = args.output_dir / f"{task}_v1.json"
        write_json(output, manifest)
        print(f"Wrote {output} and JSONL splits ({sum(counts.values())} samples)")


if __name__ == "__main__":
    main()
