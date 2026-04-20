from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .image_ops import bbox_xywh_to_mask, resize_image_tensor, to_rgb_tensor01


def _lazy_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: datasets. Install with `pip install datasets`.") from e
    return load_dataset


class SATv2Dataset(Dataset):
    """SAT-v2 multiple-choice QA dataset wrapper (HuggingFace: array/SAT-v2)."""

    def __init__(
        self,
        *,
        split: str,
        image_size: int = 256,
        image_index: int = 0,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        load_dataset = _lazy_load_dataset()
        ds = load_dataset("array/SAT-v2", split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(int(max_samples)))
        self.ds = ds
        self.image_size = int(image_size)
        self.image_index = int(image_index)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        images = row.get("images")
        if isinstance(images, list):
            img = images[self.image_index]
        else:
            img = images

        img_t = resize_image_tensor(to_rgb_tensor01(img), size=self.image_size)
        question = str(row.get("question", ""))
        choices = list(row.get("answers", []))
        correct = str(row.get("correct_answer", ""))

        label = -1
        for i, c in enumerate(choices):
            if str(c).strip() == correct.strip():
                label = i
                break
        if label < 0:
            # fallback: case-insensitive match
            correct_low = correct.strip().lower()
            for i, c in enumerate(choices):
                if str(c).strip().lower() == correct_low:
                    label = i
                    break
        if label < 0:
            raise ValueError(f"Could not find correct_answer={correct!r} in answers={choices!r}")

        return {
            "image": img_t,
            "question": question,
            "choices": [str(c) for c in choices],
            "label": int(label),
            "meta": {"question_type": row.get("question_type", None)},
        }


_VSTAR_OPT_RE = re.compile(r"\(([A-D])\)\s*(.*?)\s*(?=\([A-D]\)|$)", re.DOTALL)


def parse_vstar_text(text: str) -> Tuple[str, List[str]]:
    """Parse V*Bench-style prompt text into (question, [A,B,C,D] choices)."""
    t = (text or "").strip()
    # Strip leading "Question:" and trailing "Answer:" blocks if present.
    t = re.sub(r"^\s*Question:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*Answer:\s*$", "", t, flags=re.IGNORECASE)

    parts = _VSTAR_OPT_RE.findall(t)
    if not parts:
        # Fallback: maybe options are on separate lines without (A) tokens.
        raise ValueError(f"Could not parse V*Bench options from text: {text[:200]!r}")

    # Question is the text before the first option marker.
    first = t.find("(A)")
    question = t[:first].strip() if first >= 0 else ""

    opts = {k: v.strip() for (k, v) in parts}
    choices = [opts.get("A", ""), opts.get("B", ""), opts.get("C", ""), opts.get("D", "")]
    return question, choices


class VStarBenchWithBboxDataset(Dataset):
    """V*Bench (V-STaR) dataset with bounding boxes (HF: jae-minkim/vstar-bench-with-bbox)."""

    def __init__(
        self,
        *,
        split: str = "test",
        image_size: int = 512,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        load_dataset = _lazy_load_dataset()
        ds = load_dataset("jae-minkim/vstar-bench-with-bbox", split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(int(max_samples)))
        self.ds = ds
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        img = row.get("image")
        img_t = resize_image_tensor(to_rgb_tensor01(img), size=self.image_size)

        text = str(row.get("text", ""))
        question, choices = parse_vstar_text(text)

        label = str(row.get("label", "")).strip().upper()
        if label not in ("A", "B", "C", "D"):
            raise ValueError(f"Unexpected label={label!r}")
        label_idx = {"A": 0, "B": 1, "C": 2, "D": 3}[label]

        # Build a supervision mask by union of bboxes.
        bbox_targets = row.get("bbox_target") or []
        orig_size = row.get("original_image_size") or None  # [W,H]
        mask = None
        if orig_size is not None and isinstance(bbox_targets, list) and bbox_targets:
            masks = []
            for bb in bbox_targets:
                masks.append(
                    bbox_xywh_to_mask(
                        bbox_xywh=bb,
                        original_size_wh=orig_size,
                        out_size=self.image_size,
                        device=torch.device("cpu"),
                    )
                )
            mask = torch.clamp(torch.stack(masks, dim=0).sum(dim=0), 0.0, 1.0)  # (1,H,W)

        return {
            "image": img_t,
            "question": question,
            "choices": choices,
            "label": int(label_idx),
            "target_mask": mask,  # (1,H,W) on CPU or None
            "meta": {
                "category": row.get("category", None),
                "question_id": row.get("question_id", None),
                "original_image_size": orig_size,
            },
        }

