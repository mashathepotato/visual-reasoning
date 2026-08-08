#!/usr/bin/env python3
"""Run resumable direct-VLM baselines on every completed benchmark family.

The default protocol evaluates the complete public external sets (Ganis-Kievit,
SAT-v2 test, and BLINK Multi-view validation) plus fixed, balanced subsets of
the larger synthetic rotation and maze tasks. Responses and token usage are
appended to a JSONL cache after every item so interrupted runs resume safely.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import wilson_accuracy_ci  # noqa: E402
from utils.fot.rotation_dataset import RotationPairDataset  # noqa: E402
from utils.llm_baselines import (  # noqa: E402
    MAZE_SOLVE_PROMPT,
    MAZE_TRACE_PROMPT,
    ROTATION_PROMPT,
    build_maze_solve_instances,
    build_maze_trace_samples,
    load_3d_blocks_pairs,
    make_pair_image,
    pair_image_from_gray_arrays,
    render_maze,
    to_u8_from_minus1_1,
    verify_moves,
)

BLINK_DATASET = "BLINK-Benchmark/BLINK"
BLINK_CONFIG = "Multi-view_Reasoning"
BLINK_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"
DEFAULT_TASKS = (
    "tetris_ood",
    "colored_ood",
    "ganis3d",
    "maze_trace",
    "maze_solve",
    "sat_v2",
    "blink_multiview",
)
MAX_DYNAMIC_OUTPUT_TOKENS = 32768


class OutputBudgetExhausted(RuntimeError):
    """The model spent the entire allowed output budget without a final answer."""


@dataclass
class EvalItem:
    task: str
    item_id: str
    images: List[Any]
    prompt: str
    label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    score_response: Optional[Callable[[str], Dict[str, Any]]] = None


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    media_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "sota_vlm_baselines_2026-08-08")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--tetris-samples", type=int, default=100)
    parser.add_argument("--colored-samples", type=int, default=100)
    parser.add_argument("--maze-trace-samples", type=int, default=100)
    parser.add_argument("--maze-solve-samples", type=int, default=50)
    parser.add_argument("--sat-samples", type=int, default=150, help="SAT-v2 test has 150 items; 0 means all.")
    parser.add_argument("--sat-circular", action="store_true", help="Evaluate original and reversed SAT answer order, matching the paper protocol.")
    parser.add_argument("--blink-samples", type=int, default=0, help="0 means all 133 validation items.")
    parser.add_argument("--result-tag", default="", help="Optional suffix for result/report files; the response cache remains shared.")
    parser.add_argument("--summarize-cache", action="store_true", help="Write an explicitly partial result from cached responses without API calls.")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--dry-run", action="store_true", help="Build and fingerprint the protocol without API calls.")
    return parser.parse_args()


def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the shell environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def png_bytes(image: Any) -> bytes:
    if isinstance(image, EncodedImage):
        return image.data
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_media_type(image: Any) -> str:
    return image.media_type if isinstance(image, EncodedImage) else "image/png"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_pair_image(source: Any, target: Any, *, upscale: int = 4, pad: int = 12) -> Image.Image:
    arrays = []
    for tensor in (source, target):
        array = tensor.detach().cpu().permute(1, 2, 0).numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        array = np.repeat(np.repeat(array, upscale, axis=0), upscale, axis=1)
        arrays.append(array)
    return make_pair_image(arrays[0], arrays[1], pad=pad)


def parse_token(text: str, choices: Sequence[str]) -> Optional[str]:
    normalized = (text or "").strip().upper()
    for choice in sorted(choices, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(choice)}(?![A-Z])", normalized):
            return choice
    return None


def parse_letter(text: str, count: int) -> Optional[int]:
    normalized = (text or "").strip().upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count]
    patterns = [r"^\s*\(?([A-Z])\)?(?:\s|[.:-]|$)", r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?"]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match and match.group(1) in allowed:
            return allowed.index(match.group(1))
    return None


def extract_move_sequence(text: str) -> str:
    candidates = re.findall(r"(?<![A-Z])[UDLR]{2,}(?![A-Z])", (text or "").upper())
    return max(candidates, key=len) if candidates else ""


def binary_item(task: str, item_id: str, image: Image.Image, prompt: str, label: str, choices: Sequence[str], **metadata: Any) -> EvalItem:
    def scorer(response: str) -> Dict[str, Any]:
        prediction = parse_token(response, choices)
        return {"prediction": prediction, "correct": prediction == label, "invalid": prediction is None}

    return EvalItem(task=task, item_id=item_id, images=[image], prompt=prompt, label=label, metadata=metadata, score_response=scorer)


def balanced_indices(labels: Sequence[int], count: int) -> List[int]:
    if count <= 0 or count >= len(labels):
        return list(range(len(labels)))
    per_class = count // 2
    zeros = [i for i, value in enumerate(labels) if value == 0][:per_class]
    ones = [i for i, value in enumerate(labels) if value == 1][: count - len(zeros)]
    return sorted(zeros + ones)


def build_rotation_items(task: str, count: int) -> List[EvalItem]:
    stem = "tetris" if task == "tetris_ood" else "colored"
    dataset = RotationPairDataset(REPO_ROOT / "data" / "splits" / f"{stem}_rotation_v1.json", "test_ood_angle")
    labels = [int(row["label"]) for row in dataset.rows]
    items: List[EvalItem] = []
    for index in balanced_indices(labels, count):
        sample = dataset[index]
        label = "SAME" if int(sample["label"]) == 1 else "DIFFERENT"
        items.append(binary_item(
            task, sample["sample_id"], tensor_pair_image(sample["source"], sample["target"]),
            ROTATION_PROMPT, label, ("SAME", "DIFFERENT"),
            angle_deg=float(sample["angle_deg"]), base_id=sample["base_id"], split="test_ood_angle",
        ))
    return items


def build_ganis_items() -> List[EvalItem]:
    items: List[EvalItem] = []
    for index, sample in enumerate(load_3d_blocks_pairs(REPO_ROOT)):
        left = to_u8_from_minus1_1(np.asarray(sample["x0"]))
        right = to_u8_from_minus1_1(np.asarray(sample["x1"]))
        label = "SAME" if str(sample["label"]) == "same" else "DIFFERENT"
        items.append(binary_item(
            "ganis3d", f"ganis3d:{index:03d}:{sample['name']}", pair_image_from_gray_arrays(left, right),
            ROTATION_PROMPT, label, ("SAME", "DIFFERENT"), angle_deg=float(sample["angle"]), name=str(sample["name"]),
        ))
    return items


def build_maze_trace_items(count: int, seed: int) -> List[EvalItem]:
    samples = build_maze_trace_samples(random.Random(seed + 101), count)
    return [binary_item("maze_trace", f"maze_trace:{seed + 101}:{i:04d}", image, MAZE_TRACE_PROMPT, label, ("YES", "NO"))
            for i, (image, label) in enumerate(samples)]


def build_maze_solve_items(count: int, seed: int) -> List[EvalItem]:
    instances = build_maze_solve_instances(random.Random(seed + 102), count)
    items: List[EvalItem] = []
    for index, instance in enumerate(instances):
        def scorer(response: str, instance: Any = instance) -> Dict[str, Any]:
            moves = extract_move_sequence(response)
            success = verify_moves(instance.grid, instance.start, instance.goal, moves)
            return {"prediction": moves, "correct": success, "success": success, "invalid": not bool(moves), "path_length": len(moves)}

        items.append(EvalItem(
            task="maze_solve", item_id=f"maze_solve:{seed + 102}:{index:04d}",
            images=[render_maze(instance.grid, instance.start, instance.goal)], prompt=MAZE_SOLVE_PROMPT,
            label="valid_path", metadata={"maze_cells": 9}, score_response=scorer,
        ))
    return items


def decode_hf_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB").copy()
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB").copy()
    raise TypeError(f"Unsupported Hugging Face image value: {type(value)!r}")


def encode_hf_image(value: Any) -> EncodedImage:
    if isinstance(value, dict) and value.get("bytes") is not None:
        data = bytes(value["bytes"])
        with Image.open(io.BytesIO(data)) as image:
            image_format = str(image.format or "PNG").lower()
        media_type = "image/jpeg" if image_format in {"jpeg", "jpg"} else f"image/{image_format}"
        return EncodedImage(data=data, media_type=media_type)
    image = decode_hf_image(value)
    return EncodedImage(data=png_bytes(image), media_type="image/png")


def sat_test_parquet() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("array/SAT-v2", "data/test-00000-of-00001.parquet", repo_type="dataset"))


def build_sat_items(count: int, *, circular: bool = False) -> List[EvalItem]:
    import pyarrow.parquet as pq

    items: List[EvalItem] = []
    maximum = 150 if count <= 0 else min(150, count)
    parquet = pq.ParquetFile(sat_test_parquet())
    for index, batch in enumerate(parquet.iter_batches(batch_size=1)):
        if index >= maximum:
            break
        row = batch.to_pylist()[0]
        images = [encode_hf_image(value) for value in row["images"]]
        original_choices = [str(value) for value in row["answers"]]
        correct_text = str(row["correct_answer"]).strip()
        orderings = [("original", original_choices)]
        if circular:
            orderings.append(("reversed", list(reversed(original_choices))))
        for ordering_name, choices in orderings:
            label = next((i for i, choice in enumerate(choices) if choice.strip() == correct_text), None)
            if label is None:
                label = next(i for i, choice in enumerate(choices) if choice.strip().lower() == correct_text.lower())
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            options = "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(choices))
            prompt = (
                "Solve this spatial-reasoning multiple-choice problem. The images are provided in chronological order.\n"
                f"Question: {row['question']}\nOptions:\n{options}\nReply with only the option letter."
            )

            def scorer(response: str, label: int = label, n: int = len(choices)) -> Dict[str, Any]:
                prediction = parse_letter(response, n)
                return {"prediction": prediction, "correct": prediction == label, "invalid": prediction is None}

            suffix = "" if ordering_name == "original" else ":reversed"
            items.append(EvalItem(
                task="sat_v2", item_id=f"satv2-{index:07d}{suffix}", images=images, prompt=prompt, label=label,
                metadata={
                    "question_type": str(row["question_type"]), "num_images": len(images), "choices": choices,
                    "answer_order": ordering_name, "base_item_id": f"satv2-{index:07d}",
                },
                score_response=scorer,
            ))
    return items


def load_blink_validation() -> Iterable[Dict[str, Any]]:
    cached = Path.home() / ".cache" / "huggingface" / "datasets" / "BLINK-Benchmark___blink" / BLINK_CONFIG / "0.0.0" / BLINK_REVISION / "blink-val.arrow"
    if cached.exists():
        from datasets import Dataset
        return Dataset.from_file(str(cached))
    from datasets import load_dataset
    return load_dataset(BLINK_DATASET, BLINK_CONFIG, split="val", revision=BLINK_REVISION)


def build_blink_items(count: int) -> List[EvalItem]:
    rows = load_blink_validation()
    items: List[EvalItem] = []
    for index, row in enumerate(rows):
        if count > 0 and index >= count:
            break
        label = 0 if str(row["answer"]).strip().upper() in {"(A)", "A"} else 1
        prompt = str(row["prompt"]).strip() + "\nReply with only A or B."

        def scorer(response: str, label: int = label) -> Dict[str, Any]:
            prediction = parse_letter(response, 2)
            return {"prediction": prediction, "correct": prediction == label, "invalid": prediction is None}

        items.append(EvalItem(
            task="blink_multiview", item_id=str(row["idx"]),
            images=[encode_hf_image(row["image_1"]), encode_hf_image(row["image_2"])],
            prompt=prompt, label=label, metadata={"split": "val", "dataset_revision": BLINK_REVISION}, score_response=scorer,
        ))
    return items


def build_items(args: argparse.Namespace) -> List[EvalItem]:
    builders: Dict[str, Callable[[], List[EvalItem]]] = {
        "tetris_ood": lambda: build_rotation_items("tetris_ood", args.tetris_samples),
        "colored_ood": lambda: build_rotation_items("colored_ood", args.colored_samples),
        "ganis3d": build_ganis_items,
        "maze_trace": lambda: build_maze_trace_items(args.maze_trace_samples, args.seed),
        "maze_solve": lambda: build_maze_solve_items(args.maze_solve_samples, args.seed),
        "sat_v2": lambda: build_sat_items(args.sat_samples, circular=args.sat_circular),
        "blink_multiview": lambda: build_blink_items(args.blink_samples),
    }
    items: List[EvalItem] = []
    for task in args.tasks:
        task_items = builders[task]()
        print(f"built {task}: n={len(task_items)}", flush=True)
        items.extend(task_items)
    return items


def item_key(item: EvalItem, *, model: str, effort: str, max_output_tokens: int) -> str:
    digest = hashlib.sha256()
    for value in (model, effort, str(max_output_tokens), item.task, item.item_id, item.prompt):
        digest.update(value.encode("utf-8") + b"\0")
    for image in item.images:
        digest.update(png_bytes(image))
    return digest.hexdigest()


class JsonlResponseCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.entries: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry.get("key"), str):
                    self.entries[entry["key"]] = entry

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(key)

    def put(self, entry: Dict[str, Any]) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
                handle.flush()
            self.entries[entry["key"]] = entry


def response_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    return {"input_tokens": input_tokens, "cached_input_tokens": cached_tokens, "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens}


def call_openai(client: Any, item: EvalItem, *, model: str, effort: str, max_output_tokens: int) -> tuple[str, Dict[str, int], str]:
    content: List[Dict[str, Any]] = []
    for index, image in enumerate(item.images, start=1):
        content.append({"type": "input_text", "text": f"Image {index}:"})
        encoded = base64.b64encode(png_bytes(image)).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:{image_media_type(image)};base64,{encoded}", "detail": "original"})
    content.append({"type": "input_text", "text": item.prompt})
    last_error: Optional[Exception] = None
    token_budget = max_output_tokens
    for attempt in range(8):
        try:
            response = client.responses.create(
                model=model, reasoning={"effort": effort}, input=[{"role": "user", "content": content}],
                max_output_tokens=token_budget, store=False,
            )
            text = str(getattr(response, "output_text", "") or "").strip()
            if not text:
                details = getattr(response, "incomplete_details", None)
                reason = getattr(details, "reason", None)
                if token_budget < MAX_DYNAMIC_OUTPUT_TOKENS:
                    token_budget = min(MAX_DYNAMIC_OUTPUT_TOKENS, token_budget * 2)
                    last_error = RuntimeError(f"empty response ({reason or getattr(response, 'status', None)!r}); increasing token budget")
                    continue
                raise OutputBudgetExhausted(
                    f"empty response at {token_budget} tokens (status={getattr(response, 'status', None)!r}, reason={reason!r})"
                )
            usage = response_usage(response)
            usage["max_output_tokens_used"] = token_budget
            return text, usage, str(getattr(response, "id", ""))
        except OutputBudgetExhausted as error:
            last_error = error
            break
        except Exception as error:  # API retry path
            last_error = error
            if attempt == 7:
                break
            time.sleep(min(20.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def protocol_fingerprint(items: Sequence[EvalItem]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.task.encode() + b"\0" + item.item_id.encode() + b"\0")
        digest.update(json.dumps(item.label, sort_keys=True).encode() + b"\0")
        for image in item.images:
            digest.update(hashlib.sha256(png_bytes(image)).digest())
    return digest.hexdigest()


def score_entry(item: EvalItem, entry: Dict[str, Any]) -> Dict[str, Any]:
    scored = item.score_response(str(entry.get("response", ""))) if item.score_response else {}
    return {
        "item_id": item.item_id,
        "label": item.label,
        "response": entry.get("response", ""),
        "usage": entry.get("usage", {}),
        "request_id": entry.get("request_id", ""),
        **item.metadata,
        **scored,
    }


def summarize_task(task: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(bool(row.get("correct")) for row in rows)
    n = len(rows)
    low, high = wilson_accuracy_ci(correct, n)
    summary: Dict[str, Any] = {
        "task": task, "n": n, "metric": "success_rate" if task == "maze_solve" else "accuracy",
        "value": correct / n if n else 0.0, "ci95_low": low, "ci95_high": high,
        "ci_method": "wilson_over_items", "invalid_responses": sum(bool(row.get("invalid")) for row in rows),
    }
    if task == "sat_v2":
        circular_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            circular_groups.setdefault(str(row.get("base_item_id", row.get("item_id", "unknown"))), []).append(row)
        if len(circular_groups) < n:
            item_scores = np.asarray(
                [np.mean([bool(row.get("correct")) for row in group]) for group in circular_groups.values()], dtype=np.float64
            )
            rng = np.random.default_rng(20260808)
            boot_indices = rng.integers(0, len(item_scores), size=(10_000, len(item_scores)))
            boot_means = item_scores[boot_indices].mean(axis=1)
            summary.update({
                "n": len(item_scores), "presentations": n,
                "ci95_low": float(np.quantile(boot_means, 0.025)),
                "ci95_high": float(np.quantile(boot_means, 0.975)),
                "ci_method": "item_cluster_bootstrap_10000",
            })
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get("question_type", "unknown")), []).append(row)
        summary["by_question_type"] = {}
        for name, group in sorted(groups.items()):
            hits = sum(bool(row.get("correct")) for row in group)
            group_low, group_high = wilson_accuracy_ci(hits, len(group))
            summary["by_question_type"][name] = {"n": len(group), "accuracy": hits / len(group), "ci95_low": group_low, "ci95_high": group_high}
    return summary


def write_outputs(
    args: argparse.Namespace, items: Sequence[EvalItem], entries: Dict[str, Dict[str, Any]], *, allow_partial: bool = False
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_task: Dict[str, List[Dict[str, Any]]] = {}
    expected_by_task: Dict[str, int] = {}
    for item in items:
        expected_by_task[item.task] = expected_by_task.get(item.task, 0) + 1
        key = item_key(item, model=args.model, effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens)
        if key not in entries:
            if allow_partial:
                continue
            raise KeyError(f"Missing response for {item.task}/{item.item_id}")
        per_task.setdefault(item.task, []).append(score_entry(item, entries[key]))
    summaries = [summarize_task(task, rows) for task, rows in per_task.items()]
    for summary in summaries:
        summary["expected_n"] = expected_by_task[summary["task"]]
        summary["complete"] = int(summary.get("presentations", summary["n"])) == expected_by_task[summary["task"]]
    usage = {name: sum(int(row.get("usage", {}).get(name, 0)) for rows in per_task.values() for row in rows)
             for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")}
    estimated_cost = (usage["input_tokens"] * 5.0 + usage["output_tokens"] * 30.0) / 1_000_000 if args.model == "gpt-5.6-sol" else None
    result = {
        "schema_version": 1,
        "model": args.model,
        "provider": "openai",
        "reasoning_effort": args.reasoning_effort,
        "protocol_fingerprint_sha256": protocol_fingerprint(items),
        "complete": sum(len(rows) for rows in per_task.values()) == len(items),
        "cached_responses": sum(len(rows) for rows in per_task.values()),
        "expected_responses": len(items),
        "coverage": {
            task: {"cached": len(per_task.get(task, [])), "expected": expected}
            for task, expected in expected_by_task.items()
        },
        "summaries": summaries,
        "usage": usage,
        "estimated_cost_usd_at_2026_08_08_list_price": estimated_cost,
        "predictions": per_task,
    }
    tag = safe_result_tag(args.result_tag)
    stem = args.model.replace('/', '_') + (f"_{tag}" if tag else "")
    (args.output_dir / f"{stem}_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# Direct VLM baseline: {args.model}", "",
        f"Reasoning effort: `{args.reasoning_effort}`. Protocol fingerprint: `{result['protocol_fingerprint_sha256']}`.",
        f"Complete: `{result['complete']}` ({result['cached_responses']}/{result['expected_responses']} responses).", "",
        "| Task | Metric | cached n | expected n | Value | 95% CI | Invalid | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['n']} | {row['expected_n']} | {row['value']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['invalid_responses']} | {row['complete']} |"
        )
    lines.extend(["", f"Token usage: {json.dumps(usage, sort_keys=True)}.", f"Estimated list-price cost: ${estimated_cost:.4f}." if estimated_cost is not None else ""])
    (args.output_dir / f"{stem}_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_result_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_output_tokens < 64:
        raise SystemExit("--workers must be positive and --max-output-tokens must be at least 64")
    load_local_env(REPO_ROOT / ".env")
    items = build_items(args)
    fingerprint = protocol_fingerprint(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1, "tasks": args.tasks, "sample_counts": {task: sum(item.task == task for item in items) for task in args.tasks},
        "seed": args.seed, "model": args.model, "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens, "protocol_fingerprint_sha256": fingerprint,
        "sat_circular": args.sat_circular,
        "external_splits": {"ganis3d": "complete local n=78", "sat_v2": "complete test n=150", "blink_multiview": f"validation revision {BLINK_REVISION}"},
    }
    protocol_tag = safe_result_tag(args.result_tag)
    protocol_name = f"resolved_protocol_{protocol_tag}.json" if protocol_tag else "resolved_protocol.json"
    (args.output_dir / protocol_name).write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    cache = JsonlResponseCache(args.output_dir / f"{args.model.replace('/', '_')}_responses.jsonl")
    entries: Dict[str, Dict[str, Any]] = {}
    pending: List[tuple[str, EvalItem]] = []
    for item in items:
        key = item_key(item, model=args.model, effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens)
        cached = cache.get(key)
        if cached is not None:
            entries[key] = cached
        else:
            pending.append((key, item))
    print(f"cached={len(entries)} pending={len(pending)} workers={args.workers}", flush=True)
    if args.summarize_cache:
        write_outputs(args, items, entries, allow_partial=True)
        print(f"wrote partial cached results under {args.output_dir}", flush=True)
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (the script also checks repository .env).")
    from openai import OpenAI

    client = OpenAI()

    def evaluate_one(key_item: tuple[str, EvalItem]) -> tuple[str, Dict[str, Any]]:
        key, item = key_item
        response, usage, request_id = call_openai(client, item, model=args.model, effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens)
        entry = {
            "key": key, "task": item.task, "item_id": item.item_id, "model": args.model,
            "reasoning_effort": args.reasoning_effort, "response": response, "usage": usage,
            "request_id": request_id, "timestamp": time.time(),
            "image_sha256": [sha256_bytes(png_bytes(image)) for image in item.images],
        }
        cache.put(entry)
        return key, entry

    failures: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(evaluate_one, key_item): key_item for key_item in pending}
        for completed, future in enumerate(as_completed(future_map), start=1):
            key, item = future_map[future]
            try:
                result_key, entry = future.result()
                entries[result_key] = entry
            except Exception as error:
                failures.append({"task": item.task, "item_id": item.item_id, "error": str(error)})
                print(f"failed task={item.task} item={item.item_id}: {error}", flush=True)
            if completed % 10 == 0 or completed == len(pending):
                print(f"completed={completed}/{len(pending)} failures={len(failures)}", flush=True)
    if failures:
        (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(f"{len(failures)} requests failed; rerun to resume. See failures.json")
    write_outputs(args, items, entries)
    print(f"wrote results under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
