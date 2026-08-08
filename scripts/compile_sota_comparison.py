#!/usr/bin/env python3
"""Compile archived, direct-VLM, and published spatial baselines into one audit."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import wilson_accuracy_ci  # noqa: E402
from utils.llm_baselines import (  # noqa: E402
    JsonlCache,
    build_3d_blocks_samples,
    build_colored_shapes_samples,
    build_maze_solve_instances,
    build_maze_trace_samples,
    build_tetris_samples,
    cache_key,
    eval_maze_solve,
    eval_maze_trace,
    eval_rotation,
    pil_to_png_bytes,
)

OUTPUT_DIR = REPO_ROOT / "results" / "sota_vlm_baselines_2026-08-08"
ARCHIVE = REPO_ROOT / "results" / "baseline_archives" / "mps_paper_suite_2026-08-03" / "audit.json"
SAT_PAPER_URL = "https://arxiv.org/abs/2412.07755"
SPATIAL_DREAMER_URL = "https://arxiv.org/abs/2512.07733"
BLINK_PAPER_URL = "https://arxiv.org/abs/2404.12390"
P2_URL = "https://openaccess.thecvf.com/content/CVPR2026/html/Janjua_Dont_Show_Pixels_Show_Cues_Unlocking_Visual_Tool_Reasoning_in_CVPR_2026_paper.html"
DR_MV3D_URL = "https://dr-mv3d.github.io/"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_key(*, model: str, prompt: str, image_bytes: bytes, max_output_tokens: int, old_openai: bool) -> str:
    if not old_openai:
        return cache_key(model=model, prompt=prompt, image_bytes=image_bytes, max_output_tokens=max_output_tokens, temperature=0.0)
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8") + b"\n" + prompt.encode("utf-8") + b"\n" + image_bytes)
    return digest.hexdigest()


def recover_historical_model(*, model: str, cache_path: Path, old_openai: bool) -> Dict[str, Any]:
    cache = JsonlCache(cache_path)
    missing: List[str] = []

    def cached_vision(prompt: str, image: Any, *, max_output_tokens: int) -> str:
        key = historical_key(
            model=model, prompt=prompt, image_bytes=pil_to_png_bytes(image.convert("RGB")),
            max_output_tokens=max_output_tokens, old_openai=old_openai,
        )
        response = cache.get_response(key)
        if response is None:
            missing.append(key)
            return ""
        return response

    rng = random.Random(0)
    tetris = build_tetris_samples(rng, 500)
    colored = build_colored_shapes_samples(rng, 500)
    ganis = build_3d_blocks_samples(REPO_ROOT, rng, 500)
    maze_trace = build_maze_trace_samples(rng, 500)
    maze_solve = build_maze_solve_instances(rng, 500)
    results = {
        "tetris": eval_rotation("tetris", tetris, llm_vision=cached_vision),
        "colored": eval_rotation("colored", colored, llm_vision=cached_vision),
        "ganis3d": eval_rotation("ganis3d", ganis, llm_vision=cached_vision),
        "maze_trace": eval_maze_trace(maze_trace, llm_vision=cached_vision),
        "maze_solve": eval_maze_solve(maze_solve, llm_vision=cached_vision),
    }
    if missing:
        raise RuntimeError(f"Historical cache {cache_path} is missing {len(missing)} requests")
    summaries = []
    for task, result in results.items():
        metric = "success_rate" if task == "maze_solve" else "accuracy"
        value = float(result[metric])
        n = int(result["n"])
        correct = round(value * n)
        low, high = wilson_accuracy_ci(correct, n)
        summaries.append({
            "task": task, "metric": metric, "n": n, "value": value, "ci95_low": low, "ci95_high": high,
            "ci_method": "wilson_over_items",
        })
    return {
        "model": model, "provider": "openai" if old_openai else "anthropic", "seed": 0,
        "protocol": "procedural generator sequence from the original direct-VLM notebooks",
        "cache_path": str(cache_path.relative_to(REPO_ROOT)), "cache_sha256": sha256_file(cache_path),
        "summaries": summaries,
    }


def find_aggregate(audit: Dict[str, Any], experiment: str, split: str, metric: str) -> Dict[str, Any]:
    for row in audit["aggregate_rows"]:
        if row.get("experiment") == experiment and row.get("split") == split and row.get("metric") == metric:
            return row
    raise KeyError((experiment, split, metric))


def add_record(records: List[Dict[str, Any]], *, dataset: str, protocol: str, method: str, category: str,
               metric: str, value: Any, n: Any = "", ci_low: Any = "", ci_high: Any = "",
               source: str, comparability: str = "matched", ci_method: str = "") -> None:
    records.append({
        "dataset": dataset, "protocol": protocol, "method": method, "category": category,
        "metric": metric, "value": float(value) if value != "" else "", "n": n, "ci95_low": ci_low, "ci95_high": ci_high,
        "ci_method": ci_method, "source": source, "comparability": comparability,
    })


def archive_record(records: List[Dict[str, Any]], audit: Dict[str, Any], *, dataset: str, protocol: str,
                   experiment: str, split: str, metric: str, method: str, category: str, n: Any = "") -> None:
    row = find_aggregate(audit, experiment, split, metric)
    display_n = f"{n}/seed × {row['n_seeds']} seeds" if n != "" and row.get("n_seeds") else n
    add_record(records, dataset=dataset, protocol=protocol, method=method, category=category, metric=metric,
               value=row["mean"], n=display_n, ci_low=row.get("ci95_low", ""), ci_high=row.get("ci95_high", ""),
               ci_method=str(row.get("ci_method", "")), source="baseline_archives/mps_paper_suite_2026-08-03/audit.json")


def summary_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["task"]): row for row in payload["summaries"]}


def fmt(row: Dict[str, Any], *, percent: bool = True) -> str:
    if row.get("value", "") == "":
        return "—"
    scale = 100 if percent else 1
    value = float(row["value"]) * scale
    if row.get("ci95_low", "") != "" and row.get("ci95_high", "") != "":
        return f"{value:.1f} [{float(row['ci95_low']) * scale:.1f}, {float(row['ci95_high']) * scale:.1f}]"
    return f"{value:.1f}"


def pick(records: Sequence[Dict[str, Any]], dataset: str, method: str, metric: str) -> Dict[str, Any]:
    return next(row for row in records if row["dataset"] == dataset and row["method"] == method and row["metric"] == metric)


def pick_complete_gpt(records: Sequence[Dict[str, Any]], dataset: str, metric: str) -> Dict[str, Any]:
    return next(
        (row for row in records if row["dataset"] == dataset and row["method"] == "GPT-5.6 Sol (high)" and row["metric"] == metric),
        {"value": "", "n": "", "ci95_low": "", "ci95_high": ""},
    )


def describe_gap(ours: Dict[str, Any], current: Dict[str, Any]) -> str:
    if current.get("value", "") == "":
        return "not comparable to a complete GPT-5.6 Sol run because provider credit was exhausted"
    gap = 100.0 * (float(ours["value"]) - float(current["value"]))
    relation = "above" if gap >= 0 else "below"
    return f"{abs(gap):.1f} percentage points {relation} GPT-5.6 Sol"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    historical = {
        "schema_version": 1,
        "models": [
            recover_historical_model(model="gpt-4o-mini", cache_path=REPO_ROOT / "benchmarks" / "llm_baselines_cache.jsonl", old_openai=True),
            recover_historical_model(model="claude-opus-4-6", cache_path=REPO_ROOT / "benchmarks" / "claude_baselines_cache.jsonl", old_openai=False),
        ],
    }
    (OUTPUT_DIR / "historical_direct_vlm_results.json").write_text(json.dumps(historical, indent=2, sort_keys=True), encoding="utf-8")

    complete_gpt_path = OUTPUT_DIR / "gpt-5.6-sol_results.json"
    partial_gpt_path = OUTPUT_DIR / "gpt-5.6-sol_partial_results.json"
    gpt_path = complete_gpt_path if complete_gpt_path.exists() else partial_gpt_path
    if not gpt_path.exists():
        raise SystemExit(f"Missing direct evaluation (complete or partial): {complete_gpt_path}")
    gpt_circular_path = OUTPUT_DIR / "gpt-5.6-sol_sat_circular_results.json"
    gpt = read_json(gpt_path)
    gpt_circular = read_json(gpt_circular_path) if gpt_circular_path.exists() else None
    gpt_rows = summary_map(gpt)
    gpt_circular_rows = summary_map(gpt_circular) if gpt_circular and gpt_circular.get("complete", True) else {}
    gpt_coverage = gpt.get("coverage", {task: {"cached": row["n"], "expected": row["n"]} for task, row in gpt_rows.items()})

    def gpt_complete(task: str) -> bool:
        coverage = gpt_coverage.get(task, {})
        return bool(task in gpt_rows and coverage.get("cached") == coverage.get("expected"))

    availability = {
        "as_of_date": "2026-08-08",
        "openai": {
            "model": "gpt-5.6-sol", "status": "evaluated" if gpt.get("complete", True) else "partially_evaluated",
            "reasoning_effort": "high", "cached_responses": gpt.get("cached_responses", gpt.get("expected_responses")),
            "expected_responses": gpt.get("expected_responses"),
            "reason_if_partial": "provider_credit_balance_exhausted" if not gpt.get("complete", True) else None,
        },
        "anthropic": {
            "model": "claude-fable-5", "status": "not_evaluated", "reason": "provider_insufficient_credit",
            "fallback": "committed cached claude-opus-4-6 evaluation",
        },
    }
    (OUTPUT_DIR / "provider_availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True), encoding="utf-8"
    )
    historical_rows = {model["model"]: summary_map(model) for model in historical["models"]}
    audit = read_json(ARCHIVE)
    records: List[Dict[str, Any]] = []

    def add_gpt_record(*, dataset: str, task: str, protocol: str, metric: str) -> None:
        coverage = gpt_coverage.get(task, {"cached": 0, "expected": ""})
        row = gpt_rows.get(task)
        complete = gpt_complete(task)
        add_record(
            records, dataset=dataset, protocol=protocol,
            method="GPT-5.6 Sol (high)" if complete else "GPT-5.6 Sol (high; partial)",
            category="direct frontier VLM", metric=metric if row is None else row["metric"],
            value="" if row is None else row["value"],
            n=(row["n"] if complete and row is not None else f"{coverage.get('cached', 0)}/{coverage.get('expected', '')} cached"),
            ci_low="" if row is None else row["ci95_low"], ci_high="" if row is None else row["ci95_high"],
            ci_method="" if row is None else row["ci_method"], source=gpt_path.name,
            comparability="matched" if complete else "incomplete due provider credit exhaustion; excluded from headline claims",
        )

    for dataset, task, protocol in (
        ("Tetris rotation", "tetris_ood", "test_ood_angle; current GPT fixed balanced n=100 subset"),
        ("Colored rotation", "colored_ood", "test_ood_angle; current GPT fixed balanced n=100 subset"),
    ):
        experiment_stem = "tetris" if task == "tetris_ood" else "colored"
        archive_record(records, audit, dataset=dataset, protocol=protocol, experiment=f"fot_flow_ppo_{experiment_stem}",
                       split="test_ood_angle", metric="accuracy", method="FoT flow+PPO (legacy)", category="ours", n=1000)
        archive_record(records, audit, dataset=dataset, protocol=protocol, experiment=f"{experiment_stem}_rotation_cnn",
                       split="test_ood_angle", metric="accuracy", method="CNN trained on task", category="trained local", n=1000)
        archive_record(records, audit, dataset=dataset, protocol=protocol, experiment=f"{experiment_stem}_rotation_vit",
                       split="test_ood_angle", metric="accuracy", method="ViT trained on task", category="trained local", n=1000)
        archive_record(records, audit, dataset=dataset, protocol=protocol, experiment=f"dinov3_vits16_frozen_{experiment_stem}",
                       split="test_ood_angle", metric="accuracy", method="DINOv3 frozen + trained head", category="trained local", n=1000)
        add_gpt_record(dataset=dataset, task=task, protocol=protocol, metric="accuracy")
        old_task = "tetris" if task == "tetris_ood" else "colored"
        for model in ("gpt-4o-mini", "claude-opus-4-6"):
            old = historical_rows[model][old_task]
            add_record(records, dataset=dataset, protocol="procedural random sample, seed 0", method=model, category="direct historical VLM",
                       metric=old["metric"], value=old["value"], n=old["n"], ci_low=old["ci95_low"], ci_high=old["ci95_high"],
                       ci_method=old["ci_method"], source="historical_direct_vlm_results.json",
                       comparability="same task distribution, different item set")

    # Ganis-Kievit full n=78.
    ganis_flow = read_json(REPO_ROOT / "results" / "neurreps_flow_v1" / "ganis3d_zero_shot" / "zero_shot_results.json")
    colored_accuracy = next(row for row in ganis_flow["aggregates"] if row["source_model"] == "colored" and row["protocol"] == "provided_angle" and row["metric"] == "accuracy")
    add_record(records, dataset="Ganis-Kievit 3-D", protocol="full n=78", method="Frozen trajectory flow (colored source)", category="ours",
               metric="accuracy", value=colored_accuracy["mean"], n="78/seed × 3 seeds", ci_low=colored_accuracy["ci95_low"], ci_high=colored_accuracy["ci95_high"],
               ci_method="student_t_over_independent_seeds", source="neurreps_flow_v1/ganis3d_zero_shot/zero_shot_results.json")
    archive_record(records, audit, dataset="Ganis-Kievit 3-D", protocol="full n=78", experiment="dinov3_vits16_frozen_ganis3d",
                   split="test_ganis3d", metric="accuracy", method="DINOv3 frozen + trained head", category="trained local", n=78)
    archive_record(records, audit, dataset="Ganis-Kievit 3-D", protocol="full n=78", experiment="classical_cv_bfs_ganis3d",
                   split="test", metric="accuracy", method="ViperGPT program-of-tools", category="tool baseline", n=78)
    add_gpt_record(dataset="Ganis-Kievit 3-D", task="ganis3d", protocol="full n=78", metric="accuracy")
    for model in ("gpt-4o-mini", "claude-opus-4-6"):
        row = historical_rows[model]["ganis3d"]
        add_record(records, dataset="Ganis-Kievit 3-D", protocol="full n=78; shuffled order", method=model, category="direct historical VLM",
                   metric=row["metric"], value=row["value"], n=row["n"], ci_low=row["ci95_low"], ci_high=row["ci95_high"],
                   ci_method=row["ci_method"], source="historical_direct_vlm_results.json")

    # Maze task interfaces.
    flow = read_json(REPO_ROOT / "results" / "neurreps_flow_v1" / "posthoc_v2" / "posthoc_results.json")
    def flow_metric(name: str) -> Dict[str, Any]:
        return next(row for row in flow["aggregates"] if row["task"] == "maze" and row["metric"] == name)
    for metric, display in (("goal_reached", "goal_reached"), ("endpoint_iou", "endpoint_iou"), ("intermediate_prefix_iou", "intermediate_prefix_iou")):
        row = flow_metric(metric)
        add_record(records, dataset="Maze generation", protocol="held-out procedural validation, 3 seeds", method="Trajectory flow (no PPO)", category="ours",
                   metric=display, value=row["mean"], n="3 seeds", ci_low=row["ci95_low_display"], ci_high=row["ci95_high_display"],
                   ci_method="student_t_over_independent_seeds", source="neurreps_flow_v1/posthoc_v2/posthoc_results.json")
    for task, dataset in (("maze_trace", "Maze trace validity"), ("maze_solve", "Maze path solving")):
        expected = gpt_coverage.get(task, {}).get("expected", "")
        add_gpt_record(dataset=dataset, task=task, protocol=f"fixed current subset n={expected}",
                       metric="success_rate" if task == "maze_solve" else "accuracy")
        old_task = task
        for model in ("gpt-4o-mini", "claude-opus-4-6"):
            old = historical_rows[model][old_task]
            add_record(records, dataset=dataset, protocol="procedural random sample, seed 0", method=model, category="direct historical VLM",
                       metric=old["metric"], value=old["value"], n=old["n"], ci_low=old["ci95_low"], ci_high=old["ci95_high"],
                       ci_method=old["ci_method"], source="historical_direct_vlm_results.json",
                       comparability="same task distribution, different item set")
        archive_record(records, audit, dataset=dataset, protocol="procedural random samples across 3 seeds", experiment=f"classical_cv_bfs_{task}",
                       split="test", metric="accuracy" if task == "maze_trace" else "success_rate", method="ViperGPT program-of-tools", category="tool baseline")
    archive_record(records, audit, dataset="Maze trace validity", protocol="procedural random samples across 3 seeds", experiment="maze_trace_fot_controller",
                   split="test", metric="accuracy", method="FoT trace controller (legacy)", category="ours", n=200)

    # SAT-v2 real test.
    for experiment, method, category in (("sat_v2_fot", "FoT heatmap MCQ", "ours"), ("sat_v2_direct", "Matched direct MCQ", "trained local")):
        archive_record(records, audit, dataset="SAT-v2 / SAT-Real", protocol="first/all 150 test items; single answer ordering", experiment=experiment,
                       split="test_real", metric="accuracy", method=method, category=category, n=150)
    add_gpt_record(dataset="SAT-v2 / SAT-Real", task="sat_v2", protocol="all 150 test items; single answer ordering", metric="accuracy")
    circular_row = gpt_circular_rows.get("sat_v2")
    if circular_row is not None:
        add_record(records, dataset="SAT-v2 / SAT-Real", protocol="paper circular evaluation; original + reversed order",
                   method="GPT-5.6 Sol (high, circular)", category="direct frontier VLM", metric="accuracy",
                   value=circular_row["value"], n=circular_row["n"], ci_low=circular_row["ci95_low"], ci_high=circular_row["ci95_high"],
                   ci_method=circular_row["ci_method"], source="gpt-5.6-sol_sat_circular_results.json",
                   comparability="matched published circular protocol")
    for method, value, category, source in (
        ("GPT-4V (paper)", 0.507, "published general VLM", SPATIAL_DREAMER_URL),
        ("GPT-4o (paper)", 0.603, "published general VLM", SPATIAL_DREAMER_URL),
        ("GPT-4.1 (paper)", 0.740, "published general VLM", SPATIAL_DREAMER_URL),
        ("SpatialDreamer", 0.939, "published spatial SOTA", SPATIAL_DREAMER_URL),
    ):
        add_record(records, dataset="SAT-v2 / SAT-Real", protocol="paper circular evaluation", method=method, category=category,
                   metric="accuracy", value=value, n=150, source=source, comparability="published protocol; circular answer-order evaluation")

    # BLINK Multi-view validation.
    blink = read_json(REPO_ROOT / "results" / "neurreps_flow_v1" / "blink_multiview_zero_shot" / "zero_shot_results.json")
    colored_blink = next(row for row in blink["aggregates"] if row["source_model"] == "colored" and row["metric"] == "accuracy")
    add_record(records, dataset="BLINK Multi-view", protocol="full validation n=133", method="Frozen trajectory flow (colored source)", category="ours",
               metric="accuracy", value=colored_blink["mean"], n="133/seed × 3 seeds", ci_low=colored_blink["ci95_low"], ci_high=colored_blink["ci95_high"],
               ci_method="student_t_over_independent_seeds", source="neurreps_flow_v1/blink_multiview_zero_shot/zero_shot_results.json")
    add_gpt_record(dataset="BLINK Multi-view", task="blink_multiview", protocol="full validation n=133", metric="accuracy")
    for method, value, category, protocol, source in (
        ("GPT-4V direct", 0.5865, "published general VLM", "full validation n=133", BLINK_PAPER_URL),
        ("Gemini Pro direct", 0.4135, "published general VLM", "full validation n=133", BLINK_PAPER_URL),
        ("P2 + Gemini 2.5 Pro", 0.6391, "published tool VLM", "full validation n=133", P2_URL),
        ("DR-MV3D", 0.564, "published spatial model", "BLINK Multi-view", DR_MV3D_URL),
        ("Human", 0.9248, "human", "full validation n=133", BLINK_PAPER_URL),
        ("LoFTR specialist", 0.9022, "published CV specialist", "paper dev/test table", BLINK_PAPER_URL),
    ):
        add_record(records, dataset="BLINK Multi-view", protocol=protocol, method=method, category=category, metric="accuracy", value=value,
                   n=133 if "validation" in protocol else "", source=source,
                   comparability="matched public validation" if protocol.startswith("full validation") else "published reference; split/protocol differs")

    csv_path = OUTPUT_DIR / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    comparison_payload = {
        "schema_version": 1,
        "as_of_date": "2026-08-08",
        "published_sources": [SAT_PAPER_URL, SPATIAL_DREAMER_URL, BLINK_PAPER_URL, P2_URL, DR_MV3D_URL],
        "records": records,
    }
    (OUTPUT_DIR / "comparison_table.json").write_text(json.dumps(comparison_payload, indent=2, sort_keys=True), encoding="utf-8")

    main_rows = [
        ("Tetris unseen angles", pick(records, "Tetris rotation", "FoT flow+PPO (legacy)", "accuracy"), pick_complete_gpt(records, "Tetris rotation", "accuracy"), "GPT-4o-mini 52.4†", "No published matched SOTA"),
        ("Colored unseen angles", pick(records, "Colored rotation", "FoT flow+PPO (legacy)", "accuracy"), pick_complete_gpt(records, "Colored rotation", "accuracy"), "Claude Opus 4.6 51.6†", "No published matched SOTA"),
        ("Ganis-Kievit 3-D", pick(records, "Ganis-Kievit 3-D", "Frozen trajectory flow (colored source)", "accuracy"), pick_complete_gpt(records, "Ganis-Kievit 3-D", "accuracy"), "Claude Opus 4.6 53.8", "Program-of-tools 65.4"),
        ("SAT-Real", pick(records, "SAT-v2 / SAT-Real", "FoT heatmap MCQ", "accuracy"), pick_complete_gpt(records, "SAT-v2 / SAT-Real", "accuracy"), "GPT-4V 50.7‡", "SpatialDreamer 93.9‡"),
        ("BLINK Multi-view", pick(records, "BLINK Multi-view", "Frozen trajectory flow (colored source)", "accuracy"), pick_complete_gpt(records, "BLINK Multi-view", "accuracy"), "GPT-4V 58.7", "P2+Gemini 63.9 / LoFTR 90.2‡"),
    ]
    report = [
        "# Cross-dataset spatial baseline comparison", "",
        "The exact-rotation operator is retained only as a diagnostic and is not the target comparator. The headline compares learned FoT/trajectory-flow systems with direct frontier VLMs and published spatial systems. Values are accuracy percentages unless stated otherwise; bracketed intervals are 95% confidence intervals.", "",
        "| Dataset | Our method | GPT-5.6 Sol (direct, high) | Prior direct VLM | Strong specialist / SOTA |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, ours, current, prior, specialist in main_rows:
        report.append(f"| {name} | {fmt(ours)} | {fmt(current)} | {prior} | {specialist} |")
    circular_display = fmt(circular_row) if circular_row is not None else "not completed (provider credit exhausted)"
    sat_protocol_note = (
        "The main table uses fixed-order GPT-5.6 for a like-for-like comparison with the repository's FoT/direct runs; "
        "the complete table also reports GPT-5.6 under the matched circular protocol."
        if circular_row is not None else
        "The GPT-5.6 circular pass could not run after provider credit was exhausted; published circular results remain protocol-labelled references."
    )
    report.extend([
        "", "## Maze and intermediate-process metrics", "",
        f"The complete GPT-5.6 maze run was not obtained before provider credit was exhausted. The deterministic program-of-tools baseline is 100% on both trace validity and path solving. These are not the same interface as the trajectory flow, which reaches 100% goal activation, 97.5% endpoint IoU, and 84.2% intermediate-prefix IoU over three seeds.",
        "", "## Reading the table", "",
        "- † The historical direct VLM used the same procedural distribution but a different random item set from the committed unseen-angle manifests.",
        f"- ‡ Published SAT values use circular answer-order evaluation. {sat_protocol_note} LoFTR's 90.2% is from the BLINK paper's separate dev/test table.",
        "- The rebuilt trajectory flow currently has no same/different head on the 2-D rotation manifests. The Tetris/colored accuracy rows therefore use the legacy flow+PPO classifier; current flow quality is audited separately (single-source silhouette IoU: 87.4% Tetris, 82.5% colored).",
        f"- OpenAI credit was exhausted after {gpt.get('cached_responses', gpt.get('expected_responses'))}/{gpt.get('expected_responses')} GPT-5.6 responses. Only tasks marked complete are used in headline comparisons; partial and missing rows remain in the complete audit table for coverage accounting.",
        "- Claude Fable 5 was selected and probed, but the Anthropic account returned `credit balance is too low`; no value is reported. Cached Claude Opus 4.6 results remain included.",
        "", "## Main findings", "",
        f"- On Tetris unseen angles, the legacy FoT classifier is {describe_gap(main_rows[0][1], main_rows[0][2])}; on colored unseen angles it is {describe_gap(main_rows[1][1], main_rows[1][2])}. It narrowly exceeds the old cached direct VLMs but does not beat the frontier model. These rows do not yet test the rebuilt auditable flow.",
        "- On Ganis-Kievit 3-D, the frozen colored-source trajectory flow reaches 61.5%: 7.7 points above cached Claude Opus 4.6 and 11.5 above cached GPT-4o-mini, but 3.8 below the program-of-tools baseline. GPT-5.6 reached 57.3% on 75/78 cached items; that incomplete result is excluded from the headline.",
        f"- On SAT-Real, FoT reaches 54.4%. This is 3.7 points above published GPT-4V, but 19.6 below GPT-4.1 and 39.5 below SpatialDreamer. Those published values use circular answer-order evaluation; our GPT-5.6 circular result is {circular_display}.",
        "- BLINK Multi-view is the strongest result for the paper's generalization claim: the frozen colored-source flow reaches 74.7% on all 133 validation items, 16.0 points above GPT-4V and 10.8 above P2 + Gemini 2.5 Pro. It remains 15.5 points below the LoFTR specialist and 17.8 below humans.",
        "- Maze endpoint and intermediate-state metrics support process auditability, but they are not accuracy-equivalent to direct path-text generation. They are kept separate rather than averaged into an artificial cross-task score.",
        "", "## Complete audit table", "",
        "This table contains every collected learned, tool, VLM, specialist, and human result used in the audit. `Comparability` identifies rows whose item set, answer-order protocol, split, or interface differs.", "",
        "| Dataset | Method | Category | Metric | Value (95% CI) | n | CI method | Comparability | Protocol |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ])
    for record in records:
        interval = fmt(record)
        protocol = str(record["protocol"]).replace("|", "\\|")
        report.append(
            f"| {record['dataset']} | {record['method']} | {record['category']} | {record['metric']} | "
            f"{interval} | {record['n']} | {record['ci_method']} | {record['comparability']} | {protocol} |"
        )
    report.extend([
        "", "## Sources", "",
        "[OpenAI GPT-5.6 Sol documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Anthropic Claude Fable 5](https://www.anthropic.com/claude/fable), [SAT dataset card](https://huggingface.co/datasets/array/SAT), [SAT paper](https://arxiv.org/abs/2412.07755), [SpatialDreamer](https://arxiv.org/abs/2512.07733), [BLINK paper](https://arxiv.org/abs/2404.12390), [P2 / Perception Programs](https://openaccess.thecvf.com/content/CVPR2026/html/Janjua_Dont_Show_Pixels_Show_Cues_Unlocking_Visual_Tool_Reasoning_in_CVPR_2026_paper.html), [DR-MV3D](https://dr-mv3d.github.io/).",
    ])
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {len(records)} tidy comparison rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
