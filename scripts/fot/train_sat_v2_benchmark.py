from __future__ import annotations

"""Matched direct and FoT visual-trace models on the external SAT-v2 benchmark."""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.fot.train_sat_v2_fot_vqa import collate_fn
from utils.fot.external_datasets import SATv2Dataset
from utils.fot.metrics import wilson_accuracy_ci
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.supervised_models import count_parameters
from utils.fot.toy_datasets import ToyMCQDataset
from utils.fot.torch_utils import seeded_generator, set_seed
from utils.fot.vqa_heatmap_model import DirectMCQModel, FoTHeatmapMCQModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched SAT-v2 direct/FoT benchmarks.")
    parser.add_argument("--model", choices=("direct", "fot"), required=True)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny local schema-compatible dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-validation", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--text-buckets", type=int, default=50000)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--text-max-len", type=int, default=64)
    parser.add_argument("--flow-dim", type=int, default=32)
    parser.add_argument("--vision-feat-dim", type=int, default=128)
    parser.add_argument("--sketch-steps", type=int, default=6)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, *, save_predictions: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    predictions: List[Dict[str, Any]] = []
    offset = 0
    subgroup_correct: Dict[str, int] = {}
    subgroup_total: Dict[str, int] = {}
    for batch in loader:
        labels = batch["labels"].to(device)
        logits, _ = model(images=batch["images"].to(device), q_input_ids=batch["q_input_ids"].to(device),
            q_attention_mask=batch["q_attention_mask"].to(device),
            choice_input_ids=batch["choice_input_ids"].to(device),
            choice_attention_mask=batch["choice_attention_mask"].to(device))
        probabilities = torch.softmax(logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").cpu())
        total += int(labels.numel()); correct += int((predicted == labels).sum().item())
        if save_predictions:
            for local, (label, pred, probs) in enumerate(zip(labels.cpu(), predicted.cpu(), probabilities.cpu())):
                question_type = batch.get("question_types", ["unknown"] * len(labels))[local]
                sample_id = batch.get("sample_ids", [f"satv2-{offset + i:07d}" for i in range(len(labels))])[local]
                predictions.append({"sample_id": str(sample_id), "question_type": str(question_type),
                                    "num_images": int(batch.get("num_images", [1] * len(labels))[local]), "label": int(label),
                                    "prediction": int(pred), "probabilities": [float(x) for x in probs],
                                    "confidence": float(probs.max())})
        for question_type, label, pred in zip(batch.get("question_types", ["unknown"] * len(labels)), labels.cpu(), predicted.cpu()):
            key = str(question_type); subgroup_total[key] = subgroup_total.get(key, 0) + 1
            subgroup_correct[key] = subgroup_correct.get(key, 0) + int(label == pred)
        offset += int(labels.numel())
    ci_low, ci_high = wilson_accuracy_ci(correct, total)
    subgroups = {}
    for key in sorted(subgroup_total):
        low, high = wilson_accuracy_ci(subgroup_correct[key], subgroup_total[key])
        subgroups[key] = {"n": subgroup_total[key], "accuracy": subgroup_correct[key] / subgroup_total[key],
                          "accuracy_ci95_low": low, "accuracy_ci95_high": high}
    return {"n": total, "accuracy": correct / total, "accuracy_ci95_low": ci_low,
            "accuracy_ci95_high": ci_high, "accuracy_ci_method": "wilson_test_items",
            "log_loss": loss_sum / total, "by_question_type": subgroups}, predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1,
        "experiment": {"name": f"sat_v2_{args.model}", "seed": args.seed, "device": args.device,
                       "preliminary": bool(args.preliminary)},
        "arguments": {key: value for key, value in resolved.items() if key != "seed"}})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    if args.smoke:
        train_dataset = ToyMCQDataset(n_samples=args.max_train, image_size=args.image_size, seed=args.seed)
        validation_dataset = ToyMCQDataset(n_samples=args.max_validation, image_size=args.image_size, seed=args.seed + 1)
        test_dataset = ToyMCQDataset(n_samples=args.max_test, image_size=args.image_size, seed=args.seed + 2)
    else:
        train_dataset = SATv2Dataset(split=args.train_split, image_size=args.image_size, max_samples=args.max_train,
                                     cache_dir=args.cache_dir, streaming=args.streaming)
        validation_dataset = SATv2Dataset(split=args.validation_split, image_size=args.image_size,
                                          max_samples=args.max_validation, cache_dir=args.cache_dir,
                                          streaming=args.streaming)
        test_dataset = SATv2Dataset(split=args.test_split, image_size=args.image_size,
                                    max_samples=args.max_test, cache_dir=args.cache_dir,
                                    streaming=args.streaming)
    collate = partial(collate_fn, text_buckets=args.text_buckets, text_max_len=args.text_max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              generator=seeded_generator(args.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)
    common = dict(num_text_buckets=args.text_buckets, text_dim=args.text_dim,
                  vision_feat_dim=args.vision_feat_dim, mlp_hidden=args.mlp_hidden)
    model = (FoTHeatmapMCQModel(**common, flow_dim=args.flow_dim, sketch_steps=args.sketch_steps)
             if args.model == "fot" else DirectMCQModel(**common)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = args.output_dir / "best_checkpoint.pt"
    history: List[Dict[str, Any]] = []
    best = -1.0; best_epoch = 0; started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; count = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            logits, _ = model(images=batch["images"].to(device), q_input_ids=batch["q_input_ids"].to(device),
                q_attention_mask=batch["q_attention_mask"].to(device),
                choice_input_ids=batch["choice_input_ids"].to(device),
                choice_attention_mask=batch["choice_attention_mask"].to(device))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(labels.numel()); count += int(labels.numel())
        validation, _ = evaluate(model, validation_loader, device, save_predictions=False)
        row = {"epoch": epoch, "train_loss": loss_sum / count, "validation": validation,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row, sort_keys=True))
        if validation["accuracy"] > best:
            best = float(validation["accuracy"]); best_epoch = epoch; torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation, predictions = evaluate(model, validation_loader, device, save_predictions=True)
    write_json(args.output_dir / "predictions_validation.json", {"predictions": predictions})
    test_metrics, test_predictions = evaluate(model, test_loader, device, save_predictions=True)
    write_json(args.output_dir / "predictions_test_real.json", {"predictions": test_predictions})
    write_json(args.output_dir / "epoch_metrics.json", {"epochs": history})
    summary = {"experiment_name": f"sat_v2_{args.model}", "task": "sat_v2", "model": args.model,
        "seed": args.seed, "parameter_count": count_parameters(model), "train_samples": len(train_dataset),
        "best_epoch": best_epoch, "best_validation_accuracy": best,
        "metrics": {"validation": validation, "test_real": test_metrics},
        "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary),
        "benchmark_role": "external_dynamic_spatial_reasoning"}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
