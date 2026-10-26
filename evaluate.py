"""Explicit final CIFAR-10 evaluation and model-only inference timing.

Training never reads the official test partition. Formal evaluation requires all
six predeclared runs to have completed. Synthetic/nonformal checks are explicitly
labelled and are rejected by the paper-result summarizer.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import models
from kd_utils import (build_transform, file_sha256, load_checkpoint, load_json,
                      make_datasets, model_state_sha256, save_json,
                      software_information)


ROLES = ("teacher", "baseline", "kd")
SEEDS = (0, 1)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def formal_config(config):
    """The frozen, short study, rather than arbitrary successful training."""
    return (config.get("dataset") == "CIFAR10" and config.get("epochs") == 10
            and config.get("batch_size") == 128 and config.get("workers") == 2
            and config.get("precision") == "float32"
            and config.get("device") in ("cuda", "cuda:0")
            and config.get("augmentation") == "none"
            and config.get("split", {}).get("split_seed") == 20261026
            and config.get("split", {}).get("train_size") == 45000
            and config.get("split", {}).get("val_size") == 5000
            and config.get("temperature") == 2.0 and config.get("alpha") == 0.25
            and config.get("normalize_mean") == MEAN and config.get("normalize_std") == STD
            and config.get("optimizer") == {"name": "Adam", "lr": 0.001,
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0})


def validate_training(checkpoint_path, allow_nonformal=False, synthetic=False):
    """Check selected model tensors, config, complete training and atomic hashes.

    Returns (best checkpoint, training summary, checkpoint file SHA-256).
    This routine does not instantiate any dataset or perform inference.
    """
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    require(checkpoint_path.name == "best.pth", "Evaluate validation-selected best.pth only")
    folder = checkpoint_path.parent
    best_hash = file_sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    role, seed = config.get("role"), config.get("seed")
    require(role in ROLES and type(seed) is int and seed in SEEDS, "Unexpected run role or seed")
    require(folder.name == f"{role}_seed{seed}", "Run directory identity differs from checkpoint")
    require(load_json(folder / "config.json") == config, "config.json differs from checkpoint config")
    require(config.get("source") == {
        "models": models.source_information(),
        "train_py_sha256": file_sha256(Path(__file__).parent / "train.py"),
        "kd_utils_py_sha256": file_sha256(Path(__file__).parent / "kd_utils.py"),
    }, "Training source provenance differs from the current verified project")
    require(config.get("dataset") in ("CIFAR10", "synthetic"), "Unexpected dataset")
    require(synthetic == (config["dataset"] == "synthetic"),
            "--synthetic must match the checkpoint's explicit dataset identity")
    state_hash = model_state_sha256(checkpoint["model_state"])
    require(state_hash == checkpoint.get("model_state_sha256"), "Best model tensor hash failed")
    require(state_hash == checkpoint.get("best_model_state_sha256"), "Best model hash metadata disagrees")
    require(config.get("init_sha256") == checkpoint.get("init_sha256"), "Initialization hash disagrees")
    summary = load_json(folder / "summary.json")
    require(summary.get("status") == "completed", "Training must complete before final evaluation")
    require(summary.get("completed_epochs") == config.get("epochs")
            == summary.get("requested_epochs"), "Training completion count differs from config")
    require(type(checkpoint.get("epoch")) is int
            and 1 <= checkpoint["epoch"] <= summary["completed_epochs"], "Invalid selected epoch")
    for field, expected in (("best_epoch", checkpoint["epoch"]),
                            ("best_val_acc", checkpoint["best_val_acc"]),
                            ("init_sha256", config["init_sha256"]),
                            ("split_sha256", config["split_sha256"]),
                            ("best_checkpoint_sha256", best_hash),
                            ("best_model_state_sha256", state_hash)):
        require(summary.get(field) == expected, f"Training summary disagrees: {field}")
    require(0 <= checkpoint["best_val_acc"] <= 1 and math.isfinite(checkpoint["best_val_acc"]),
            "Invalid validation accuracy")
    last_path = folder / "last.pth"
    require(last_path.is_file() and file_sha256(last_path) == summary.get("last_checkpoint_sha256"),
            "Missing or modified final training checkpoint")
    last = load_checkpoint(last_path)
    require(last.get("config") == config and last.get("epoch") == summary["completed_epochs"],
            "last.pth differs from completed training config/epoch")
    require(last.get("best_epoch") == checkpoint["epoch"]
            and last.get("best_model_state_sha256") == state_hash,
            "Final training checkpoint points to a different validation-selected model")
    require(model_state_sha256(last["model_state"]) == last.get("model_state_sha256"),
            "Final training tensor hash failed")
    if not allow_nonformal:
        require(formal_config(config), "Not a complete frozen 10-epoch CIFAR-10 experiment")
    return checkpoint, summary, best_hash


def require_complete_matrix(runs_root):
    """Avoid opening official test data while other planned training is unfinished."""
    splits = set()
    for seed in SEEDS:
        for role in ROLES:
            path = Path(runs_root) / f"{role}_seed{seed}"
            config = load_json(path / "config.json")
            summary = load_json(path / "summary.json")
            require(formal_config(config) and config.get("role") == role and config.get("seed") == seed,
                    f"The six-run matrix is not the frozen experiment: {path}")
            require(summary.get("status") == "completed"
                    and summary.get("completed_epochs") == summary.get("requested_epochs") == 10,
                    f"All six training runs must finish before final testing: {path}")
            splits.add(config.get("split_sha256"))
    require(len(splits) == 1 and None not in splits, "The six runs do not share one validation split")


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name,
                                         suffix=".tmp", encoding="utf-8-sig",
                                         newline="", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def per_class_metrics(confusion, classes):
    rows = []
    for index, name in enumerate(classes):
        tp = int(confusion[index, index])
        support, predicted = int(confusion[index].sum()), int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_index": index, "class_name": name, "support": support,
                     "predicted_count": predicted, "true_positive": tp,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


@torch.no_grad()
def measure_latency(model, device, warmup=20, repeats=50):
    """Model forward only; image preprocessing and host/device copies excluded."""
    result = {"scope": "model_only_device_resident", "unit": "milliseconds_per_batch",
              "input": "zeros, float32, N x 3 x 32 x 32", "warmup": warmup,
              "repeats": repeats, "device": str(device)}
    for batch_size in (1, 128):
        images = torch.zeros(batch_size, 3, 32, 32, device=device)
        for _ in range(warmup):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            events = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(images)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize(device)
            times = [float(start.elapsed_time(end)) for start, end in events]
        else:
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                model(images)
                times.append((time.perf_counter() - started) * 1000)
        require(all(math.isfinite(t) and t > 0 for t in times), "Invalid latency measurements")
        result[f"batch{batch_size}_ms_median"] = float(np.median(times))
        result[f"batch{batch_size}_ms_p95"] = float(np.percentile(times, 95))
        result[f"batch{batch_size}_ms_samples"] = times
    return result


def evaluate(args):
    require(args.batch_size == 128 and args.workers >= 0, "Evaluation uses batch128 and nonnegative workers")
    require(args.latency_warmup >= 1 and args.latency_repeats >= 2, "Invalid inference timing budget")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    require(device.type in ("cuda", "cpu"), "Only CPU/CUDA devices are supported")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(args.checkpoint).resolve(strict=True)
    checkpoint, summary, weights_hash = validate_training(path, args.allow_nonformal, args.synthetic)
    config = checkpoint["config"]
    formal = formal_config(config) and not args.allow_nonformal and not args.synthetic
    if formal:
        require(args.workers == 2 and str(device) in ("cuda", "cuda:0"),
                "Formal evaluation uses the frozen CUDA/workers2 setup")
        require(args.latency_warmup == 20 and args.latency_repeats == 50,
                "Formal inference timing uses 20 warmups and 50 repeated forwards")
        require_complete_matrix(path.parent.parent)
    model = models.build_model(config["role"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.float().to(device).eval()
    if args.synthetic:
        _, test_data, _ = make_datasets(args.data_root, path.parent.parent / "splits", synthetic=True)
        classes = [str(index) for index in range(10)]
        partition, expected_count = "synthetic_validation_for_verification", 64
    else:
        # The only dataset construction in the project with train=False.
        test_data = datasets.CIFAR10(str(args.data_root), train=False, download=False,
                                     transform=build_transform())
        classes = list(test_data.classes)
        partition, expected_count = "official_test", 10000
    require(len(test_data) == expected_count, "Unexpected evaluation partition length")
    loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    confusion = np.zeros((10, 10), dtype=np.int64)
    predictions = []
    loss_sum, total = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            require(bool(torch.isfinite(logits).all()), "Non-finite evaluation logits")
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item())
            confidence, predicted = logits.softmax(dim=1).max(dim=1)
            actual = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()
            confidence = confidence.cpu().numpy()
            confusion += np.bincount(actual * 10 + predicted, minlength=100).reshape(10, 10)
            predictions.extend({"sample_index": total + i, "true_class": int(label),
                                "predicted_class": int(guess), "confidence": float(probability)}
                               for i, (label, guess, probability) in enumerate(zip(actual, predicted, confidence)))
            total += len(labels)
    require(total == expected_count and int(confusion.sum()) == total and math.isfinite(loss_sum),
            "Evaluation did not produce complete finite metrics")
    latency = measure_latency(model, device, args.latency_warmup, args.latency_repeats)
    require(file_sha256(path) == weights_hash, "Checkpoint changed during evaluation; results not saved")
    class_metrics = per_class_metrics(confusion, classes)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": path.parent.name, "role": config["role"], "seed": config["seed"],
              "checkpoint": "best.pth", "weights_sha256": weights_hash,
              "model_state_sha256": checkpoint["model_state_sha256"], "config": config,
              "selected_epoch": checkpoint["epoch"], "best_val_acc": checkpoint["best_val_acc"],
              "training_completed_epochs": summary["completed_epochs"], "formal": formal,
              "partition": partition, "accuracy_unit": "percent", "total": total,
              "correct": int(np.trace(confusion)), "accuracy_percent": 100 * int(np.trace(confusion)) / total,
              "loss": loss_sum / total, "parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
              "checkpoint_bytes": path.stat().st_size, "latency": latency,
              "per_class": class_metrics, "per_class_metric_unit": "fraction",
              "macro_f1": float(np.mean([row["f1"] for row in class_metrics])),
              "confusion_matrix_file": "confusion_matrix.csv", "predictions_file": "predictions.csv",
              "confusion_matrix_axes": "rows=true class, columns=predicted class",
              "evaluation_source_sha256": file_sha256(Path(__file__)),
              "runtime": {**software_information(device), "device": str(device),
                          "batch_size": 128, "workers": args.workers, "cpu_threads": 4}}
    folder = path.parent
    write_csv(folder / "predictions.csv", predictions,
              ["sample_index", "true_class", "predicted_class", "confidence"])
    write_csv(folder / "confusion_matrix.csv",
              [{"true_class": name, **{str(j): int(count) for j, count in enumerate(row)}}
               for name, row in zip(classes, confusion)], ["true_class", *map(str, range(10))])
    result["predictions_sha256"] = file_sha256(folder / "predictions.csv")
    result["confusion_matrix_sha256"] = file_sha256(folder / "confusion_matrix.csv")
    save_json(folder / "evaluation.json", result)
    print(f"{path.parent.name}: accuracy={result['accuracy_percent']:.4f}%, "
          f"cross_entropy={result['loss']:.6f}, formal={formal}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-nonformal", action="store_true",
                        help="Verification only: labels output nonformal; main summarizer rejects it")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
