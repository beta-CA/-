"""Explicit final evaluation of the predeclared exploratory extension.

Original runs/evaluations are immutable. All planned training must complete
before official test images are loaded; synthetic checks are labelled nonformal.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import extended_protocol as protocol
import models
from evaluate import measure_latency, per_class_metrics, require, write_csv
from kd_utils import (build_transform, file_sha256, load_json, make_datasets,
                      save_json, software_information)

ROOT = Path(__file__).resolve().parent


def expected_source():
    return {"evaluate_extended_py_sha256": file_sha256(Path(__file__)),
            "original_evaluate_py_sha256": file_sha256(ROOT / "evaluate.py")}


def validate_training_matrix(project_root, runs_root):
    """Read-only audit of training dependencies before any official test access."""
    from train_extended import validate_run
    from evaluate import validate_training as validate_legacy
    project_root, runs_root = Path(project_root).resolve(), Path(runs_root).resolve()
    protocol.verify_manifest(project_root)
    expected = {item["run_id"] for item in protocol.training_cases()}
    # A failed Windows directory rename can leave a validated initialization
    # backup. It is not a published training case and has no test evaluation.
    present = {path.parent.name for path in runs_root.glob("*/config.json")
               if not (path.parent.name.startswith(".") and ".initializing-" in path.parent.name)}
    require(present == expected, f"Expected exactly 19 extension training runs; missing={sorted(expected-present)}, "
            f"unexpected={sorted(present-expected)}")
    checked = {}
    for item in protocol.all_cases():
        case_id = item["run_id"]
        if item["reuse_legacy_teacher"]:
            path = project_root / "runs" / case_id / "best.pth"
            best, summary, digest = validate_legacy(path)
        else:
            path = runs_root / case_id / "best.pth"
            best, _, summary, digest = validate_run(path)
        config = best["config"]
        require(config["role"] == item["role"] and config["seed"] == item["seed"],
                f"Case identity differs: {case_id}")
        require(config["epochs"] == item["epochs"] and summary["completed_epochs"] == item["epochs"],
                f"Case training budget incomplete: {case_id}")
        checked[case_id] = {"checkpoint": best, "summary": summary, "hash": digest, "path": path}
    configs = [entry["checkpoint"]["config"] for entry in checked.values()]
    first = configs[0]
    for config in configs:
        for field in ("dataset", "data_root", "split_sha256", "batch_size", "workers", "device", "precision",
                      "optimizer", "normalize_mean", "normalize_std", "augmentation", "software"):
            require(config.get(field) == first.get(field), f"Shared protocol differs: {field}")
    for seed in protocol.MAIN_SEEDS:
        base = checked[f"baseline_seed{seed}"]["checkpoint"]["config"]
        teacher = checked[f"teacher_seed{seed}"]
        teacher_path = protocol.teacher_checkpoint(project_root, runs_root, seed).resolve()
        for item in protocol.training_cases():
            if item["role"] != "kd" or item["seed"] != seed:
                continue
            config = checked[item["run_id"]]["checkpoint"]["config"]
            require(config["init_sha256"] == base["init_sha256"],
                    f"Student initialization not paired: {item['run_id']}")
            dependency = config.get("teacher", {})
            require(dependency.get("seed") == seed and dependency.get("weights_sha256") == teacher["hash"]
                    and dependency.get("model_state_sha256") == teacher["checkpoint"]["model_state_sha256"]
                    and Path(dependency.get("checkpoint", "")).resolve() == teacher_path,
                    f"KD does not use its matching frozen teacher: {item['run_id']}")
    return checked


def evaluate_case(checkpoint_path, project_root, runs_root, data_root=None, device="cuda", workers=2,
                  synthetic=False, allow_nonformal=False, latency_warmup=20, latency_repeats=50,
                  dataset=None):
    """Evaluate one extended case. Official testing always checks the full matrix."""
    from train_extended import validate_run
    project_root, runs_root = Path(project_root).resolve(), Path(runs_root).resolve()
    path = Path(checkpoint_path).resolve(strict=True)
    data_root = Path(data_root).resolve() if data_root is not None else project_root / "data"
    require(workers >= 0 and latency_warmup >= 1 and latency_repeats >= 2, "Invalid evaluation options")
    require(bool(synthetic) == bool(allow_nonformal),
            "Verification requires both --synthetic and --allow-nonformal; real cases use the formal protocol")
    best, _, summary, digest = validate_run(path, allow_synthetic=synthetic)
    config = best["config"]
    require((config["dataset"] == "synthetic") == synthetic, "Evaluation/checkpoint dataset identities differ")
    formal = not synthetic
    torch.set_num_threads(4)
    device = torch.device(device)
    require(device.type in ("cpu", "cuda"), "Only CPU and CUDA are supported")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    if formal:
        require(str(device) in ("cuda", "cuda:0") and workers == 2,
                "Formal evaluation uses CUDA and workers=2")
        require(latency_warmup == 20 and latency_repeats == 50, "Formal latency protocol is 20/50")
        require(str(data_root) == config["data_root"], "Evaluation data root differs from training")
        validate_training_matrix(project_root, runs_root)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = models.build_model(config["role"])
    model.load_state_dict(best["model_state"], strict=True)
    model = model.float().to(device).eval().requires_grad_(False)
    if dataset is None:
        if synthetic:
            _, dataset, _ = make_datasets(data_root, runs_root / "splits", synthetic=True)
        else:
            dataset = datasets.CIFAR10(str(data_root), train=False, download=False, transform=build_transform())
    classes = [str(i) for i in range(10)] if synthetic else list(dataset.classes)
    expected_count = 64 if synthetic else 10000
    require(len(dataset) == expected_count, "Unexpected evaluation partition size")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    confusion = np.zeros((10, 10), dtype=np.int64)
    predictions = []
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            require(bool(torch.isfinite(logits).all()), "Nonfinite evaluation logits")
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item())
            confidence, predicted = logits.softmax(dim=1).max(dim=1)
            actual = labels.numpy()
            predicted, confidence = predicted.cpu().numpy(), confidence.cpu().numpy()
            confusion += np.bincount(actual * 10 + predicted, minlength=100).reshape(10, 10)
            predictions.extend({"sample_index": count + i, "true_class": int(label),
                                "predicted_class": int(guess), "confidence": float(probability)}
                               for i, (label, guess, probability) in enumerate(zip(actual, predicted, confidence)))
            count += len(labels)
    require(count == expected_count and int(confusion.sum()) == count and math.isfinite(loss_sum),
            "Evaluation was incomplete or nonfinite")
    latency = measure_latency(model, device, latency_warmup, latency_repeats)
    require(file_sha256(path) == digest, "Checkpoint changed during evaluation; no result saved")
    per_class = per_class_metrics(confusion, classes)
    result = {"schema_version": 1, "study": protocol.STUDY,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": path.parent.name, "case_id": config["case_id"], "role": config["role"],
              "seed": config["seed"], "group": config["group"],
              "checkpoint": "best.pth", "weights_sha256": digest,
              "model_state_sha256": best["model_state_sha256"], "config": config,
              "selected_epoch": best["epoch"], "best_val_acc": best["best_val_acc"],
              "training_completed_epochs": summary["completed_epochs"], "formal": formal,
              "partition": "synthetic_validation_for_verification" if synthetic else "official_test",
              "accuracy_unit": "percent", "total": count, "correct": int(np.trace(confusion)),
              "accuracy_percent": 100 * int(np.trace(confusion)) / count, "loss": loss_sum / count,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
              "checkpoint_bytes": path.stat().st_size, "latency": latency, "per_class": per_class,
              "per_class_metric_unit": "fraction", "macro_f1": float(np.mean([row["f1"] for row in per_class])),
              "confusion_matrix_file": "confusion_matrix.csv", "predictions_file": "predictions.csv",
              "confusion_matrix_axes": "rows=true class, columns=predicted class",
              "evaluation_source": expected_source(),
              "runtime": {**software_information(device), "device": str(device), "workers": workers,
                          "batch_size": 128, "cpu_threads": 4},
              "exploratory_extension": True,
              "test_note": "Original test results were previously inspected; no extension recipe is selected by test score."}
    folder = path.parent
    write_csv(folder / "predictions.csv", predictions,
              ["sample_index", "true_class", "predicted_class", "confidence"])
    write_csv(folder / "confusion_matrix.csv",
              [{"true_class": name, **{str(j): int(v) for j, v in enumerate(row)}}
               for name, row in zip(classes, confusion)], ["true_class", *map(str, range(10))])
    result["predictions_sha256"] = file_sha256(folder / "predictions.csv")
    result["confusion_matrix_sha256"] = file_sha256(folder / "confusion_matrix.csv")
    save_json(folder / "evaluation.json", result)
    print(f"{config['case_id']}: accuracy={result['accuracy_percent']:.4f}%, CE={result['loss']:.6f}, formal={formal}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Optional single-case evaluation; formal mode still checks all19")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-nonformal", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=50)
    args = parser.parse_args()
    project = args.project_root.resolve()
    runs = args.runs_root.resolve() if args.runs_root is not None else project / "runs_extended"
    output = args.output_root.resolve() if args.output_root is not None else project / "results_extended"
    data = args.data_root.resolve() if args.data_root is not None else project / "data"
    if args.checkpoint is not None:
        evaluate_case(args.checkpoint, project, runs, data, args.device, args.workers,
                      args.synthetic, args.allow_nonformal, args.latency_warmup, args.latency_repeats)
        return
    require(not args.synthetic and not args.allow_nonformal,
            "Synthetic evaluation must explicitly identify one --checkpoint")
    checked = validate_training_matrix(project, runs)
    # Verify referenced legacy evaluations without rewriting or rerunning them.
    from summarize import audit_run as audit_legacy_run
    for seed in (0, 1):
        audit_legacy_run(project / "runs" / f"teacher_seed{seed}")
    test = datasets.CIFAR10(str(data), train=False, download=False, transform=build_transform())
    evaluated = []
    for case in protocol.training_cases():
        result = evaluate_case(checked[case["run_id"]]["path"], project, runs, data,
                               args.device, args.workers, False, False,
                               args.latency_warmup, args.latency_repeats, dataset=test)
        evaluated.append({"case_id": result["case_id"], "weights_sha256": result["weights_sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "evaluation_index.json", {"study": protocol.STUDY, "new_cases_evaluated": evaluated,
              "legacy_teacher_evaluations_reused": ["teacher_seed0", "teacher_seed1"],
              "exploratory_extension": True})


if __name__ == "__main__":
    main()
