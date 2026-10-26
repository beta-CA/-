"""Audit all six formal runs before producing tables, figures and weight hashes.

All variation statistics describe only the two predeclared training seeds. No
missing, incomplete, synthetic, stale or differently configured run is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np

from evaluate import (MEAN, ROLES, SEEDS, STD, formal_config, per_class_metrics,
                      require, validate_training, write_csv)
from kd_utils import file_sha256, load_json, save_json


CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
METRICS = ("accuracy_percent", "loss", "macro_f1", "parameter_count", "parameter_bytes_fp32",
           "checkpoint_bytes", "training_seconds_total", "batch1_ms_median", "batch128_ms_median")
LOG_FIELDS = ["epoch", "train_total_loss", "train_ce_loss", "train_kd_loss", "train_acc",
              "val_loss", "val_acc", "lr", "seconds"]
LABELS = {"teacher": "Teacher", "baseline": "Student (CE)", "kd": "Student (CE + KD)"}
COLORS = {"teacher": "#4F6D7A", "baseline": "#527FC3", "kd": "#DB8350"}


def require_close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-8),
            f"Inconsistent/nonfinite metric: {label}")
    return float(actual)


def read_training_log(path, checkpoint):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == LOG_FIELDS, f"Unexpected training-log columns: {path}")
        rows = []
        for expected, raw in enumerate(reader, 1):
            require(None not in raw and all(value is not None for value in raw.values()),
                    f"Malformed training-log row: {path}")
            row = {key: int(value) if key == "epoch" else float(value) for key, value in raw.items()}
            require(row["epoch"] == expected and all(math.isfinite(v) for v in row.values()),
                    f"Nonsequential/nonfinite training log: {path}")
            require(0 <= row["train_acc"] <= 1 and 0 <= row["val_acc"] <= 1
                    and row["train_total_loss"] >= 0 and row["train_ce_loss"] >= 0
                    and row["train_kd_loss"] >= -1e-6 and row["val_loss"] >= 0
                    and row["seconds"] > 0 and row["lr"] == 0.001,
                    f"Invalid training-log metrics: {path}")
            rows.append(row)
    require(len(rows) == 10, f"Formal run must contain ten complete epochs: {path}")
    selected = max(rows, key=lambda row: row["val_acc"])
    require(selected["epoch"] == checkpoint["epoch"]
            and selected["val_acc"] == checkpoint["best_val_acc"],
            f"Validation selection does not match training log: {path}")
    require(selected == checkpoint["last_metrics"], f"Selected-epoch metrics differ from checkpoint: {path}")
    return rows


def read_prediction_artifacts(folder, evaluation):
    confusion_path, predictions_path = folder / "confusion_matrix.csv", folder / "predictions.csv"
    require(evaluation.get("confusion_matrix_file") == confusion_path.name
            and evaluation.get("predictions_file") == predictions_path.name, "Unexpected evaluation artifact paths")
    require(file_sha256(confusion_path) == evaluation.get("confusion_matrix_sha256"),
            f"Confusion matrix checksum differs: {folder}")
    require(file_sha256(predictions_path) == evaluation.get("predictions_sha256"),
            f"Prediction checksum differs: {folder}")
    with confusion_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["true_class", *map(str, range(10))], "Wrong confusion-matrix schema")
        raw = list(reader)
    require([row["true_class"] for row in raw] == CLASSES, "Confusion matrix has wrong class order")
    confusion = np.asarray([[int(row[str(index)]) for index in range(10)] for row in raw], dtype=np.int64)
    require(confusion.shape == (10, 10) and bool((confusion >= 0).all())
            and bool((confusion.sum(axis=1) == 1000).all()), "Confusion matrix does not describe CIFAR-10 test")
    rebuilt = np.zeros((10, 10), dtype=np.int64)
    true_labels = []
    with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["sample_index", "true_class", "predicted_class", "confidence"],
                "Wrong prediction schema")
        for expected, row in enumerate(reader):
            index, actual, predicted = int(row["sample_index"]), int(row["true_class"]), int(row["predicted_class"])
            confidence = float(row["confidence"])
            require(index == expected and 0 <= actual < 10 and 0 <= predicted < 10
                    and math.isfinite(confidence) and 0 <= confidence <= 1,
                    "Invalid prediction row")
            rebuilt[actual, predicted] += 1
            true_labels.append(actual)
    require(len(true_labels) == 10000 and np.array_equal(rebuilt, confusion),
            "Predictions and confusion matrix disagree")
    correct = int(np.trace(confusion))
    require(type(evaluation.get("correct")) is int and evaluation["correct"] == correct
            and evaluation.get("total") == 10000, "Evaluation accuracy counts disagree")
    require_close(evaluation.get("accuracy_percent"), 100 * correct / 10000, "test accuracy")
    expected_classes = per_class_metrics(confusion, CLASSES)
    require(evaluation.get("per_class_metric_unit") == "fraction"
            and evaluation.get("per_class") == expected_classes, "Per-class metrics disagree with confusion matrix")
    require_close(evaluation.get("macro_f1"), float(np.mean([row["f1"] for row in expected_classes])), "macro F1")
    return confusion, hashlib.sha256(bytes(true_labels)).hexdigest()


def read_latency(evaluation):
    latency = evaluation.get("latency", {})
    require(latency.get("scope") == "model_only_device_resident"
            and latency.get("unit") == "milliseconds_per_batch"
            and latency.get("warmup") == 20 and latency.get("repeats") == 50,
            "Wrong inference-timing protocol")
    for batch in (1, 128):
        values = latency.get(f"batch{batch}_ms_samples")
        require(isinstance(values, list) and len(values) == 50
                and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                "Missing or invalid raw latency samples")
        require_close(latency.get(f"batch{batch}_ms_median"), float(np.median(values)), "latency median")
        require_close(latency.get(f"batch{batch}_ms_p95"), float(np.percentile(values, 95)), "latency p95")
    return latency


def audit_run(folder):
    checkpoint, summary, weights_hash = validate_training(folder / "best.pth")
    config = checkpoint["config"]
    evaluation = load_json(folder / "evaluation.json")
    require(evaluation.get("schema_version") == 1 and evaluation.get("formal") is True
            and evaluation.get("partition") == "official_test"
            and evaluation.get("accuracy_unit") == "percent", f"Nonformal/wrong evaluation: {folder}")
    require(evaluation.get("run_id") == folder.name and evaluation.get("role") == config["role"]
            and evaluation.get("seed") == config["seed"] and evaluation.get("config") == config,
            f"Evaluation identity/config differs: {folder}")
    require(evaluation.get("checkpoint") == "best.pth"
            and evaluation.get("weights_sha256") == weights_hash
            and evaluation.get("model_state_sha256") == checkpoint["model_state_sha256"]
            and evaluation.get("selected_epoch") == checkpoint["epoch"]
            and evaluation.get("training_completed_epochs") == 10,
            f"Evaluation checkpoint metadata differs: {folder}")
    require_close(evaluation.get("best_val_acc"), checkpoint["best_val_acc"], "validation accuracy")
    require(evaluation.get("evaluation_source_sha256") == file_sha256(Path(__file__).parent / "evaluate.py"),
            "Evaluation source changed after these metrics were generated")
    runtime = evaluation.get("runtime", {})
    require(runtime.get("device") in ("cuda", "cuda:0") and runtime.get("batch_size") == 128
            and runtime.get("workers") == 2 and runtime.get("cpu_threads") == 4,
            "Evaluation runtime differs from frozen protocol")
    require(type(evaluation.get("loss")) in (int, float) and math.isfinite(evaluation["loss"])
            and evaluation["loss"] >= 0, "Invalid cross-entropy loss")
    parameters = sum(t.numel() for t in checkpoint["model_state"].values())
    # The official CNNs have no BatchNorm running-stat buffers, so state entries
    # are precisely their trainable floating-point parameters.
    require(type(evaluation.get("parameter_count")) is int
            and evaluation["parameter_count"] == parameters
            and evaluation.get("parameter_bytes_fp32") == parameters * 4,
            "Parameter count/FP32 bytes differ from model tensors")
    require(evaluation.get("checkpoint_bytes") == (folder / "best.pth").stat().st_size,
            "Checkpoint file size differs")
    confusion, labels_hash = read_prediction_artifacts(folder, evaluation)
    latency = read_latency(evaluation)
    log = read_training_log(folder / "metrics.csv", checkpoint)
    row = {"run_id": folder.name, "role": config["role"], "seed": config["seed"],
           "selected_epoch": checkpoint["epoch"], "best_val_accuracy_percent": 100 * checkpoint["best_val_acc"],
           "accuracy_percent": evaluation["accuracy_percent"], "loss": evaluation["loss"],
           "macro_f1": evaluation["macro_f1"], "parameter_count": parameters,
           "parameter_bytes_fp32": parameters * 4, "checkpoint_bytes": evaluation["checkpoint_bytes"],
           "training_seconds_total": sum(item["seconds"] for item in log),
           "batch1_ms_median": latency["batch1_ms_median"],
           "batch128_ms_median": latency["batch128_ms_median"],
           "init_sha256": config["init_sha256"], "split_sha256": config["split_sha256"],
           "weights_sha256": weights_hash, "model_state_sha256": checkpoint["model_state_sha256"]}
    return {"row": row, "config": config, "evaluation": evaluation, "log": log,
            "confusion": confusion, "true_labels_sha256": labels_hash}


def audit_matrix(runs_root):
    runs_root = Path(runs_root).resolve(strict=True)
    expected_names = {f"{role}_seed{seed}" for role in ROLES for seed in SEEDS}
    present_evaluations = {path.parent.name for path in runs_root.glob("*/evaluation.json")}
    require(present_evaluations == expected_names,
            f"Expected exactly six evaluated formal runs; missing={sorted(expected_names-present_evaluations)}, "
            f"unexpected={sorted(present_evaluations-expected_names)}")
    audited = {(role, seed): audit_run(runs_root / f"{role}_seed{seed}")
               for seed in SEEDS for role in ROLES}
    configs = [item["config"] for item in audited.values()]
    require(len({config["split_sha256"] for config in configs}) == 1,
            "Training/validation splits differ across runs")
    require(len({item["true_labels_sha256"] for item in audited.values()}) == 1,
            "Official test sample order/labels differ across runs")
    first = configs[0]
    for config in configs:
        for field in ("dataset", "epochs", "batch_size", "workers", "device", "precision", "optimizer",
                      "temperature", "alpha", "augmentation", "normalize_mean", "normalize_std", "source", "software"):
            require(config.get(field) == first.get(field), f"Training protocol differs: {field}")
    runtimes = [item["evaluation"]["runtime"] for item in audited.values()]
    require(all(runtime == runtimes[0] for runtime in runtimes), "Evaluation hardware/software runtimes differ")
    for seed in SEEDS:
        teacher, baseline, kd = [audited[(role, seed)] for role in ROLES]
        require(baseline["row"]["init_sha256"] == kd["row"]["init_sha256"],
                f"Student initialization is not paired: seed{seed}")
        require(baseline["row"]["parameter_count"] == kd["row"]["parameter_count"],
                "Baseline and KD students do not have the same deployed architecture")
        dependency = kd["config"].get("teacher", {})
        require(dependency.get("seed") == seed
                and dependency.get("weights_sha256") == teacher["row"]["weights_sha256"]
                and dependency.get("model_state_sha256") == teacher["row"]["model_state_sha256"],
                f"KD dependency is not the matching completed teacher: seed{seed}")
        require(Path(dependency.get("checkpoint", "")).resolve() ==
                (runs_root / f"teacher_seed{seed}" / "best.pth").resolve(),
                "KD teacher path differs from this experiment matrix")
        require(teacher["config"].get("teacher") is None and baseline["config"].get("teacher") is None,
                "Teacher/baseline unexpectedly uses a teacher dependency")
    return audited


def make_tables(audited):
    rows = [audited[(role, seed)]["row"] for role in ROLES for seed in SEEDS]
    groups = []
    for role in ROLES:
        group = {"role": role, "n_seeds": 2, "seeds": "0;1"}
        for metric in METRICS:
            values = [audited[(role, seed)]["row"][metric] for seed in SEEDS]
            group[f"{metric}_mean"] = statistics.mean(values)
            group[f"{metric}_sample_std"] = statistics.stdev(values)
        groups.append(group)
    paired = []
    for seed in SEEDS:
        teacher, baseline, kd = [audited[(role, seed)]["row"] for role in ROLES]
        paired.append({"seed": seed,
                       "kd_minus_baseline_accuracy_pp": kd["accuracy_percent"] - baseline["accuracy_percent"],
                       "kd_minus_baseline_loss": kd["loss"] - baseline["loss"],
                       "kd_minus_baseline_macro_f1": kd["macro_f1"] - baseline["macro_f1"],
                       "teacher_minus_baseline_accuracy_pp": teacher["accuracy_percent"] - baseline["accuracy_percent"],
                       "teacher_minus_kd_accuracy_pp": teacher["accuracy_percent"] - kd["accuracy_percent"],
                       "student_parameter_reduction_vs_teacher_percent":
                           100 * (1 - kd["parameter_count"] / teacher["parameter_count"]),
                       "teacher_over_kd_batch1_latency_ratio":
                           teacher["batch1_ms_median"] / kd["batch1_ms_median"],
                       "kd_extra_training_seconds_vs_baseline":
                           kd["training_seconds_total"] - baseline["training_seconds_total"]})
    paired_summary = {"n_seeds": 2, "seeds": [0, 1]}
    for metric in paired[0]:
        if metric != "seed":
            values = [row[metric] for row in paired]
            paired_summary[f"{metric}_mean"] = statistics.mean(values)
            paired_summary[f"{metric}_sample_std"] = statistics.stdev(values)
    return rows, groups, paired, paired_summary


def save_plots(audited, groups, output_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [LABELS[role] for role in ROLES]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(labels, [row["accuracy_percent_mean"] for row in groups],
           yerr=[row["accuracy_percent_sample_std"] for row in groups],
           color=[COLORS[role] for role in ROLES], capsize=5)
    ax.set(ylabel="Official test accuracy (%)", title="Mean and sample SD over two training seeds")
    fig.savefig(output_root / "accuracy_comparison.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for role in ROLES:
        for seed in SEEDS:
            log = audited[(role, seed)]["log"]
            style = "-" if seed == 0 else "--"
            axes[0].plot([row["epoch"] for row in log], [100 * row["val_acc"] for row in log],
                         style, color=COLORS[role], label=f"{LABELS[role]}, seed {seed}")
            axes[1].plot([row["epoch"] for row in log], [row["val_loss"] for row in log],
                         style, color=COLORS[role])
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].set(xlabel="Epoch", ylabel="Validation cross-entropy")
    axes[0].legend(fontsize=8)
    fig.savefig(output_root / "validation_curves.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for role in ROLES:
        for seed in SEEDS:
            log = audited[(role, seed)]["log"]
            style = "-" if seed == 0 else "--"
            axes[0].plot([row["epoch"] for row in log], [row["train_ce_loss"] for row in log],
                         style, color=COLORS[role], label=f"{LABELS[role]}, seed {seed}")
            if role == "kd":
                axes[1].plot([row["epoch"] for row in log], [row["train_kd_loss"] for row in log],
                             style, color=COLORS[role], label=f"KD, seed {seed}")
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy (CE component)")
    axes[1].set(xlabel="Epoch", ylabel="Training KL distillation component (T squared included)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output_root / "training_loss_curves.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for column, role in enumerate(ROLES):
        for row, seed in enumerate(SEEDS):
            ax = axes[row, column]
            normalized = audited[(role, seed)]["confusion"] / 1000.0
            im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            ax.set(title=f"{LABELS[role]}, seed {seed}", xlabel="Predicted class", ylabel="True class")
            ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)
            ax.set_yticks(range(10), CLASSES, fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Fraction within true class", shrink=0.8)
    fig.savefig(output_root / "confusion_matrices.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(labels, [row["parameter_bytes_fp32_mean"] / 1024 ** 2 for row in groups],
                color=[COLORS[role] for role in ROLES])
    axes[0].set(ylabel="FP32 parameter storage (MiB)", title="Deployed model parameters only")
    axes[1].bar(labels, [row["batch1_ms_median_mean"] for row in groups],
                yerr=[row["batch1_ms_median_sample_std"] for row in groups],
                color=[COLORS[role] for role in ROLES], capsize=5)
    axes[1].set(ylabel="Batch-1 forward latency (ms)", title="Device-resident model only; mean seed medians")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(output_root / "model_size_latency.png", dpi=200)
    plt.close(fig)


def summarize(runs_root, output_root):
    runs_root, output_root = Path(runs_root).resolve(), Path(output_root).resolve()
    # Every validation precedes output creation, so an incomplete matrix cannot
    # produce a plausible-looking main table or overwrite an earlier valid one.
    audited = audit_matrix(runs_root)
    rows, groups, paired, paired_summary = make_tables(audited)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "runs_metrics.csv", rows, list(rows[0]))
    write_csv(output_root / "group_summary.csv", groups, list(groups[0]))
    write_csv(output_root / "paired_gains.csv", paired, list(paired[0]))
    class_rows = [{"role": role, "seed": seed, **row}
                  for role in ROLES for seed in SEEDS
                  for row in audited[(role, seed)]["evaluation"]["per_class"]]
    write_csv(output_root / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    notes = ["All results are from the complete six-run CIFAR-10 matrix.",
             "Means and sample SD describe two training seeds; no significance claim is implied.",
             "Paired-gain SD is calculated from within-seed differences, not from independent group SDs.",
             "CE-only and KD students share architecture and initialization within each seed.",
             "Latency is device-resident forward time in milliseconds per batch, excluding preprocessing/copies.",
             "Teacher/student parameter storage differs; KD does not reduce the baseline student's parameter count.",
             "Training CE is accumulated while weights change and dropout is enabled; it is not clean-train accuracy."]
    save_json(output_root / "summary.json", {"schema_version": 1, "matrix_status": "complete",
              "seeds": list(SEEDS), "roles": list(ROLES), "accuracy_unit": "percent",
              "paired_accuracy_difference_unit": "percentage_points", "groups": groups,
              "paired_gains_summary": paired_summary, "notes": notes})
    artifacts = []
    for role in ROLES:
        for seed in SEEDS:
            folder = runs_root / f"{role}_seed{seed}"
            for name in ("best.pth", "config.json", "summary.json", "metrics.csv", "evaluation.json",
                         "confusion_matrix.csv", "predictions.csv"):
                path = folder / name
                artifacts.append({"role": role, "seed": seed,
                                  "path_relative_to_runs": path.relative_to(runs_root).as_posix(),
                                  "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    save_json(output_root / "weights_manifest.json", {"schema_version": 1,
              "weight_selection": "highest validation accuracy, earliest epoch on ties",
              "split_sha256": rows[0]["split_sha256"], "artifacts": artifacts,
              "note": "Upload all six best.pth plus code/config/logs; this manifest does not publish files."})
    save_plots(audited, groups, output_root)
    print(f"Audited all six complete runs. Tables, figures and weight hashes saved to {output_root}", flush=True)
    return {"runs": rows, "groups": groups, "paired": paired, "paired_summary": paired_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    summarize(args.runs_root, args.output_root)


if __name__ == "__main__":
    main()
