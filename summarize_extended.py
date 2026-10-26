"""Strict tables/figures for the predeclared exploratory three-hour extension.

Main results have three paired seeds. Each one-factor ablation has two paired
seeds; the shared default is reused, never treated as an extra independent run.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import numpy as np

import extended_protocol as protocol
from evaluate import require, write_csv
from evaluate_extended import expected_source, validate_training_matrix
from kd_utils import CSV_FIELDS, file_sha256, load_json, save_json
from summarize import (CLASSES, audit_run as audit_legacy_run, read_latency,
                       read_prediction_artifacts, require_close)

ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy_percent", "loss", "macro_f1", "parameter_count", "parameter_bytes_fp32",
           "checkpoint_bytes", "training_seconds_total", "new_training_seconds",
           "batch1_ms_median", "batch128_ms_median")
ROLE_LABELS = {"teacher": "Teacher (10 epochs)", "baseline": "Student CE (20 epochs)",
               "kd": "Student CE + KD (20 epochs)"}
COLORS = {"teacher": "#4F6D7A", "baseline": "#527FC3", "kd": "#DB8350"}


def read_extended_log(path, best, last):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_FIELDS, f"Unexpected log fields: {path}")
        rows = []
        for expected, raw in enumerate(reader, 1):
            require(None not in raw and all(v is not None for v in raw.values()), "Malformed training row")
            row = {key: int(value) if key == "epoch" else float(value) for key, value in raw.items()}
            require(row["epoch"] == expected and all(math.isfinite(x) for x in row.values()),
                    "Nonfinite/nonsequential training history")
            require(0 <= row["train_acc"] <= 1 and 0 <= row["val_acc"] <= 1
                    and row["train_ce_loss"] >= 0 and row["val_loss"] >= 0
                    and row["train_total_loss"] >= -1e-6 and row["train_kd_loss"] >= -1e-6
                    and row["lr"] == 0.001 and row["seconds"] > 0, "Invalid training metrics")
            rows.append(row)
    require(len(rows) == best["config"]["epochs"] == last["epoch"], "Incomplete training log")
    selected = max(rows, key=lambda x: x["val_acc"])
    require(selected["epoch"] == best["epoch"] and selected == best["last_metrics"],
            "Validation-best checkpoint differs from training log")
    require(rows[-1] == last["last_metrics"], "Final checkpoint metrics differ from final log row")
    return rows


def audit_extended_run(folder):
    from train_extended import validate_run
    folder = Path(folder)
    best, last, summary, digest = validate_run(folder / "best.pth")
    config = best["config"]
    evaluation = load_json(folder / "evaluation.json")
    require(evaluation.get("schema_version") == 1 and evaluation.get("study") == protocol.STUDY
            and evaluation.get("formal") is True and evaluation.get("partition") == "official_test"
            and evaluation.get("exploratory_extension") is True,
            f"Nonformal or wrongly identified evaluation: {folder}")
    require(evaluation.get("run_id") == evaluation.get("case_id") == config["case_id"] == folder.name
            and evaluation.get("config") == config and evaluation.get("role") == config["role"]
            and evaluation.get("seed") == config["seed"] and evaluation.get("group") == config["group"],
            "Evaluation identity/config mismatch")
    require(evaluation.get("checkpoint") == "best.pth" and evaluation.get("weights_sha256") == digest
            and evaluation.get("model_state_sha256") == best["model_state_sha256"]
            and evaluation.get("selected_epoch") == best["epoch"]
            and evaluation.get("training_completed_epochs") == config["epochs"]
            and evaluation.get("evaluation_source") == expected_source(), "Stale evaluation provenance")
    require(evaluation.get("accuracy_unit") == "percent", "Accuracy must be a percentage")
    require_close(evaluation.get("best_val_acc"), best["best_val_acc"], "validation accuracy")
    require(type(evaluation.get("loss")) in (int, float) and math.isfinite(evaluation["loss"])
            and evaluation["loss"] >= 0, "Invalid test cross-entropy")
    runtime = evaluation.get("runtime", {})
    require(runtime.get("device") in ("cuda", "cuda:0") and runtime.get("workers") == 2
            and runtime.get("batch_size") == 128 and runtime.get("cpu_threads") == 4,
            "Evaluation runtime differs from frozen protocol")
    params = sum(t.numel() for t in best["model_state"].values())
    require(evaluation.get("parameter_count") == params and evaluation.get("parameter_bytes_fp32") == params * 4
            and evaluation.get("checkpoint_bytes") == (folder / "best.pth").stat().st_size,
            "Parameter storage/checkpoint size differs")
    confusion, labels_hash = read_prediction_artifacts(folder, evaluation)
    latency = read_latency(evaluation)
    log = read_extended_log(folder / "metrics.csv", best, last)
    inherited = (config.get("warm_start") or {}).get("completed_epochs", 0)
    require(type(inherited) is int and 0 <= inherited < config["epochs"], "Invalid inherited epoch count")
    row = {"case_id": folder.name, "role": config["role"], "seed": config["seed"],
           "group": config["group"], "epochs": config["epochs"],
           "temperature": config["temperature"], "alpha": config["alpha"],
           "selected_epoch": best["epoch"], "best_val_accuracy_percent": 100 * best["best_val_acc"],
           "accuracy_percent": evaluation["accuracy_percent"], "loss": evaluation["loss"],
           "macro_f1": evaluation["macro_f1"], "parameter_count": params,
           "parameter_bytes_fp32": params * 4, "checkpoint_bytes": evaluation["checkpoint_bytes"],
           "training_seconds_total": sum(item["seconds"] for item in log),
           "inherited_epochs": inherited,
           "new_training_seconds": sum(item["seconds"] for item in log if item["epoch"] > inherited),
           "new_training_epochs": len(log) - inherited,
           "batch1_ms_median": latency["batch1_ms_median"], "batch128_ms_median": latency["batch128_ms_median"],
           "init_sha256": config["init_sha256"], "split_sha256": config["split_sha256"],
           "weights_sha256": digest, "model_state_sha256": best["model_state_sha256"], "legacy_reused": False}
    return {"row": row, "config": config, "evaluation": evaluation, "log": log,
            "confusion": confusion, "true_labels_sha256": labels_hash, "folder": folder}


def legacy_teacher_entry(folder):
    item = audit_legacy_run(folder)
    row = item["row"]
    item["row"] = {"case_id": row["run_id"], "role": "teacher", "seed": row["seed"], "group": "main",
                   "epochs": 10, "temperature": 2.0, "alpha": 0.25,
                   **{key: row[key] for key in ("selected_epoch", "best_val_accuracy_percent", *METRICS)
                      if key in row},
                   "inherited_epochs": 10, "new_training_epochs": 0, "new_training_seconds": 0.0,
                   "init_sha256": row["init_sha256"], "split_sha256": row["split_sha256"],
                   "weights_sha256": row["weights_sha256"], "model_state_sha256": row["model_state_sha256"],
                   "legacy_reused": True}
    item["folder"] = Path(folder)
    return item


def audit_matrix(project_root, runs_root):
    """Require the complete extension plus unmodified original reference results."""
    project_root, runs_root = Path(project_root).resolve(), Path(runs_root).resolve()
    validate_training_matrix(project_root, runs_root)
    expected = {x["run_id"] for x in protocol.training_cases()}
    evaluated = {x.parent.name for x in runs_root.glob("*/evaluation.json")}
    require(evaluated == expected, f"Expected exactly19 evaluated extended cases; missing={sorted(expected-evaluated)}, "
            f"unexpected={sorted(evaluated-expected)}")
    audited = {}
    for spec in protocol.all_cases():
        case_id = spec["run_id"]
        audited[case_id] = (legacy_teacher_entry(project_root / "runs" / case_id)
                            if spec["reuse_legacy_teacher"] else audit_extended_run(runs_root / case_id))
    items = list(audited.values())
    require(len({x["true_labels_sha256"] for x in items}) == 1, "Test labels/order differ across cases")
    require(all(x["evaluation"]["runtime"] == items[0]["evaluation"]["runtime"] for x in items),
            "Evaluation runtimes differ across referenced and extended cases")
    refs = {}
    for seed in (0, 1):
        for role in ("baseline", "kd"):
            case_id = f"{role}_seed{seed}"
            refs[case_id] = audit_legacy_run(project_root / "runs" / case_id)
            current = audited[case_id]
            old = refs[case_id]
            warm = current["config"].get("warm_start") or {}
            require(warm.get("completed_epochs") == 10
                    and Path(warm.get("checkpoint", "")).resolve() == (project_root / "runs" / case_id / "last.pth").resolve(),
                    f"Main20 run did not continue its declared original10 checkpoint: {case_id}")
            require(current["config"]["init_sha256"] == old["config"]["init_sha256"]
                    and current["config"]["split_sha256"] == old["config"]["split_sha256"]
                    and current["log"][:10] == old["log"], "Inherited initialization/split/log history changed")
            if role == "kd":
                require(current["config"]["teacher"] == old["config"]["teacher"],
                        "Learning-budget comparison changed the teacher")
            old["folder"] = project_root / "runs" / case_id
    return {"cases": audited, "legacy10_refs": refs}


def describe(items, label, **extra):
    rows = [item["row"] for item in items]
    seeds = sorted(row["seed"] for row in rows)
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds), "Descriptive groups require distinct paired seeds")
    result = {"setting": label, "n_seeds": len(rows), "seeds": ";".join(map(str, seeds)), **extra}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_sample_std"] = statistics.stdev(values)
    return result


def paired_row(item, baseline, comparison, value):
    row, base = item["row"], baseline["row"]
    require(row["seed"] == base["seed"] and row["epochs"] == base["epochs"] == 20
            and row["init_sha256"] == base["init_sha256"], "Ablation reference is not a same-budget paired student")
    return {"comparison": comparison, "value": value, "seed": row["seed"], "case_id": row["case_id"],
            "baseline_case_id": base["case_id"], "accuracy_gain_pp": row["accuracy_percent"] - base["accuracy_percent"],
            "loss_difference": row["loss"] - base["loss"], "macro_f1_difference": row["macro_f1"] - base["macro_f1"],
            "training_seconds_difference": row["training_seconds_total"] - base["training_seconds_total"]}


def add_gain_statistics(group, paired):
    for metric in ("accuracy_gain_pp", "loss_difference", "macro_f1_difference", "training_seconds_difference"):
        values = [row[metric] for row in paired]
        group[f"{metric}_mean"] = statistics.mean(values)
        group[f"{metric}_sample_std"] = statistics.stdev(values)
    return group


def make_tables(audited):
    cases, refs = audited["cases"], audited["legacy10_refs"]
    rows = [cases[spec["run_id"]]["row"] for spec in protocol.all_cases()]
    main = [describe([cases[f"{role}_seed{seed}"] for seed in protocol.MAIN_SEEDS], role,
                     role=role, epochs=10 if role == "teacher" else 20)
            for role in ("teacher", "baseline", "kd")]
    paired = [paired_row(cases[f"kd_seed{s}"], cases[f"baseline_seed{s}"], "main", "T2_alpha0.25")
              for s in protocol.MAIN_SEEDS]
    main_gain = add_gain_statistics({"comparison": "main", "n_seeds": 3, "seeds": "0;1;2"}, paired)
    temperatures, alphas = [], []
    for temperature in protocol.TEMPERATURES:
        items = [cases[protocol.temperature_case_id(temperature, s)] for s in protocol.ABLATION_SEEDS]
        gains = [paired_row(item, cases[f"baseline_seed{item['row']['seed']}"], "temperature", temperature)
                 for item in items]
        paired.extend(gains)
        temperatures.append(add_gain_statistics(describe(items, f"T={temperature:g}", temperature=temperature,
                                                          alpha=0.25, epochs=20), gains))
    # Alpha=0 is the existing CE-only control, not an extra training run.
    base_items = [cases[f"baseline_seed{s}"] for s in protocol.ABLATION_SEEDS]
    baseline_gains = [paired_row(item, item, "alpha", 0.0) for item in base_items]
    paired.extend(baseline_gains)
    alphas.append(add_gain_statistics(describe(base_items, "CE-only reference (effective alpha=0)",
                                              alpha=0.0, temperature=2.0, epochs=20), baseline_gains))
    for alpha in protocol.ALPHAS:
        items = [cases[protocol.alpha_case_id(alpha, s)] for s in protocol.ABLATION_SEEDS]
        gains = [paired_row(item, cases[f"baseline_seed{item['row']['seed']}"], "alpha", alpha) for item in items]
        paired.extend(gains)
        alphas.append(add_gain_statistics(describe(items, f"alpha={alpha:g}", alpha=alpha,
                                                   temperature=2.0, epochs=20), gains))
    budget = []
    for seed in (0, 1):
        for role in ("baseline", "kd"):
            case_id = f"{role}_seed{seed}"
            before, after = refs[case_id]["row"], cases[case_id]["row"]
            budget.append({"role": role, "seed": seed, "source10": case_id, "source20": case_id,
                           "accuracy10_percent": before["accuracy_percent"], "accuracy20_percent": after["accuracy_percent"],
                           "accuracy20_minus10_pp": after["accuracy_percent"]-before["accuracy_percent"],
                           "loss10": before["loss"], "loss20": after["loss"],
                           "selected_epoch10": before["selected_epoch"], "selected_epoch20": after["selected_epoch"],
                           "added_training_seconds": after["new_training_seconds"], "inherited_first10_epochs": True})
    return {"runs": rows, "main": main, "temperature": temperatures, "alpha": alphas,
            "paired": paired, "main_paired_summary": main_gain, "learning_budget": budget}


def save_plots(audited, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = audited["cases"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar([ROLE_LABELS[r] for r in ("teacher", "baseline", "kd")],
           [x["accuracy_percent_mean"] for x in tables["main"]],
           yerr=[x["accuracy_percent_sample_std"] for x in tables["main"]], capsize=5,
           color=[COLORS[r] for r in ("teacher", "baseline", "kd")])
    ax.set(ylabel="Official test accuracy (%)", title="Main comparison: three independent paired seeds")
    fig.savefig(output / "main_accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for role in ("teacher", "baseline", "kd"):
        for seed, style in zip(protocol.MAIN_SEEDS, ("-", "--", ":")):
            log = cases[f"{role}_seed{seed}"]["log"]
            axes[0].plot([r["epoch"] for r in log], [100*r["val_acc"] for r in log], style,
                         color=COLORS[role], label=f"{role}, seed{seed}")
            axes[1].plot([r["epoch"] for r in log], [r["val_loss"] for r in log], style, color=COLORS[role])
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].set(xlabel="Epoch", ylabel="Validation cross-entropy")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.axvline(10, color="gray", alpha=0.5, linewidth=1)
    fig.savefig(output / "main_validation_curves.png", dpi=180)
    plt.close(fig)
    for field, table_key, xlabel, filename in (("temperature", "temperature", "Temperature T (alpha=0.25)", "temperature_ablation.png"),
                                               ("alpha", "alpha", "KD weight alpha (T=2; alpha=0 is CE-only)", "alpha_ablation.png")):
        entries = tables[table_key]
        x = [entry[field] for entry in entries]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        axes[0].errorbar(x, [e["accuracy_percent_mean"] for e in entries],
                         yerr=[e["accuracy_percent_sample_std"] for e in entries], marker="o", capsize=4)
        baseline = statistics.mean(cases[f"baseline_seed{s}"]["row"]["accuracy_percent"] for s in (0, 1))
        axes[0].axhline(baseline, linestyle="--", color="gray", label="Paired CE-only mean")
        axes[0].legend(fontsize=8)
        axes[1].errorbar(x, [e["accuracy_gain_pp_mean"] for e in entries],
                         yerr=[e["accuracy_gain_pp_sample_std"] for e in entries], marker="o", capsize=4, color=COLORS["kd"])
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[0].set(xlabel=xlabel, ylabel="Test accuracy (%)", title="Two seeds, mean and sample SD")
        axes[1].set(xlabel=xlabel, ylabel="Paired gain over CE-only (percentage points)", title="One factor varied; no joint search")
        for ax in axes:
            ax.set_xticks(x)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for ax, role in zip(axes, ("baseline", "kd")):
        for row in [x for x in tables["learning_budget"] if x["role"] == role]:
            ax.plot([10, 20], [row["accuracy10_percent"], row["accuracy20_percent"]], marker="o", label=f"seed{row['seed']}")
        ax.set(xticks=[10, 20], xlabel="Training budget (epochs)", ylabel="Selected-model test accuracy (%)",
               title=f"{role}: same trajectory continued; two seeds")
        ax.legend()
    fig.savefig(output / "learning_budget_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, role in zip(axes, ("teacher", "baseline", "kd")):
        confusion = sum(cases[f"{role}_seed{s}"]["confusion"] for s in protocol.MAIN_SEEDS) / 3000.0
        im = ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"{role}: mean of three seeds", xlabel="Predicted class", ylabel="True class")
        ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)
        ax.set_yticks(range(10), CLASSES, fontsize=7)
    fig.colorbar(im, ax=axes.tolist(), label="Fraction within true class", shrink=0.75)
    fig.savefig(output / "main_confusion_matrices.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    labels = ["Teacher", "Student CE", "Student CE + KD"]
    axes[0].bar(labels, [x["parameter_bytes_fp32_mean"] / 1024**2 for x in tables["main"]],
                color=[COLORS[r] for r in ("teacher", "baseline", "kd")])
    axes[0].set(ylabel="FP32 parameter storage (MiB)", title="Deployed parameters only")
    axes[1].bar(labels, [x["batch128_ms_median_mean"] for x in tables["main"]],
                yerr=[x["batch128_ms_median_sample_std"] for x in tables["main"]], capsize=4,
                color=[COLORS[r] for r in ("teacher", "baseline", "kd")])
    axes[1].set(ylabel="Batch128 forward latency (ms)", title="Device-resident model; three seed medians")
    fig.savefig(output / "model_size_latency.png", dpi=180)
    plt.close(fig)


def summarize(project_root, runs_root, output_root):
    project_root, runs_root, output_root = map(lambda p: Path(p).resolve(), (project_root, runs_root, output_root))
    audited = audit_matrix(project_root, runs_root)
    tables = make_tables(audited)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, key in (("runs_metrics", "runs"), ("main_summary", "main"), ("temperature_summary", "temperature"),
                      ("alpha_summary", "alpha"), ("paired_gains", "paired"), ("learning_budget_comparison", "learning_budget")):
        rows = tables[key]
        write_csv(output_root / f"{name}.csv", rows, list(rows[0]))
    class_rows = [{"case_id": case_id, "role": item["config"]["role"], "seed": item["config"]["seed"], **row}
                  for case_id, item in audited["cases"].items() for row in item["evaluation"]["per_class"]]
    write_csv(output_root / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    notes = ["Exploratory extension: original test results had already been inspected.",
             "All cases and one-factor sweeps were fixed before extension testing; no test-based recipe selection.",
             "Main comparison uses three seeds; each ablation and10-vs20 comparison uses only seeds0/1.",
             "Default T2/alpha0.25 weights are shared references, not independent extra repetitions.",
             "Alpha0 is the existing CE-only baseline; its stored unused alpha field does not enter its loss.",
             "Teacher budget is10 epochs and student budget20; only student ablations share the20-epoch budget.",
             "Sample SD is descriptive; epochs, cases, and test images are not extra independent training seeds.",
             "Inherited10-epoch histories are counted once in total-study time, and excluded from new-work time.",
             "Latency is model-only device-resident forward time; batch1 has known measurement instability.",
             "No joint temperature-by-alpha search, universal improvement, or optimum is established."]
    save_json(output_root / "summary.json", {"study": protocol.STUDY, "matrix_status": "complete",
              "unique_cases": 21, "new_or_continued_training_cases": 19, "exploratory_extension": True,
              "main_seeds": [0, 1, 2], "ablation_seeds": [0, 1], "accuracy_unit": "percent",
              "paired_accuracy_difference_unit": "percentage_points",
              "main": tables["main"], "temperature": tables["temperature"], "alpha": tables["alpha"],
              "main_paired_summary": tables["main_paired_summary"], "learning_budget": tables["learning_budget"],
              "total_study_training_validation_seconds": sum(r["training_seconds_total"] for r in tables["runs"]),
              "new_training_validation_seconds": sum(r["new_training_seconds"] for r in tables["runs"]),
              "total_actual_training_epochs": 390, "new_training_epochs": 330,
              "summary_source_sha256": file_sha256(Path(__file__)), "notes": notes})
    files = []
    seen = set()
    for kind, items in (("extension_case", audited["cases"]), ("original10_student_reference", audited["legacy10_refs"])):
        for case_id, item in items.items():
            for name in ("best.pth", "last.pth", "config.json", "summary.json", "metrics.csv", "evaluation.json",
                         "predictions.csv", "confusion_matrix.csv"):
                path = item["folder"] / name
                if path in seen:
                    continue
                seen.add(path)
                try:
                    relative = path.relative_to(project_root).as_posix()
                except ValueError:
                    relative = str(path)
                files.append({"kind": kind, "case_id": case_id, "path_relative_to_project": relative,
                              "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    save_json(output_root / "weights_manifest.json", {"study": protocol.STUDY, "artifacts": files,
              "weight_selection": "highest validation accuracy; earliest epoch wins ties",
              "split_sha256": tables["runs"][0]["split_sha256"],
              "note": "Includes immutable legacy dependencies. This manifest does not publish files."})
    save_plots(audited, tables, output_root)
    print(f"Audited21 unique cases and19 new/continued runs. Extended results: {output_root}", flush=True)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    summarize(project, args.runs_root or project / "runs_extended", args.output_root or project / "results_extended")


if __name__ == "__main__":
    main()
