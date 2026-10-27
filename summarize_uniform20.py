"""Audit and summarize the separate, freshly trained uniform-20-epoch study.

Derived from summarize_extended.py; unchanged low-level integrity and descriptive
statistics helpers are reused. Historical runs and published results are not read
as new experimental observations. This script creates no result without all 21
formal training runs and their final evaluations.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

import numpy as np

import uniform_protocol as protocol
from evaluate import require, write_csv
from evaluate_uniform20 import expected_source, protect_outputs, validate_training_matrix
from kd_utils import file_sha256, load_json, save_json
from summarize import read_latency, read_prediction_artifacts, require_close
from summarize_extended import (METRICS, read_extended_log, describe, paired_row,
                                add_gain_statistics)

ROOT = Path(__file__).resolve().parent
ROLE_LABELS = {"teacher": "教师模型", "baseline": "普通学生", "kd": "蒸馏学生"}
CLASSES_ZH = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
COLORS = {"teacher": "#4F6D7A", "baseline": "#527FC3", "kd": "#DB8350"}


def audit_uniform_run(folder):
    from train_uniform20 import validate_run
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
    require(config.get("warm_start") is None, "Uniform-budget study must train every case from scratch")
    inherited = 0
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


def audit_matrix(project_root, runs_root):
    """Audit exactly the 21 current-study cases; reject missing or mixed results."""
    project_root, runs_root = Path(project_root).resolve(), Path(runs_root).resolve()
    validate_training_matrix(project_root, runs_root)
    expected = {x["run_id"] for x in protocol.all_cases()}
    evaluated = {x.parent.name for x in runs_root.glob("*/evaluation.json")}
    require(evaluated == expected,
            f"Expected exactly {len(expected)} evaluated current-study cases; "
            f"missing={sorted(expected-evaluated)}, unexpected={sorted(evaluated-expected)}")
    audited = {spec["run_id"]: audit_uniform_run(runs_root / spec["run_id"])
               for spec in protocol.all_cases()}
    items = list(audited.values())
    require(len({x["true_labels_sha256"] for x in items}) == 1, "Test labels/order differ across cases")
    require(all(x["evaluation"]["runtime"] == items[0]["evaluation"]["runtime"] for x in items),
            "Evaluation runtimes differ across cases")
    return {"cases": audited}


def make_tables(audited):
    cases = audited["cases"]
    rows = [cases[spec["run_id"]]["row"] for spec in protocol.all_cases()]
    main = [describe([cases[f"{role}_seed{seed}"] for seed in protocol.MAIN_SEEDS], ROLE_LABELS[role],
                     role=role, epochs=protocol.TRAINING_EPOCHS)
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
    alphas.append(add_gain_statistics(describe(base_items, "普通学生参考（蒸馏权重为0）",
                                              alpha=0.0, temperature=2.0, epochs=20), baseline_gains))
    for alpha in protocol.ALPHAS:
        items = [cases[protocol.alpha_case_id(alpha, s)] for s in protocol.ABLATION_SEEDS]
        gains = [paired_row(item, cases[f"baseline_seed{item['row']['seed']}"], "alpha", alpha) for item in items]
        paired.extend(gains)
        alphas.append(add_gain_statistics(describe(items, f"alpha={alpha:g}", alpha=alpha,
                                                   temperature=2.0, epochs=20), gains))
    return {"runs": rows, "main": main, "temperature": temperatures, "alpha": alphas,
            "paired": paired, "main_paired_summary": main_gain}


def save_plots(audited, tables, output):
    """Draw Chinese-language PNG and editable SVG figures from audited records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "SimSun", "WenQuanYi Zen Hei", "Arial Unicode MS")
    font = next((name for name in candidates if name in available), None)
    require(font is not None, "中文图表需要中文字体，请安装微软雅黑、思源黑体或 Noto Sans CJK SC")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "svg.fonttype": "none"})
    cases = audited["cases"]
    roles = ("teacher", "baseline", "kd")

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)

    labels = [f"{ROLE_LABELS[role]}\n20轮" for role in roles]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("accuracy_percent", "loss", "macro_f1"),
                                  ("测试准确率（%）", "测试交叉熵", "宏平均F1")):
        ax.bar(labels, [x[f"{metric}_mean"] for x in tables["main"]],
               yerr=[x[f"{metric}_sample_std"] for x in tables["main"]],
               color=[COLORS[role] for role in roles], capsize=4)
        ax.set(ylabel=ylabel, title="三个随机种子的均值与样本标准差")
    save(fig, "main_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for role in roles:
        for seed, style in zip(protocol.MAIN_SEEDS, ("-", "--", ":")):
            log = cases[f"{role}_seed{seed}"]["log"]
            axes[0].plot([r["epoch"] for r in log], [100*r["val_acc"] for r in log], style,
                         color=COLORS[role], label=f"{ROLE_LABELS[role]}，种子{seed}")
            axes[1].plot([r["epoch"] for r in log], [r["val_loss"] for r in log], style, color=COLORS[role])
    axes[0].set(xlabel="训练轮次", ylabel="验证准确率（%）", title="主实验验证准确率")
    axes[1].set(xlabel="训练轮次", ylabel="验证交叉熵", title="主实验验证损失")
    axes[0].legend(fontsize=8, ncol=2)
    save(fig, "main_validation_curves")

    for field, table_key, xlabel, name in (
            ("temperature", "temperature", "温度T（蒸馏权重α=0.25）", "temperature_ablation"),
            ("alpha", "alpha", "蒸馏权重α（温度T=2，α=0为普通学生）", "alpha_ablation")):
        entries = tables[table_key]
        x = [entry[field] for entry in entries]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        axes[0].errorbar(x, [e["accuracy_percent_mean"] for e in entries],
                         yerr=[e["accuracy_percent_sample_std"] for e in entries], marker="o", capsize=4)
        baseline = statistics.mean(cases[f"baseline_seed{s}"]["row"]["accuracy_percent"]
                                   for s in protocol.ABLATION_SEEDS)
        axes[0].axhline(baseline, linestyle="--", color="gray", label="同种子普通学生均值")
        axes[0].legend(fontsize=8)
        axes[1].errorbar(x, [e["accuracy_gain_pp_mean"] for e in entries],
                         yerr=[e["accuracy_gain_pp_sample_std"] for e in entries],
                         marker="o", capsize=4, color=COLORS["kd"])
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[0].set(xlabel=xlabel, ylabel="测试准确率（%）", title="两个种子的均值与样本标准差")
        axes[1].set(xlabel=xlabel, ylabel="相对普通学生的增益（百分点）", title="每次只改变一个因素")
        for ax in axes:
            ax.set_xticks(x)
        save(fig, name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, role in zip(axes, roles):
        matrices = []
        for seed in protocol.MAIN_SEEDS:
            matrix = cases[f"{role}_seed{seed}"]["confusion"].astype(float)
            matrices.append(matrix / matrix.sum(axis=1, keepdims=True))
        im = ax.imshow(np.mean(matrices, axis=0), vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"{ROLE_LABELS[role]}：三个种子均值", xlabel="预测类别", ylabel="真实类别")
        ax.set_xticks(range(10), CLASSES_ZH, rotation=90, fontsize=8)
        ax.set_yticks(range(10), CLASSES_ZH, fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), label="各真实类别中的比例", shrink=0.75)
    save(fig, "main_confusion_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, [x["parameter_bytes_fp32_mean"] / 1024**2 for x in tables["main"]],
                color=[COLORS[r] for r in roles])
    axes[0].set(ylabel="单精度参数存储（MiB）", title="仅统计部署模型参数")
    axes[1].bar(labels, [x["batch128_ms_median_mean"] for x in tables["main"]],
                yerr=[x["batch128_ms_median_sample_std"] for x in tables["main"]], capsize=4,
                color=[COLORS[r] for r in roles])
    axes[1].set(ylabel="批量128前向耗时（毫秒）", title="三个种子计时中位数的均值与样本标准差")
    save(fig, "model_size_latency")


def summarize(project_root, runs_root, output_root):
    project_root, runs_root, output_root = map(lambda p: Path(p).resolve(),
                                               (project_root, runs_root, output_root))
    protect_outputs(project_root, runs_root, output_root)
    audited = audit_matrix(project_root, runs_root)
    tables = make_tables(audited)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, key in (("runs_metrics", "runs"), ("main_summary", "main"),
                      ("temperature_summary", "temperature"), ("alpha_summary", "alpha"),
                      ("paired_gains", "paired")):
        rows = tables[key]
        write_csv(output_root / f"{name}.csv", rows, list(rows[0]))
    class_rows = [{"case_id": case_id, "role": item["config"]["role"],
                   "seed": item["config"]["seed"], **row}
                  for case_id, item in audited["cases"].items()
                  for row in item["evaluation"]["per_class"]]
    write_csv(output_root / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    notes = ["统一20轮的探索性重训；旧实验测试结果已经被查看过。",
             "教师、普通学生和蒸馏学生全部从头训练20轮，不读取旧实验作为新结果。",
             "所有21组训练完成后才进行本次最终测试；选模只使用验证集。",
             "主实验为三个种子，温度和权重消融各为两个相同种子。",
             "默认温度2、蒸馏权重0.25为共享参考，不作为额外独立重复。",
             "蒸馏权重0为已有普通学生参考，不重复训练数学上等价的配置。",
             "样本标准差仅描述种子间波动，不据此宣称统计显著性。",
             "前向耗时不包含数据读取、预处理或传输；单图计时不稳定，不能据此保证加速。",
             "未搜索温度与权重的组合，不宣称得到联合最优参数。"]
    source = {name: file_sha256(ROOT / name) for name in
              ("summarize_uniform20.py", "summarize_extended.py", "summarize.py")}
    total_epochs = sum(row["epochs"] for row in tables["runs"])
    require(total_epochs == len(protocol.all_cases()) * protocol.TRAINING_EPOCHS,
            "Training epoch total differs from current protocol")
    save_json(output_root / "summary.json", {
        "study": protocol.STUDY, "matrix_status": "complete", "unique_cases": len(tables["runs"]),
        "fresh_training_cases": len(tables["runs"]), "exploratory_extension": True,
        "epochs_each": protocol.TRAINING_EPOCHS,
        "main_seeds": list(protocol.MAIN_SEEDS), "ablation_seeds": list(protocol.ABLATION_SEEDS),
        "accuracy_unit": "percent", "paired_accuracy_difference_unit": "percentage_points",
        "main": tables["main"], "temperature": tables["temperature"], "alpha": tables["alpha"],
        "main_paired_summary": tables["main_paired_summary"],
        "total_study_training_validation_seconds": sum(r["training_seconds_total"] for r in tables["runs"]),
        "total_actual_training_epochs": total_epochs,
        "summary_sources_sha256": source, "notes": notes})
    files = []
    for case_id, item in audited["cases"].items():
        for name in ("best.pth", "last.pth", "config.json", "summary.json", "metrics.csv",
                     "evaluation.json", "predictions.csv", "confusion_matrix.csv"):
            path = item["folder"] / name
            relative = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else str(path)
            files.append({"kind": "uniform20_case", "case_id": case_id,
                          "path_relative_to_project": relative, "bytes": path.stat().st_size,
                          "sha256": file_sha256(path)})
    for path in sorted((runs_root / "splits").glob("*")):
        if path.is_file():
            relative = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else str(path)
            files.append({"kind": "split_index", "case_id": None,
                          "path_relative_to_project": relative, "bytes": path.stat().st_size,
                          "sha256": file_sha256(path)})
    save_json(output_root / "weights_manifest.json", {
        "study": protocol.STUDY, "artifacts": files,
        "weight_selection": "highest validation accuracy; earliest epoch wins ties",
        "split_sha256": tables["runs"][0]["split_sha256"],
        "note": "仅包含本次统一20轮实验的真实权重与结果；此清单不会自动上传文件。"})
    save_plots(audited, tables, output_root)
    print(f"已核对{len(tables['runs'])}组统一20轮真实实验。结果目录：{output_root}", flush=True)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    summarize(project, args.runs_root or project / "runs_uniform20",
              args.output_root or project / "results_uniform20")


if __name__ == "__main__":
    main()
