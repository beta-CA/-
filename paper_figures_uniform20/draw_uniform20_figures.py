"""Create readable, editable single-column figures from saved uniform-20 data.

No model is loaded, trained, or tested. Figure width is the paper's exact
8.125 cm column width. Statistical definitions belong to the paper text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


WIDTH_CM = 8.125
WIDTH_IN = WIDTH_CM / 2.54
DPI = 600
FONT_PT = 9.0
ROLES = ("teacher", "baseline", "kd")
LABELS = {"teacher": "教师", "baseline": "普通学生", "kd": "蒸馏学生"}
STYLE_REFERENCES = [
    {"title": "Distilling the Knowledge in a Neural Network", "url": "https://arxiv.org/pdf/1503.02531"},
    {"title": "Decoupled Knowledge Distillation", "url": "https://arxiv.org/pdf/2203.08679"},
    {"title": "Nature official figure preparation guide", "url": "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/"},
    {"title": "Cambridge official artwork guidelines", "url": "https://www.cambridge.org/core/services/aop-file-manager/file/5da6dc5e807d44780a804757"},
]
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "axes.unicode_minus": False,
    "axes.linewidth": .75,
    "lines.linewidth": 1.0,
    "xtick.major.width": .65,
    "ytick.major.width": .65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "savefig.facecolor": "white",
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def axis(height=2.47):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    fig.subplots_adjust(left=.205, right=.975, top=.965, bottom=.205)
    ax.tick_params(direction="out", pad=2.5)
    # Keep the visual field clear; no shaded bands or background gridlines.
    return fig, ax


def save_figure(fig, output, stem, number, definition, manifest):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check every drawn text item against the physical canvas before export.
    outside = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            rect = text.get_window_extent(renderer)
            if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > fig.bbox.width + 1 or rect.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside final figure canvas for {stem}: {outside}")
    png, svg = output / f"{stem}.png", output / f"{stem}.svg"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    manifest.append({"number": number, "stem": stem,
                     "width_cm": WIDTH_CM, "height_cm": float(fig.get_size_inches()[1] * 2.54),
                     "font_pt": FONT_PT, "dpi": DPI,
                     "png": png.name, "svg": svg.name,
                     "png_sha256": sha(png), "svg_sha256": sha(svg),
                     "statistical_definition_for_body_text": definition,
                     "has_figure_note": False, "legend_inside_canvas": True})
    plt.close(fig)


def draw_method(output, manifest):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.23))
    fig.subplots_adjust(left=.015, right=.985, bottom=.018, top=.99)
    ax.set(xlim=(0, 100), ylim=(-12, 103))
    ax.axis("off")

    def box(x, y, width, height, label):
        ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                               fill=False, edgecolor="black", linewidth=.75))
        ax.text(x, y, label, va="center", ha="center", linespacing=1.25,
                fontsize=FONT_PT, fontfamily="SimSun")

    def arrow(points, dashed=False):
        style = (0, (3, 2)) if dashed else "-"
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot((a[0], b[0]), (a[1], b[1]), color="black", linewidth=.75, linestyle=style)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=7.5,
                                     color="black", linewidth=.75, linestyle=style,
                                     shrinkA=0, shrinkB=.3))

    box(50, 96, 39, 9, "输入图像")
    box(25, 77, 37, 14, "教师网络\n参数固定")
    box(75, 77, 37, 14, "学生网络\n参数更新")
    box(25, 54, 37, 15, "温度软化\n教师概率")
    box(75, 54, 37, 15, "温度软化\n学生概率")
    box(50, 32, 45, 14, "蒸馏损失\n" + r"$T^2$×相对熵")
    box(22, 10, 30, 10, "真实标签")
    box(75, 10, 37, 13, "标签损失\n交叉熵")
    box(50, -6, 85, 10, r"总损失：$(1-\alpha)$×标签损失＋$\alpha$×蒸馏损失")
    arrow([(50,91.5),(50,88),(25,88),(25,84)])
    arrow([(50,88),(75,88),(75,84)])
    arrow([(25,70),(25,61.5)])
    arrow([(75,70),(75,61.5)])
    arrow([(25,46.5),(25,32),(27.5,32)])
    arrow([(75,46.5),(75,32),(72.5,32)])
    arrow([(93.5,77),(96,77),(96,10),(93.5,10)])
    arrow([(37,10),(56.5,10)])
    arrow([(50,25),(50,20),(3,20),(3,-6),(7.5,-6)])
    arrow([(75,3.5),(75,-1)])
    # Explicit label rather than a separate, small bottom legend.
    arrow([(92.5,-6),(99,-6),(99,88),(84,88),(84,84)], dashed=True)
    save_figure(fig, output, "图1_知识蒸馏流程", 1,
                "原创知识蒸馏流程；教师固定，学生同时学习温度软目标与真实标签，虚线为反向更新。", manifest)


def draw_learning(root, output, manifest, metric, number, stem):
    fig, ax = axis()
    specs = (("teacher", "-", "o", "black"),
             ("baseline", "--", "s", ".48"),
             ("kd", "-.", "^", "black"))
    for role, style, marker, colour in specs:
        histories = [rows(root / "runs_uniform20" / f"{role}_seed{seed}" / "metrics.csv")
                     for seed in (0, 1, 2)]
        values = np.asarray([[float(row[metric]) for row in history] for history in histories])
        if metric == "val_acc":
            values *= 100
        ax.plot(np.arange(1, 21), values.mean(axis=0), color=colour, linestyle=style,
                marker=marker, markersize=3.4, markerfacecolor="white", markeredgewidth=.7,
                markevery=[0,4,9,14,19], label=LABELS[role], linewidth=1.0)
    ax.set(xticks=[1,5,10,15,20], xlim=(.7,20.4), xlabel="训练轮次")
    if metric == "val_acc":
        ax.set(ylabel="验证准确率/%", ylim=(46,78), yticks=[50,60,70])
        ax.legend(loc="lower right", frameon=False, handlelength=2.1,
                  borderpad=.1, labelspacing=.45, handletextpad=.5)
    else:
        ax.set(ylabel="验证交叉熵", ylim=(.55,1.85), yticks=[.6,.9,1.2,1.5,1.8])
        ax.legend(loc="upper left", frameon=False, handlelength=2.1,
                  borderpad=.1, labelspacing=.45, handletextpad=.5)
    save_figure(fig, output, stem, number,
                "教师、普通学生及默认蒸馏学生均训练20轮；曲线为种子0、1、2的逐轮算术均值，无平滑处理。", manifest)


def draw_ablation(summary, output, manifest, kind):
    fig, ax = axis()
    entries = summary[kind]
    if kind == "alpha":
        entries = [entry for entry in entries if entry["alpha"] > 0]
    x = [entry[kind] for entry in entries]
    error = ax.errorbar(x, [entry["accuracy_percent_mean"] for entry in entries],
                       yerr=[entry["accuracy_percent_sample_std"] for entry in entries],
                       fmt="-o", color="black", markerfacecolor="white",
                       markersize=4.0, markeredgewidth=.8, linewidth=1.0,
                       capsize=3.2, capthick=.8, elinewidth=.8,
                       label="蒸馏学生", zorder=3)
    baseline = summary["alpha"][0]["accuracy_percent_mean"]
    reference = ax.axhline(baseline, color=".48", linestyle="--", linewidth=1.0,
                          label="普通学生", zorder=1)
    ax.set(xticks=x, ylim=(69.35,74.35), yticks=[70,71,72,73,74], ylabel="测试准确率/%")
    if kind == "temperature":
        ax.set(xlabel=r"温度 $T$", xlim=(.65,8.35))
        ax.set_xticklabels(["1", "2", "4", "8"])
        stem, number = "图3_温度消融", 3
        definition = "固定α=0.25，种子0和1，误差线为样本标准差；普通学生水平线为同两种子均值69.810%。"
    else:
        ax.set(xlabel=r"蒸馏权重 $\alpha$", xlim=(.055,.795))
        ax.set_xticklabels(["0.1", "0.25", "0.5", "0.75"])
        stem, number = "图4_权重消融", 4
        definition = "固定T=2，种子0和1，误差线为样本标准差；普通学生水平线为有效α=0参考，均值69.810%。"
    ax.xaxis.label.set_fontfamily("SimSun")
    ax.legend(handles=[error, reference], labels=["蒸馏学生", "普通学生"],
              loc="upper left", ncol=2, frameon=False, columnspacing=1.0,
              handlelength=1.8, handletextpad=.45, borderpad=.1)
    save_figure(fig, output, stem, number, definition, manifest)


def ece_from_predictions(path, bins=15):
    predictions = rows(path)
    counts = np.zeros(bins, dtype=np.int64)
    confidence_sum = np.zeros(bins, dtype=np.float64)
    correct_sum = np.zeros(bins, dtype=np.float64)
    for row in predictions:
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Invalid stored confidence")
        bucket = min(int(confidence * bins), bins - 1)
        counts[bucket] += 1
        confidence_sum[bucket] += confidence
        correct_sum[bucket] += int(row["true_class"] == row["predicted_class"])
    if len(predictions) != 10000:
        raise ValueError("Each ECE input must contain all 10000 official test predictions")
    return float(np.abs(correct_sum - confidence_sum).sum() / len(predictions))


def draw_calibration(root, output, manifest):
    fig, ax = axis()
    values_by_role = {}
    for offset, role, fill in ((-.18, "baseline", "white"), (.18, "kd", ".50")):
        values = [100 * ece_from_predictions(root / "runs_uniform20" / f"{role}_seed{s}" / "predictions.csv")
                  for s in (0,1,2)]
        values_by_role[role] = values
        ax.bar(np.arange(3) + offset, values, width=.32, facecolor=fill,
               edgecolor="black", linewidth=.75, label=LABELS[role], zorder=3)
    ax.set(xlim=(-.65,2.65), xticks=[0,1,2], xticklabels=["0","1","2"],
           xlabel="随机种子", ylim=(0,17.5), yticks=[0,4,8,12,16], ylabel="期望校准误差/%")
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderpad=.1,
              labelspacing=.45, handletextpad=.5)
    save_figure(fig, output, "图6_置信度校准诊断", 6,
                "事后补充诊断：15个等宽最高类别置信度分箱；每个柱来自对应种子的10000条最终预测，数值越小越好。", manifest)
    return values_by_role


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.project_root.resolve(strict=True), args.output_root.resolve()
    if output.is_relative_to(root / "runs_uniform20") or output.is_relative_to(root / "results_uniform20"):
        raise ValueError("Figure exports must not overwrite source experiment artifacts")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "results_uniform20" / "summary.json").read_text(encoding="utf-8"))
    if summary["study"] != "distilling_uniform20_v2" or summary["unique_cases"] != 21 or summary["epochs_each"] != 20:
        raise ValueError("Expected the complete, uniform-20 formal study")
    manifest = []
    draw_method(output, manifest)
    draw_learning(root, output, manifest, "val_acc", 2, "图2_验证准确率曲线")
    draw_ablation(summary, output, manifest, "temperature")
    draw_ablation(summary, output, manifest, "alpha")
    draw_learning(root, output, manifest, "val_loss", 5, "图5_验证交叉熵曲线")
    ece = draw_calibration(root, output, manifest)
    inputs = [root / "results_uniform20" / "summary.json"]
    inputs.extend(root / "runs_uniform20" / f"{role}_seed{s}" / "metrics.csv"
                  for role in ROLES for s in (0,1,2))
    inputs.extend(root / "runs_uniform20" / f"{role}_seed{s}" / "predictions.csv"
                  for role in ("baseline","kd") for s in (0,1,2))
    result = {"study": summary["study"], "figures": manifest, "font_pt_at_printed_width": FONT_PT,
              "printed_width_cm": WIDTH_CM, "dpi": DPI, "all_labels_chinese": True,
              "style": "monochrome with distinct line styles and point markers; uncluttered legends; no external notes",
              "inputs_sha256": {str(path.relative_to(root)): sha(path) for path in inputs},
              "source_project_root": ".", "read_only_analysis": True,
              "ece_percent_by_role": ece, "style_references": STYLE_REFERENCES,
              "source_script_sha256": sha(Path(__file__))}
    (output / "figure_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(manifest), "output": str(output), "font_pt": FONT_PT}, ensure_ascii=False))


if __name__ == "__main__":
    main()
