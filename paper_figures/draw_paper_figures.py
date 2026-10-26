"""Reproduce Chinese monochrome figures from the audited saved KD results.

All figures are authored at their final single-column width (3.2 in), use
7.5-point labels, and are exported as editable SVG plus 500-dpi PNG. This
script does not load a model, train, or evaluate official test images.
"""
from pathlib import Path
import csv
import json
import hashlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.titlesize": 7.5, "axes.unicode_minus": False,
    "axes.linewidth": .65, "lines.linewidth": .9,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
})


def load_json(relative):
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def load_csv(relative):
    with (DATA / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


summary = load_json("inputs/results_extended/summary.json")
audit = load_json("audited_results.json")
classes = load_json("classes_analysis.json")
runs = {r["case_id"]: r for r in load_csv("inputs/results_extended/runs_metrics.csv")}
manifest = []


def save(fig, stem, caption, note):
    png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
    fig.savefig(png, dpi=500, facecolor="white")
    fig.savefig(svg, facecolor="white")
    assert abs(fig.get_size_inches()[0] - 3.2) < 1e-10
    manifest.append({"stem": stem, "caption": caption, "note": note,
                     "width_inches": 3.2, "height_inches": float(fig.get_size_inches()[1]),
                     "label_font_pt": 7.5, "dpi": 500,
                     "png": png.name, "svg": svg.name,
                     "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                     "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest()})
    plt.close(fig)


def basic_axes(height=2.8, bottom=.19):
    fig, ax = plt.subplots(figsize=(3.2, height))
    fig.subplots_adjust(left=.21, right=.975, top=.965, bottom=bottom)
    ax.grid(axis="y", color=".88", linewidth=.4)
    ax.set_axisbelow(True)
    return fig, ax


def method_figure():
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    fig.subplots_adjust(left=.025, right=.975, top=.99, bottom=.025)
    ax.set(xlim=(0, 100), ylim=(-23, 145))
    ax.axis("off")
    layout = []

    def box(x, y, w, h, text):
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h,
                               fill=False, linewidth=.7, edgecolor="black"))
        # An explicit CJK default font avoids mathtext selecting TNR for the
        # surrounding Chinese; mathematical variables still use TNR italic.
        ax.text(x, y, text, ha="center", va="center", linespacing=1.4, fontfamily="SimSun")
        layout.append({"center": [x, y], "width": w, "height": h, "text": text})

    def arrow(points, dashed=False):
        for start, stop in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], stop[0]], [start[1], stop[1]], color="black",
                    linewidth=.65, linestyle=(0, (3, 2)) if dashed else "-")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=7, linewidth=.65, color="black",
                                     linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=.5))

    box(50, 132, 47, 12, "同一输入图像")
    box(24, 109, 36, 14, "教师网络\n参数固定")
    box(76, 109, 36, 14, "学生网络\n参数可更新")
    box(24, 86, 36, 12, "教师输出")
    box(76, 86, 36, 12, "学生输出")
    box(24, 63, 36, 15, "温度软化\n教师软目标")
    box(76, 63, 36, 15, "温度软化\n学生软输出")
    box(50, 40, 44, 14, "蒸馏损失\n" + r"$T^2$×相对熵")
    box(24, 16, 28, 12, "真实标签")
    box(76, 16, 36, 14, "标签损失\n交叉熵")
    box(50, -9, 75, 17, "加权总损失\n" + r"$(1-\alpha)$×标签损失＋$\alpha$×蒸馏损失")
    arrow([(50,126),(50,121),(24,121),(24,116)])
    arrow([(50,121),(76,121),(76,116)])
    arrow([(24,102),(24,92)])
    arrow([(76,102),(76,92)])
    arrow([(24,80),(24,70.5)])
    arrow([(76,80),(76,70.5)])
    ax.text(29,75,r"$T$",ha="left",va="center")
    ax.text(81,75,r"$T$",ha="left",va="center")
    arrow([(24,55.5),(24,40),(28,40)])
    arrow([(76,55.5),(76,40),(72,40)])
    arrow([(94,86),(96,86),(96,16),(94,16)])
    arrow([(38,16),(58,16)])
    arrow([(50,33),(50,29),(4,29),(4,-9),(12.5,-9)])
    arrow([(76,9),(76,-.5)])
    # The dashed feedback takes the outer right route, outside forward paths.
    arrow([(87.5,-9),(99,-9),(99,119),(85,119),(85,116)], dashed=True)
    ax.text(50,-21,"实线：前向计算　虚线：反向更新",ha="center",va="center")
    save(fig, "图1_知识蒸馏流程", "教师与学生的知识蒸馏流程",
         "原创示意图。教师固定为评价模式；温度软目标只用于蒸馏项；标签交叉熵使用学生原始输出。")
    (OUT / "方法图布局.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")


def validation_figure():
    fig, ax = basic_axes(2.8)
    for role, style, marker, label in [("teacher", "-", "o", "教师"),
                                         ("baseline", "--", "s", "普通学生"),
                                         ("kd", "-.", "^", "默认蒸馏学生")]:
        logs = [audit["cases"][f"{role}_seed{s}"]["log"] for s in (0, 1, 2)]
        vals = np.array([[r["val_acc"]*100 for r in log] for log in logs])
        x = np.arange(1, vals.shape[1] + 1)
        mean, sd = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        ax.fill_between(x, mean-sd, mean+sd, color=".85", alpha=.45, linewidth=0)
        ax.plot(x, mean, style, color="black", marker=marker, markersize=3.1,
                markerfacecolor="white", markeredgewidth=.6, markevery=2, label=label)
    ax.set(xlim=(1,20), xticks=[1,5,10,15,20], xlabel="训练轮次/次", ylabel="验证准确率/%")
    ax.legend(loc="lower right", frameon=True, edgecolor=".65", fancybox=False, borderpad=.4)
    save(fig, "图2_验证准确率曲线", "主实验的验证准确率变化",
         "曲线为3个随机种子均值，灰色区域为样本标准差。教师10轮，学生20轮。")


def ablation_figure(kind):
    table = summary["temperature" if kind == "temperature" else "alpha"]
    fig, ax = basic_axes(3.15, bottom=.29)
    x = np.array([r[kind] for r in table])
    for seed, marker, dash in [(0,"o",(0,(4,2))), (1,"^",(0,(1,2)))]:
        values=[]
        for row in table:
            if kind == "temperature":
                case=f"kd_seed{seed}" if row[kind] == 2 else f"kd_T{int(row[kind])}_seed{seed}"
            elif row[kind] == 0:
                case=f"baseline_seed{seed}"
            else:
                case=f"kd_seed{seed}" if row[kind] == .25 else f"kd_A{str(row[kind]).replace('.', 'p')}_seed{seed}"
            values.append(float(runs[case]["accuracy_percent"]))
        ax.plot(x, values, color=".45", linestyle=dash, marker=marker, markersize=3.4,
                markerfacecolor="white", markeredgewidth=.65, linewidth=.7, label=f"随机种子{seed}")
    ax.errorbar(x, [r["accuracy_percent_mean"] for r in table],
                yerr=[r["accuracy_percent_sample_std"] for r in table],
                fmt="-s", color="black", markersize=3.6, capsize=2.4, capthick=.6,
                linewidth=1, label="均值及样本标准差")
    ax.axhline(69.81, color=".45", linestyle="--", linewidth=.75, label="普通学生均值")
    ax.set(xticks=x, ylim=(69.1,74.0), ylabel="测试准确率/%",
           xlabel=r"温度 $T$" if kind=="temperature" else r"蒸馏项权重 $\alpha$")
    ax.xaxis.label.set_fontfamily("SimSun")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.24), ncol=2, frameon=False,
              columnspacing=.8, handlelength=1.7, handletextpad=.4, borderpad=0)
    if kind=="temperature":
        ax.set_xticklabels(["1","2","4","8"])
        save(fig,"图3_温度消融","不同温度下的测试准确率",
             "固定α=0.25；各设置采用随机种子0和1。误差线为样本标准差。")
    else:
        ax.set_xticklabels(["0","0.1","0.25","0.5","0.75"])
        save(fig,"图4_权重消融","不同蒸馏权重下的测试准确率",
             "固定T=2；α=0为普通学生参考，各设置采用随机种子0和1。误差线为样本标准差。")


def budget_figure():
    rows=load_csv("inputs/results_extended/learning_budget_comparison.csv")
    fig, ax=basic_axes(2.8)
    for role, style, marker, label in [("baseline","--","o","普通学生"),("kd","-","s","默认蒸馏学生")]:
        subset=[r for r in rows if r["role"]==role]
        before=[float(r["accuracy10_percent"]) for r in subset]
        after=[float(r["accuracy20_percent"]) for r in subset]
        for b,a in zip(before,after):
            ax.plot([10,20],[b,a],color=".7",linestyle=style,linewidth=.6)
        ax.errorbar([10,20],[statistics.mean(before),statistics.mean(after)],
                    yerr=[statistics.stdev(before),statistics.stdev(after)],
                    color="black",linestyle=style,marker=marker,markersize=4,
                    markerfacecolor="white",markeredgewidth=.7, capsize=3,
                    label=label)
    ax.set(xticks=[10,20],xlim=(8,22),ylim=(69.3,72.1),xlabel="训练预算/轮",ylabel="测试准确率/%")
    ax.legend(loc="upper left",frameon=False)
    save(fig,"图5_训练预算比较","不同训练预算下的测试准确率",
         "只比较随机种子0和1的同一连续训练轨迹；每个预算均按验证准确率选模。粗线与误差线为均值及样本标准差，细线为单次训练。")


def calibration_figure():
    fig, ax=basic_axes(2.8)
    for offset,role,hatch,label in [(-.18,"baseline","","普通学生"),(.18,"kd","///","默认蒸馏学生")]:
        values=[100*classes["calibration_by_run"][f"{role}_seed{s}"]["ece"] for s in (0,1,2)]
        ax.bar(np.arange(3)+offset,values,width=.33,facecolor="white",edgecolor="black",
               linewidth=.65,hatch=hatch,label=label)
    ax.set(xticks=[0,1,2],xticklabels=["0","1","2"],xlabel="随机种子",
           ylim=(0,17),yticks=[0,4,8,12,16],ylabel="期望校准误差/%")
    ax.legend(loc="upper left",frameon=False)
    save(fig,"图6_置信度校准诊断","普通学生与蒸馏学生的置信度校准诊断",
         "事后诊断采用15个等宽置信度分箱；期望校准误差越低越好。")


def class_figure():
    cn={"airplane":"飞机","automobile":"汽车","bird":"鸟","cat":"猫","deer":"鹿",
        "dog":"狗","frog":"青蛙","horse":"马","ship":"船","truck":"卡车"}
    entries=classes["classes"]
    fig,ax=basic_axes(3.2,bottom=.17)
    y=np.arange(10)
    means=[r["f1_gain_pp"]["mean"] for r in entries]
    sds=[r["f1_gain_pp"]["sample_sd"] for r in entries]
    ax.errorbar(means,y,xerr=sds,fmt="o",color="black",markerfacecolor="white",markersize=3.4,
                markeredgewidth=.65,linewidth=.7,capsize=2,capthick=.6)
    ax.axvline(0,color=".5",linestyle="--",linewidth=.65)
    ax.set(yticks=y,yticklabels=[cn[r["class_name"]] for r in entries],xlabel="类别综合指标增益/百分点")
    ax.invert_yaxis()
    ax.grid(axis="y",visible=False)
    ax.grid(axis="x",color=".88",linewidth=.4)
    save(fig,"图7_类别综合指标增益","各类别综合指标的配对变化",
         "综合指标指F1；点和误差线为3个种子下默认蒸馏相对普通学生增益的均值及样本标准差。")


if __name__=="__main__":
    method_figure()
    validation_figure()
    ablation_figure("temperature")
    ablation_figure("alpha")
    budget_figure()
    calibration_figure()
    class_figure()
    output={"source_snapshot":"data","read_only_analysis":True,
            "all_charts_chinese":True,"monochrome":True,
            "unit_label_style":"quantity/unit", "chinese_font":"SimSun",
            "latin_font":"Times New Roman", "mathematical_variables_italic":True,
            "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figures":manifest}
    (OUT/"figure_manifest.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":len(manifest),"output":str(OUT)},ensure_ascii=False))
