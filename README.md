# 知识蒸馏温度与损失权重实验

课程论文在 CIFAR-10 上比较教师、普通学生和蒸馏学生。正式实验包含 21 组从头训练的模型，每组训练 20 轮，按验证准确率选择最佳权重。主实验使用 3 个种子，温度和损失权重各用 2 个种子做单因素比较。[实验结果](RESULTS_UNIFORM20.md) · [实验方案](UNIFORM20_EXPERIMENT_PLAN.md)

- 当前代码在本仓库的 `main` 分支。正式实验表、图和逐组统计在 [`results_uniform20/`](results_uniform20/)。
- 正式权重归档 `distilling-weights-v2.0.0.zip` 包含 21 组模型的 `best.pth`、`last.pth`、配置、训练记录和测试结果。发布页还提供代码快照、`SHA256SUMS.txt` 和逐文件清单。[v2.0.0 发布页](https://github.com/beta-CA/-/releases/tag/v2.0.0)

## 环境与训练

安装适配本机 CUDA 的 PyTorch 和 torchvision，再安装 [`requirements.txt`](requirements.txt) 中的其余依赖。正式实验使用 CUDA、FP32、批量大小 128 和 2 个数据加载进程。实测版本与设备见 [`environment.lock.txt`](environment.lock.txt)。

从仓库根目录检查环境并预览 21 组任务。

```powershell
python main.py
python run_uniform20_experiments.py --dry-run
```

重新训练使用独立副本。把副本中的已发布 `results_uniform20/` 改名留作对照，并为新训练留出空的 `runs_uniform20/` 和 `results_uniform20/`。入口会准备 CIFAR-10 并核验官方 MD5，然后依次训练、测试和汇总。也可先单独运行 `python prepare_data.py` 准备数据。

```powershell
python run_uniform20_experiments.py
```

PyCharm 可直接运行 `run_uniform20_experiments.py`。原有 `run_extended_experiments.py` 入口指向同一套 20 轮流程。中断后重新执行同一入口可从最近完整轮次恢复，已完成且通过校验的组会跳过。`runs_uniform20/` 保存训练记录和权重，`results_uniform20/` 保存测试汇总，`reports/` 保存运行进度和控制台日志。

## 使用已发布权重

下载 v2.0.0 权重归档，对照发布页的 `SHA256SUMS.txt` 校验后解压。下面以种子 0 的蒸馏学生为例，从解压目录加载验证集选出的 `best.pth`。

```python
import torch

from kd_utils import build_transform
from models import build_model

checkpoint = torch.load(
    "runs_uniform20/kd_seed0/best.pth", map_location="cpu", weights_only=True
)
model = build_model(checkpoint["config"]["role"])
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
preprocess = build_transform()
```

`preprocess` 将 32×32 的 RGB 图像转为张量，并按正式实验的通道均值和标准差归一化。正式恢复与审计会检查训练路径和来源哈希。在不同机器上可加载已发布权重做推理，重新训练时从空输出目录生成本机记录。

重新下载 v2.0.0 权重后，在 CIFAR-10 的 10,000 张正式测试图像上对全部 21 组 `best.pth` 做了独立复验。共 210,000 条类别预测均与发布记录一致，逐组准确率、交叉熵和宏平均 F1 均复现到论文报告精度。[正式实验结果](RESULTS_UNIFORM20.md)

## 绘图

[`paper_figures_uniform20/`](paper_figures_uniform20/) 提供统一 20 轮实验的中文 PNG、可编辑 SVG 和绘图脚本。将权重归档中的 `runs_uniform20/` 解压到仓库根目录后，可根据日志和结果表重绘。

```powershell
python paper_figures_uniform20/draw_uniform20_figures.py --output-root ./redrawn_figures
```

图源及统计定义见 [绘图说明](paper_figures_uniform20/FIGURE_STYLE_AND_DATA.md)。

## 历史实验与来源

`results/`、`results_extended/` 和 `paper_figures/` 是早期 10 轮与续训实验记录。历史实验的入口、结果和权重见 [v1.0.0 发布页](https://github.com/beta-CA/-/releases/tag/v1.0.0) 及 [当时的复现说明](https://github.com/beta-CA/-/blob/v1.0.0/README.md)。

蒸馏方法参考 [Hinton、Vinyals 和 Dean 的原论文](https://arxiv.org/abs/1503.02531)。模型和实现参考 [PyTorch 知识蒸馏教程](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)，原教程的模型、源码、许可证和固定提交信息位于 [`vendor/pytorch_tutorial/`](vendor/pytorch_tutorial/)。[代码来源说明](NOTICE.md)
