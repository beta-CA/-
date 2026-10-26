# 知识蒸馏温度与损失权重实验

这是课程论文的可复现实验代码与真实结果，研究CIFAR-10上输出知识蒸馏的温度、损失权重和训练预算。
论文中的方法是经典知识蒸馏的复现与参数分析，没有将单因素参数实验作为新算法。

- 代码仓库：https://github.com/beta-CA/-
- 代码与全部正式权重下载：[https://github.com/beta-CA/-/releases/tag/v1.0.0](https://github.com/beta-CA/-/releases/tag/v1.0.0)
- 发布版本：`v1.0.0`
- 权重文件：`distilling-weights-v1.0.0.zip`；代码快照：`distilling-code-v1.0.0.zip`
- 校验文件：Release资产中的 `SHA256SUMS.txt` 与 `PUBLISH_CONTENTS.json`

## 方法与开源来源

方法参考Hinton、Vinyals和Dean的论文 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)。
网络结构和蒸馏实现参考[PyTorch官方知识蒸馏教程](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)，
教程仓库固定提交为 `c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6`。
`vendor/pytorch_tutorial/` 保留原教程模型代码、参考源码、BSD-3-Clause许可证和来源哈希。
这不是Hinton等原作者发布的实验实现，也没有复现其全部数据集与实验。
其余数据划分、配对实验、检查点恢复、统一测试和结果汇总代码用于本课程实验。具体来源见 `NOTICE.md`。

## 环境

实测环境记录在 `environment.lock.txt`：Python 3.12.13，PyTorch 2.13.0+cu126，
torchvision 0.28.0+cu126，NumPy 2.4.4，Matplotlib 3.11.0，CUDA运行时12.6，
NVIDIA GeForce RTX 3060 Laptop GPU 6 GiB。
`requirements.txt` 给出基本依赖范围；请根据自己硬件和PyTorch官方安装说明选择匹配的CUDA版本。
正式实验入口固定使用CUDA、FP32、批量大小128和2个数据加载工作进程。
不同软件或硬件环境下，数值与运行时间可能变化。

## 从空输出目录重新训练

建议将仓库放在独立目录中，先保留仓库里的结果文件副本用于对照。
正式复现从没有 `runs/`、`runs_extended/` 的工作目录开始，不要把下载的历史权重解压到这个重训目录。
仓库不包含CIFAR-10数据集。入口会从官方来源准备数据并核验官方MD5，也可先执行数据准备：

```powershell
python prepare_data.py
```

首次完整运行按以下顺序执行，每一步等前一步完成：

```powershell
python run_all_experiments.py
python run_extended_experiments.py
```

在PyCharm中也可以依次右键这两个文件运行，不需要填写参数。
第一步训练种子0和1的教师、普通学生和默认蒸馏学生，共6组，每组10轮。
第二步先独立训练种子2教师，原学生继承第10轮的模型、Adam和随机状态继续到20轮，
再运行温度与损失权重消融；它会校验并跳过已完成组，并从已提交轮次恢复。
第二步依赖第一步的真实教师、学生与汇总结果，不能从空目录只运行第二步。
中断后重新运行相同入口。训练过程中不要修改训练代码、配置、环境和原有结果。
只预览任务时执行 `python run_all_experiments.py --dry-run` 或 `python run_extended_experiments.py --dry-run`。

主实验采用种子0、1、2；温度消融采用T=1、2、4、8，固定α=0.25；
权重消融采用α=0.1、0.25、0.5、0.75，固定T=2，两类消融均使用种子0和1。
α是KL蒸馏项权重，真实标签交叉熵权重为1−α，KL包含T²缩放。
默认T=2、α=0.25的模型跨表复用，没有重复计为独立实验。
教师预算10轮，学生预算20轮，均按验证准确率最高选模、并列取较早轮。
所有新增训练完成后统一测试。配置详见 `configs/experiments.json` 和 `configs/extended_experiments.json`。

## 正式结果与权重

`results/` 保存最初6组10轮实验的汇总；`results_extended/` 保存补充实验的正式表格、图像与文件哈希清单。
权重归档保留以下相对路径：

```text
runs/                        # 原6组正式实验
runs/splits/                 # 原训练/验证划分索引
runs_extended/               # 19组新增或续训正式实验
runs_extended/splits/        # 补充实验使用的同一划分索引
results/                     # 原实验结果表与图
results_extended/            # 补充实验结果表与图
```

每个正式组包含 `best.pth`、`last.pth`、`config.json`、`metrics.csv`、`summary.json`、
`evaluation.json`、`predictions.csv` 和 `confusion_matrix.csv`。
`best.pth` 是验证集选出的模型，`last.pth` 另外保存优化器与随机状态供原环境断点恢复。
最终21个不同模型由3个教师、3个普通学生和15个蒸馏模型组成；归档另外保留4个原10轮学生作为训练预算对照，
所以归档共有25个正式训练目录，而不是25个最终独立条件。

| 主实验模型 | 3种子测试准确率（均值±样本标准差） | 参数量 |
|---|---:|---:|
| 教师 | 75.530% ± 0.060% | 1,186,986 |
| 普通学生 | 69.757% ± 0.144% | 267,738 |
| 默认蒸馏学生 | 71.250% ± 0.406% | 267,738 |

默认蒸馏配对提升1.493个百分点，三个种子均为正。T=8和α=0.5分别在各自2种子单因素实验中取得最高观察均值，
二者并未作为联合配置测试。默认蒸馏平均测试交叉熵有所恶化，不能宣称所有指标全面改善。
日志累计训练及每轮验证约181.34分钟；不含中断等待、最终测试和绘图的完整墙钟时间。
所有结果来自实际保存的正式实验，合成验证夹具与临时初始化备份不在发布资产中。
补充前已查看原测试结果，本研究属于探索性扩展；种子数量较少，不报告普遍性或显著性结论。

## 历史路径与在新机器上使用权重

为保留真实来源与文件哈希，正式检查点及其 `config.json`、`evaluation.json` 中的历史绝对路径保持原样。
这些路径是原训练位置的元数据，不是新机器需要创建的目录，也不包含登录口令或访问令牌。
原严格续训和审计入口会核对路径、环境、源码及依赖哈希，因此下载权重后直接在另一台机器原地续训或完整审计可能被拒绝。
重新训练请按上面的空输出目录流程进行，生成新机器自己的路径与来源记录。
核对权重文件可使用Release的SHA-256清单；推理时只加载模型状态，不需要恢复历史路径：

```python
import torch
from models import build_model

# 仅对本项目已校验的可信检查点使用weights_only=False。
checkpoint = torch.load("runs_extended/kd_seed0/best.pth",
                        map_location="cpu", weights_only=False)
model = build_model(checkpoint["config"]["role"])
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
```

输入为3×32×32图像，像素转为[0,1]后按均值[0.485,0.456,0.406]与标准差[0.229,0.224,0.225]归一化。
完整评估需要对应数据与划分。发布权重包含Python/NumPy随机状态，只应从本仓库核验后的可信来源加载。

## 发布内容和排除项

代码包不包含数据集、本机IDE配置、`configs/local_paths.json`、运行报告中的私有环境记录、缓存或备份。
源目录中的原训练/模型/评估/汇总源码及vendor来源文件逐字节保留，避免改变已记录的来源哈希。
公开副本仅将 `main.py` 的本机解释器提示改为通用提示，并移除 `requirements.txt` 首行的本机路径注释；
README和gitignore为发布用途重新编写。这些变化不修改训练算法或已发布检查点。
`NOTICE.md`和vendor许可证保留上游归属；若再分发上游代码，应同时保留这些文件。
