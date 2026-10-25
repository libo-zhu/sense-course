# Augsburg多传感器融合与模态缺失实验

多源传感器融合课程项目。公开文档采用匿名成员标签，个人署名与学号仅保留在本地课程提交材料。

本项目在 **Mac 24GB统一内存、MPS** 上完成 Augsburg HSI+SAR+DSM 的正式核心与扩展实验。包含单模态、输入拼接、概率平均、教师、JAD学生、MSH和随机模态丢弃扩展。**当前正式证据只有seed42，82条测试条件、18个训练阶段；完整数值及训练协议可核查。**

## 1. 安装环境

```bash
git clone https://github.com/libo-zhu/sense-course.git
cd sense-course
uv sync --locked
```

使用Python3.11、PyTorch2.5.1、NumPy1.26.4。uv会按锁文件创建`.venv`。Mac使用原生arm64运行环境。Linux/无MPS可显式选择`--device cpu`。代码不需要CUDA，也不会在明确指定MPS不可用时静默改用CPU。

## 2. 无需数据的离线小demo

```bash
uv run --locked python scripts/offline_demo.py --device auto
```

使用合成的三模态7×7输入，各阶段三次更新，验证教师、JAD、融合、冻结和单输入保存重载，结果在`outputs/offline-demo/report.json`。它是环境检查，**不输出数据集准确率**。

## 3. 获取公开数据

```bash
uv run --locked python scripts/prepare_data.py
```

默认准备Augsburg：下载作者三模态训练MAT、SAR/DSM小型测试文件及S2FL原始场景压缩包，再提取所需影像和标签。已有完整文件会复用。原始RAR约489MiB，MAT文件另需下载；Google Drive的配额/连接可能影响首次获取。Mac自带的bsdtar用于RAR解压，Linux需`libarchive-tools`提供bsdtar。数据来自第三方公开基准，遵循来源条件，**不把数据许可等同于本仓库MIT代码许可**。

来源：[S2FL数据](https://github.com/danfenghong/ISPRS_S2FL)、[MSH-Net作者代码及预处理数据入口](https://github.com/shicaiwei123/TGRS-MSHNet)。巨大HSI测试展开MAT不必下载，代码直接从原始场景裁剪等价邻域。

可选获取另一数据集：

```bash
uv run --locked python scripts/prepare_data.py --dataset all
```

该命令仅准备数据，不表示已经完成Houston正式实验。`check_data.py`检查两个数据集，运行它之前应使用`--dataset all`。

## 4. 真实数据小样本demo

```bash
uv run --locked python mac_demo.py --device mps --dataset augsburg --all-students --validation-source official
```

结果在`outputs/demo/`。默认只做极少更新和小测试子集，OA不是论文性能。正式实验不能用demo的数值代替。

## 5. 正式核心、扩展与断点恢复

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --preset extended
```

主要阶段500epoch、最终融合100epoch，训练608点、验证153点，最后评估全部77533个官方测试点。同一配置重复命令可恢复/复用已完成模型。中间epoch重跑，完整epoch的模型和优化器状态保存在`last.pt`。改变超参数应改`--output`，避免混用旧检查点。不要并行启动多组训练争用统一内存。

仅跑某个MSH，程序自动补齐依赖：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only msh-hsi
```

用极短训练验证正式入口：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --quick
```

quick输出到`outputs/experiments-quick`，不能写成最终性能。完整组合列表、消融ID、噪声条件和命令见[EXPERIMENTS.md](EXPERIMENTS.md)。

## 6. 无需重训，评估随附最佳模型

```bash
uv run --locked python infer_saved.py --model msh-hsi --device mps --limit 64
uv run --locked python infer_saved.py --model msh-hsi --device mps --limit 0
```

需要先准备真实数据。64点仅用于加载检查，`--limit 0`评估完整官方测试集。可替换为`msh-sar`、`msh-dsm`、`teacher`、`moddrop`、`single-hsi`等已有权重目录名。结果保存在`outputs/inference/`。随附的是`best.pt`和历史，不含恢复训练所需的`last.pt`；它们用于独立推理，不能用`--evaluate-only`冒充完整训练缓存。

## 7. 已完成的结果及核查

| 单输入 | 单模态OA | JAD OA | MSH OA | MSH AA |
|---|---:|---:|---:|---:|
| HSI | 89.29 | 89.38 | 89.59 | 66.25 |
| SAR | 81.18 | 78.67 | 80.48 | 43.07 |
| DSM | 74.69 | 74.24 | 75.41 | 39.98 |

完整教师OA91.69%，输入拼接91.54%，ModDrop91.91%。数据单位为%。随机模态丢弃在SAR+DSM条件下OA达到82.27%，较普通教师提高75.45个百分点；SAR学生在噪声标准差0.10时较单模态提高11.45个百分点。逐项结果、论文对照和协议说明见[详细分析](analysis/Augsburg全实验与论文对照分析.md)及技术报告。

`results/augsburg-seed42/`保存配置、82条件指标、混淆矩阵、18个best权重与训练历史。逐样本预测为`predictions.zip`，恢复后可以完整复算：

```bash
uv run --locked python scripts/unpack_results.py
uv run --locked python analyze_paper_comparison.py --run results/augsburg-seed42 --output outputs/paper-comparison
```

分析需要已准备的Augsburg原始标签；原论文PDF不重新分发，论文表VI的转录值和来源已保留。

`outputs/experiments/summary.csv`是重新训练后的总表，`summary-mean-std.csv`是多seed汇总。seed数不足2时不伪造标准差。逐条件目录有`metrics.json`、`confusion_matrix.csv`、`predictions.csv`；预测标签从0开始，混淆矩阵行是真实类、列是预测类。

## 8. 重要实验边界

- 原官方训练761点，本地608训练+153内部验证，未全量重训。所有方法共用划分。
- 保留作者整场景归一化，不是训练点独立估计范围。
- 153个验证patch均与训练patch邻域重叠，高验证OA不代表空间独立泛化。
- 正式入口当前没有论文的随机翻转增强；融合初始值/约束、选择规则与作者发布脚本不同。
- 当前只有seed42。作者构造器内部固定初始化，新增运行seed不自动等于独立随机初始化。
- ModDrop为借鉴已有思想的课程扩展，不声称原创算法。MSH/JAD来源清楚注明。

## 9. 检查与文档

```bash
uv run --locked python scripts/check_environment.py
uv run --locked python verify_experiments.py
```

验证指标计算、单样本尾batch、模型冻结、CPU epoch恢复一致性和检查点。`docs/`保存6页IEEE双栏报告、14页总结PPT、讲稿、要求对照表与来源说明。参考验证记录在`validation/`，属于工程证据，不是额外正式训练seed。

## 来源与公开材料

公开PDF、PPT、TeX源码和验证记录均不包含课程成员的姓名与学号。报告包含12条参考文献及正文引用，来源核查见[参考文献来源](docs/参考文献来源.md)。

上游：[MSH-Net](https://github.com/shicaiwei123/TGRS-MSHNet)，commit `c1b5f0409ca5e545b4f767070caab95fdd6fdc63`，MIT。保留`upstream/LICENSE`和`THIRD_PARTY_NOTICES.md`。代码采用MIT；数据、原论文等第三方内容遵守其各自条件。
