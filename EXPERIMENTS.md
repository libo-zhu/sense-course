# 多模态实验清单与运行手册

以下均在 `mac-reproduction` 目录执行，使用已配置好的 uv 环境和 Mac MPS。

```bash
cd sense-course
```

入口是 `run_experiments.py`。它按完整 epoch 训练，和只运行几个 batch 的 `mac_demo.py` 分开。**本文中的 quick 结果只验证流程；正式结果需要去掉 `--quick`。**

## 1. 数据与四组模态组合

| 组合 ID（`--scenario`） | 数据集 | 教师训练输入 | 学生/最终 MSH 的独立实验 | 建议 |
|---|---|---|---|---|
| `aug-hs-sar-dsm` | Augsburg | HSI＋SAR＋DSM | 只保留 HSI；只保留 SAR；只保留 DSM | 优先，三种数据模态，训练集较小 |
| `aug-hs-sar` | Augsburg | HSI＋SAR | 只保留 HSI；只保留 SAR | 看增加 DSM 是否改善学生 |
| `hou-hs-lidar` | Houston2013 | HSI＋LiDAR | 只保留 HSI；只保留 LiDAR | 第二数据集验证 |
| `hou-hs-ms` | Houston2013 | HSI＋MS | 只保留 HSI；只保留 MS | 论文另一组实验；MS 是由 HSI 降光谱分辨率得到 |

HSI 是高光谱，SAR 是合成孔径雷达，DSM 是数字表面模型（来自高度测量），LiDAR 是激光雷达高度数据，MS 是多光谱。

**一组组合中的每个 retained modality 都训练一个独立学生和一个独立 MSH。** 例如 `msh-hsi` 始终只输入 HSI，部署时没有教师。三模态 MSH 实验是“保留一个、缺失两个”；没有在这里实现“保留两个”的 MSH 学生。三模态教师/简单融合/ModDrop 则会评估全部七个非空输入子集。

## 2. 每组必做实验（core）

下面以三模态 Augsburg 为例；二模态组会自动删掉不适用的行。

| ID（`--only`） | 训练时输入/监督 | 测试时输入 | 回答的问题 |
|---|---|---|---|
| `single-hsi` | HSI＋CE | HSI | 单传感器基线 |
| `single-sar` | SAR＋CE | SAR | 单传感器基线 |
| `single-dsm` | DSM＋CE | DSM | 单传感器基线 |
| `early-concat` | 三模态按通道拼接后进入单分支网络＋CE | 完整模态，以及每种缺失组合 | 简单输入级融合基线 |
| `late-mean` | 复用三个已训练单模态模型，不额外训练 | 三模型输出类别概率等权平均；同时测试零填充缺失输入 | 简单决策级融合基线 |
| `teacher` | 作者的多分支特征融合网络＋CE | 完整模态，以及每种缺失组合 | 完整模态参考，以及直接缺失的退化程度 |
| `jad-hsi` / `jad-sar` / `jad-dsm` | 学生仅看指定模态；冻结教师看全部模态；CE＋MMD＋KL | 仅指定模态 | 不加最终共享/特有融合时的蒸馏效果 |
| `msh-hsi` / `msh-sar` / `msh-dsm` | 复用 JAD 学生和对应单模态基线；冻结特征与 BN，只训练分类头和 η | 两个分支均只看指定模态 | 最终 MSH-Net 效果 |

核心对比：在**相同测试输入**下比较 `single-X`、`jad-X`、`msh-X`，以及 `teacher` 的 `keep-X`。完整模态教师是信息更充分的参考，不保证一定是最高准确率。

三模态情况下，`early-concat`、`late-mean`、`teacher` 分别测试：

- 完整：HSI＋SAR＋DSM。
- 缺失一个：HSI＋SAR、HSI＋DSM、SAR＋DSM。
- 缺失两个：仅 HSI、仅 SAR、仅 DSM。

缺失模态用**归一化输入上的全零张量**代替。这是固定模型遭遇缺失的基线，不是用剩余模态重新训练的模型。`late-mean` 在缺失时仍保留全部分支，缺失分支读零输入；没有改成只平均可用分支。

一组三模态 core 会训练 11 个模型阶段，生成 30 行测试条件；二模态 core 训练 8 个阶段，生成 15 行。`late-mean` 不额外训练。

## 3. 可选消融、噪声与作业改进（extended）

`--preset extended` 包括全部 core，再增加：

| ID | 具体改变 | 合理的对照 |
|---|---|---|
| `kl-only-X` | 学生 CE＋βKL，令 α＝0 | 对比 `jad-X`，检查特征分布蒸馏作用 |
| `mmd-only-X` | 学生 CE＋αMMD，令 β＝0 | 对比 `jad-X`，检查输出蒸馏作用 |
| `msh-fixed-eta-at-test-X` | 复用已训练 MSH，测试时把 η 强制设为 0.5 | 对比 `msh-X`，只是测试时权重干预，不是从头固定 η 训练 |
| `moddrop` | 同教师结构；每个训练样本以 50% 概率随机丢掉一个完整模态；其余样本完整输入 | 对比 `teacher` 在相同缺失组合下的鲁棒性 |

`X` 替换为当前组的模态名，例如 `kl-only-hsi`、`mmd-only-sar`。ModDrop 是适合大作业的小扩展/基线，不能宣称为新的学术创新；是否改善结果须看实际实验。

扩展模式还会对 `single-X`、`jad-X`、`msh-X` 和 `msh-fixed-eta-at-test-X` 评估高斯噪声标准差 **0、0.05、0.1、0.2**。噪声加在已归一化的可用输入上，随后截断到 [0,1]；这是测试时噪声，训练不加噪声。不对 KL-only/MMD-only 学生额外扫描噪声。相同输入形状/顺序使用相同固定随机噪声。

三模态 extended 共 18 个训练阶段、82 行测试条件；二模态为 13 个阶段、48 行。可复用同目录中已经跑完的 core。

## 4. 建议的运行顺序

### A. 先检查新实验入口（约几个 batch，不能写成最终结果）

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --quick
```

默认保存 `outputs/experiments-quick/`。quick 每阶段 1 epoch，64 个训练样本、32 个验证样本、64 个官方测试样本。不要拿它的 OA 判断方法好坏。

### B. 先正式跑完 Augsburg 三模态，单个随机种子

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps
```

默认每个教师/单模态/学生阶段 500 epoch，最终融合阶段 100 epoch，batch size 32，seed 42。默认保存 `outputs/experiments/`。首次运行会训练所有依赖；重复同一命令会复用完成的模型，并重新评估测试集。

### C. 分别跑论文另外三组（每条可以独立执行）

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar --device mps
uv run --locked python run_experiments.py --scenario hou-hs-lidar --device mps
uv run --locked python run_experiments.py --scenario hou-hs-ms --device mps
```

四组一次顺序运行：

```bash
uv run --locked python run_experiments.py --scenario all --device mps
```

### D. 三个随机种子，报告均值与标准差

```bash
uv run --locked python run_experiments.py --scenario all --device mps --seeds 42 43 44
```

各组合各 seed 独立保存；已经完成的 seed 42 自动复用。不同 seed 共用固定训练/验证划分，改变训练顺序等训练随机过程。作者模型构造器内部有固定初始化 seed；这里保留作者初始化，因此不能把这三个 seed 描述成“三种独立随机初始化”。

### E. 只做某个具体实验（自动补齐依赖）

完整模态教师及其全部缺失组合：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only teacher
```

单模态基线 / 简单融合：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only single-hsi single-sar single-dsm
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only early-concat late-mean
```

三个独立 MSH 缺失实验（也可以拆成三条命令）：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only msh-hsi msh-sar msh-dsm
```

这里会自动训练/加载 `teacher`、三个 `jad-X` 和三个 `single-X`，但只输出请求的三个 MSH 的测试结果。想同时输出基线/JAD 结果，就把它们的 ID 也写进 `--only`，或直接运行 core。

只做 JAD 学生：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --only jad-hsi jad-sar jad-dsm
```

### F. 在 core 基础上增加消融和改进

一次跑完整扩展；相同输出目录会复用 core 模型，只训练新增阶段：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --preset extended
```

只加 ModDrop：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --preset extended --only moddrop
```

只做 HSI 的蒸馏消融、η 干预、以及单模态/JAD/MSH 的噪声对比：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --preset extended --only single-hsi jad-hsi msh-hsi kl-only-hsi mmd-only-hsi msh-fixed-eta-at-test-hsi
```

更多种子可在每条命令末尾添加 `--seeds 42 43 44`。也可将 `--scenario` 换成另外三组。查看所有合法 ID：

```bash
uv run --locked python run_experiments.py --preset extended --list
```

### G. 仅评估已训练模型

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --evaluate-only
```

`--evaluate-only` 也支持 `--only msh-hsi`；缺少需要的完整训练检查点会直接报错，不会偷偷重新训练。评估 extended 的模型时，带上 `--preset extended`。

### H. 自定义训练长度与中断续跑

如果想先跑中等长度的预实验，使用独立目录：

```bash
uv run --locked python run_experiments.py --scenario aug-hs-sar-dsm --device mps --epochs 100 --fusion-epochs 30 --output outputs/pilot-100
```

按 Ctrl+C 中断后，**重复原命令**即可恢复。每个完整 epoch 保存 `last.pt`，含优化器状态；中断发生在 epoch 中间时，该 epoch 会重跑。断点恢复按 epoch 重新设置训练随机种子；CPU 已验证与不中断训练参数完全一致，MPS 不承诺跨设备/版本的逐位确定性。

相同目录不允许更改 epoch 数、batch size、学习率、蒸馏权重、划分等配置，以免混用不匹配模型。改这些参数请换 `--output`。可以在相同目录追加 seed、实验 ID 或从 core 扩展到 extended。跨组合的单模态基线为独立路径，未做跨组合缓存复用。

如 MPS 资源紧张，把 batch 改为 16 或 8，并使用新目录；CPU 运行则把 `--device mps` 改为 `--device cpu`。不要并行开启多组训练，以免争用 24GB 统一内存。

## 5. 结果保存在哪里

默认正式输出的绝对路径：

```text
/path/to/sense-course/outputs/experiments/
├── summary.csv                    所有已完成组合/seed/测试条件，OA、AA、Kappa
├── summary-mean-std.csv            按组合/方法/保留模态/噪声聚合各 seed
├── summary-mean-std.json           同上，保留结构化 seed 列表
└── aug-hs-sar-dsm/
    └── seed-42/
        ├── config.json            超参数、精确训练/验证索引、协议说明
        ├── runtime.json           实际设备、PyTorch、MPS 可用状态
        ├── completed.json         本次请求全部完成的标志；失败/中断时没有新完成标志
        ├── results.json           已评估的全部条件（可追加）
        ├── models/
        │   ├── teacher/
        │   │   ├── best.pt        验证集最优模型，包含所选 epoch
        │   │   ├── last.pt        最新完整 epoch，模型＋优化器＋history，用于续跑
        │   │   └── history.json   每 epoch 训练 loss、验证 OA/CE、学习率、用时
        │   ├── single-hsi/ ...
        │   ├── jad-hsi/ ...
        │   └── msh-hsi/ ...       部署权重为两个单模态分支＋η，没有教师参数
        └── evaluation/
            ├── msh-hsi__keep-hsi__noise-0/
            │   ├── metrics.json         OA/AA/Kappa、各类 recall/support、混淆矩阵、η等
            │   ├── confusion_matrix.csv 行是真实类，列是预测类
            │   └── predictions.csv      官方测试索引、真实类、预测类（类别从0开始）
            ├── teacher__keep-hsi-sar__noise-0/  缺失DSM
            ├── teacher__keep-hsi__noise-0/      缺失SAR和DSM
            └── ...
```

`--output` 指向哪里，上面的目录树就建在哪里。不要把 `outputs/demo`、`outputs/experiments-quick` 和正式 `outputs/experiments` 的数字混在一起。

- OA：总体准确率；AA：每个有测试样本的类别召回率的平均；Kappa：机会一致性校正系数。
- OA/AA 按 **0～1** 保存，写成百分数需乘 100。各类 recall 也为 0～1。
- 只有一个 seed 时，标准差留空/null，不会伪造为 0；多个 seed 使用样本标准差（ddof=1）。这些 seed 不是独立数据集，因此均值±标准差不是置信区间。
- `metrics.json` 的评估用时包含数据读取/传输，不是纯 GPU 核心延迟；不同输入条件的速度请在同一设备和配置下比较。

## 6. 如何形成作业中的结果表

建议最少做三张表：

1. **完整模态融合**：单模态、early-concat、late-mean、teacher，展示多源信息是否有帮助。
2. **固定保留模态**：每种 X 一行，对比 single-X、teacher 的 keep-X、jad-X、msh-X。三模态教师另外列出缺失一种时的三种条件。
3. **消融/鲁棒性**：KL-only、MMD-only、JAD、MSH；噪声强度曲线；有余力再加 teacher 与 ModDrop 的缺失对比。

可把 ModDrop 当作自己的工程改动进行分析，但要如实说明算法来源与改动。不要仅复现 MSH 后把整套论文方法称为自己提出的方法。

## 7. 正式实验协议与论文原数值的关系

- 默认将官方训练样本按类别分层划分为 **80%训练＋20%验证**；所有方法、所有训练 seed 共用同一划分。Augsburg 原有 761 个训练点，正式这里使用 608 训练＋153 验证；Houston 原有 2832 点，使用 2265＋567。不是用全部官方训练点反复观察官方测试准确率选模型。
- 每阶段按验证 OA 选 `best.pt`；OA 相同再按验证 CE 选。完成本次请求的全部训练后才读取官方测试集。正式评估用全部官方测试点：Augsburg 77533、Houston 12197。没有再做训练＋验证的全量重训。
- 保留作者的整幅场景逐波段 min/max 归一化和 7×7 patch；它利用了未标注场景的范围统计，不是“仅训练点统计”的新协议。训练/验证点不重复，但相邻点的 7×7 patch 可能重叠；这是该像素分类设置，不是严格空间隔离泛化实验。
- 使用作者网络和 JDA，前面已做 Mac 内存优化；优化器为 SGD（lr=0.001，momentum=0.9，weight_decay=0.001），5 epoch 线性 warmup＋cosine；融合用 sigmoid η 保证 [0,1]。HSI 学生默认 α=2、β=1，其他学生 α=1、β=5，温度 T=2；可通过命令行覆盖后使用新输出目录。这些是当前作业协议的明确默认值，不声称逐项复刻作者所有实验脚本配置。
- 因训练/验证划分、优化策略和 η 约束等差别，**这里是基于论文的可复现实验实现，不承诺复现论文表格原数值**。报告应写明协议，比较同一实现内的公平对照。
- 默认没有类别加权或超参数搜索。Augsburg 类别不平衡，必须一起看 AA 和每类召回率，不能只看 OA。若改类别权重，要作为另一实验并记录。
- 本次只执行短程流程验证，没有替你跑完 500/100 epoch 的正式实验，也没有得到可报告的最终性能。完整测试会比原小 demo 慢很多。

## 8. 验证入口

```bash
uv run --locked python verify_experiments.py
```

专项验证 OA/AA/Kappa、无样本类别处理、训练验证索引不重叠、单样本尾 batch、CPU epoch 断点恢复与连续训练一致、模型独立推理。四组 core 的 CPU 短程运行与 Augsburg extended 的原生 MPS 短程运行另存于 `outputs/verification-v2-*`，这些都不是论文成绩。
