# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` 是一个研究项目：以成年雄性果蝇公开的 MaleCNS 连接组作为初始状态，让神经活动、神经调制和局部突触可塑性随时间演化，并与 FlyBody / MuJoCo 身体及物理环境形成闭环。

它并不是“把果蝇连接组当作人工神经网络的线路来训练”。当前实现不使用反向传播、梯度下降、Q 学习、策略梯度、外部神经网络控制器，也不使用为了躲避障碍物而手写的行为规则。与学习相关的权重变化由局部神经活动、资格迹和多巴胺能神经调制共同产生，并发生在模拟神经系统内部。

> **当前状态：** 基准实验 canonical v1 及其完整复现脚本已经完成。canonical v1 证明了从当前代码到实验结果的来源链可以完整追踪，也确认局部可塑性确实会改变 MaleCNS 中保存的权重。但在这个较短的实验中，**没有观察到明确的行为改善**。

**科学参考文献：** [`docs/zh-CN/references.md`](docs/zh-CN/references.md) 汇总了当前实现所参考的主要一手论文和官方资料，并说明每项资料具体支撑了哪些实现判断。

## 历史开发系的对比动画

![历史开发系 v240 到 v966 的 Before/After 对比](docs/assets/historical-v240-v966-before-after.gif)

这段动画由公开展示用的原始视频 `before-after-neural.mp4` 转制而来。为了便于在 README 中观看，播放速度提高到了 **1.5 倍**。它同时展示了 `v240 → v966` 的 Before/After、身体运动以及神经活动可视化，因此被放在页面靠前的位置，作为理解项目的直观入口。

但它**不是 canonical v1 的实验依据**。v240 / v960 / v966 / v1704 属于早期开发系，与当前基准实验的来源记录不同，部分运行定义也不同。原始视频、转换后的 GIF 及其哈希值记录在 [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json) 中。

## 基准实验（canonical v1）

对外发布时的基准结果位于 [`canonical/canonical-v1/`](canonical/canonical-v1/README.md)。它与早期开发结果有意分开保存。

| 项目 | canonical v1 |
|---|---|
| 科学计算所用代码 | `7fa464aad7269d34f46f1171080b51e095d1d811`（工作树无修改） |
| MaleCNS 快照 | 166,700 个神经元 / 25,582,938 条有向边 |
| DAN 定义 | 公开注释 `class=DAN` 与多巴胺判定同时成立，共 338 个 |
| 身体 / 环境 | v7 / v7 |
| 视觉 | `direct-ray`，每个小眼 13 条射线 |
| 平衡棒输入 | 97 个神经元的时序子集，增益 `0.05`，`interaction-load-v2` |
| 学习条件 | 6 个回合，同时运行 2 个个体，异步共享权重，`boundary-band` |
| 共享权重版本 | v0 → v6 |
| 神经更新总步数 | 484 |
| 保存权重实际发生变化的边 | 2,163,179 |
| 初始状态固定评估 | 通过 1 个门后，在控制步 120 撞上下一道门 |
| 最终状态固定评估 | 通过 1 个门后，在控制步 120 撞上下一道门 |

固定评估同时关闭可塑性和任务事件触发的 DAN 刺激（`reward_current=0`, `aversive_current=0`）。因此，canonical v1 能够说明的是：**在当前局部可塑性规则下，保存的权重确实发生了变化**。它并不能证明行为改善、泛化能力或长期学习稳定性。

2026 年 9 月 19 日，我们从无修改的 `7fa464a` 开始重新执行了完整复现流程。由源数据生成的静态产物哈希值全部与记录的基准值一致；学习顺利进行到共享权重 v6；发生变化的边数同样是 2,163,179；初始和最终检查点都能重新载入；两次固定评估的结果也与基准记录一致。

详细资料：

- [`canonical/canonical-v1/reference-report.md`](canonical/canonical-v1/reference-report.md) — 基准实验结果摘要
- [`canonical/canonical-v1/reference-manifest.json`](canonical/canonical-v1/reference-manifest.json) — 完整来源记录
- [`docs/zh-CN/results.md`](docs/zh-CN/results.md) — 基准结果与历史开发结果的区分
- [`docs/zh-CN/reproducibility-fixes-2026-09-19.md`](docs/zh-CN/reproducibility-fixes-2026-09-19.md) — 来源记录与运行定义的审计、修复说明

## 当前已经实现的内容

当前执行路径如下：

```text
Flyppy 物理环境
        ↓
FlyBody 复眼几何 + 局部 direct-ray
        ↓
向公开 MaleCNS 中对应的 R1-R6 body ID 注入电流
        ↓
整个 MaleCNS 的神经活动
        ↓
局部资格迹 + class-DAN 神经调制与可塑性
        ↓
公开运动神经元逐个产生的放电
        ↓
全身外周肌肉状态
        ↓
FlyBody / MuJoCo 物理驱动
        ↓
Flyppy 物理环境
```

通过门或发生碰撞等任务事件不会直接改写权重。事件会被转换为对选定真实 DAN 神经元群的电流刺激，之后的突触变化由神经系统内部的局部规则计算。

当前几个重要边界如下：

- MaleCNS 的原始快照只读，不在学习过程中改写。
- 视觉输入不使用外部障碍物分类器，并保留 R1-R6 的局部视网膜位置关系。
- 运动输出不使用“先对神经元群求平均再选择动作”的解码器，而是保留公开运动神经元各自的`body ID`。
- 可视化界面只负责观察，不会把状态反馈给学习过程。
- 并行学习只有一份共享权重；膜电位、放电状态、不应期、活动历史、神经调制和资格迹等短期状态会在每个回合重新初始化。
- 当前并行学习的检查点只把共享权重视为跨回合保存的状态，并不是一个虚拟个体全部状态的持久化快照。

实现细节见 [`docs/zh-CN/architecture.md`](docs/zh-CN/architecture.md) 和 [`docs/zh-CN/code-structure.md`](docs/zh-CN/code-structure.md)。

## 本项目没有宣称的内容

`virtual-fly` 试图依据生物学资料在计算机中重建果蝇，但并不宣称当前模拟器已经完整复制了真实果蝇。

尤其是，canonical v1 **没有**证明以下内容：

- 6 个回合之后出现明确的行为改善
- 对未见条件具有泛化能力
- 长期学习过程稳定
- 所有神经元、突触、感觉器官和飞行肌肉都得到完整的生物物理重建
- 历史开发结果 v240 / v960 / v966 / v1704 与当前 canonical v1 是在相同条件和相同运行定义下得到的

我们会尽可能区分：上游数据中直接观测到的值、文献采用值、推断值、工程假设和校准值。

## 安装

### 环境要求

- Python `>=3.12,<3.15`
- [`uv`](https://docs.astral.sh/uv/)
- Rust 开发环境 / Cargo
- 能够运行 MuJoCo 的本地环境

当前基准结果在 macOS / Apple Metal GPU 上生成。由源数据确定性生成的静态产物会通过哈希值核对，但我们不保证包含异步执行的 GPU / MuJoCo 轨迹在不同硬件之间逐位完全一致。

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
```

大型外部数据和实验生成物不会直接提交到 Git。

## 复现 canonical v1

复现脚本会一次完成：从官方来源重新获取 MaleCNS 数据、重新生成快照和派生产物、进行 6 个回合的基准学习、验证初始和最终检查点、执行固定评估并生成来源记录。

```bash
bash canonical/canonical-v1/reproduce.sh
```

大型输出默认保存在仓库之外：

```text
${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions/
```

可通过 `VF_CANONICAL_OUTPUT_ROOT=/path/to/output` 指定其他保存位置。科学计算部分始终在由无修改的 `7fa464a` 创建的独立工作树中执行。

## 当前开发用学习

通常使用以下入口：

```bash
bash scripts/dev/train_flyppy_population.sh
```

`scripts/dev/train_flyppy_v3_population.sh` 仅作为兼容旧调用方式的包装脚本保留。开发过程中的运行结果和持续更新的报告不会自动成为基准结果。

## 仓库结构

```text
canonical/      基准实验包与来源记录
crates/         Rust 神经运行系统与执行程序
docs/           三语公开文档（`en/`、`ja/`、`zh-CN/`）、内部文档与历史资料
reports/        小型诊断结果与历史开发报告
scripts/        数据准备、分析、兼容命令和启动脚本
src/            当前 Python 包（`virtual_fly`）
tests/          运行定义、调度、运行系统和复现性测试
visualization/  只用于观察的神经活动可视化
artifacts/      本地大型数据、检查点和视频，不纳入 Git
release/        发布文件的来源记录，大型文件本体不纳入 Git
```

文档导航见 [`docs/zh-CN/README.md`](docs/zh-CN/README.md)。

## 结果、来源记录与大型数据

历史开发结果具有研究追踪价值，因此会被保留，但不会被重新解释为当前的基准结果。具体区分见 [`docs/zh-CN/results.md`](docs/zh-CN/results.md)。

MaleCNS 生数据、快照、检查点、轨迹数据、渲染视频、构建缓存等大型文件不会提交到 Git；仓库只追踪体积较小的来源记录、哈希值、配置和报告。

## 引用

引用信息见 [`CITATION.cff`](CITATION.cff)。用于长期存档的 GitHub 发布版本将从公开分支上的标签创建，并计划与 Zenodo 联动。详细流程见 [`docs/internal/en/release-and-zenodo.md`](docs/internal/en/release-and-zenodo.md)。

## 许可证

`virtual-fly` 自有的源代码和文档采用 [MIT License](LICENSE)。外部数据、外部软件、外部模型资源以及包含第三方材料的生成内容，仍分别受其原始许可条款约束。详情见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
