# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` 是一个研究项目：以成年雄性果蝇公开的 MaleCNS 连接组作为初始状态，让神经活动、神经调制和局部突触可塑性随时间演化，并与 FlyBody / MuJoCo 身体及物理环境形成闭环。

它并不是“把果蝇连接组当作人工神经网络的线路来训练”。当前实现不使用反向传播、梯度下降、Q 学习、策略梯度、外部神经网络控制器，也不使用为了躲避障碍物而手写的行为规则。与学习相关的权重变化由局部神经活动、资格迹和多巴胺能神经调制共同产生，并发生在模拟神经系统内部。

> **当前状态：** 基准实验 canonical v1 及其完整复现脚本已经完成。canonical v1 证明了从当前代码到实验结果的来源链可以完整追踪，也确认局部可塑性确实会改变 MaleCNS 中保存的权重。但在这个较短的实验中，**没有观察到明确的行为改善**。

**科学参考文献：** [`docs/zh-CN/references.md`](docs/zh-CN/references.md) 汇总了当前实现所参考的主要一手论文和官方资料，并说明每项资料具体支撑了哪些实现判断。

## 历史开发系的对比动画

![历史开发系 v240 到 v966 的 Before/After 对比](docs/assets/historical-v240-v966-before-after.gif)

这段动画来自历史 `v240 → v966` 对比的 README 专用渲染版本，并以 **1.5 倍**速度播放。它把身体运动和神经活动可视化放在同一画面中，因此被放在页面靠前的位置，用来直观展示项目内容。

但它**不是 canonical v1 的实验依据**。Before 一侧的 v240 并不是未经学习的 MaleCNS 初始状态，而是已经经历 240 个训练回合的历史检查点。历史回放期间关闭了可塑性，但任务事件触发的 PAM 刺激仍然启用，而且这不是留出评估。After 一侧的 v966 也来自混合了多个早期条件的开发系，与 canonical v1 的来源和部分运行条件不同。

发布到 X 的视频、生成 README GIF 所用的源视频，以及计划附在 GitHub Release 中的 MP4，是同一 v240/v966 历史对比的不同渲染或转码文件，并不是同一个二进制文件。它们各自的用途、分辨率和哈希值分别记录在 [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json) 中。

## 基准实验（canonical v1）

对外发布时的基准结果位于 [`canonical/canonical-v1/`](canonical/canonical-v1/README.md)。它与早期开发结果有意分开保存。

| 项目 | canonical v1 |
|---|---|
| 公开使用的科学代码等价提交 | `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`（实验时 clean commit 的隐私脱敏等价版本） |
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

canonical 实验和 2026 年 9 月 19 日的端到端复现验证，实际都是在无修改的 commit `7fa464aad7269d34f46f1171080b51e095d1d811` 上执行的。公开前，为了只删除历史资料绝对路径中的本地 OS 用户名，我们对 Git 历史进行了隐私重写。内容上对应的公开提交是 `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`。在 canonical 时点，两者的树差异仅限于 14 个历史报告/来源记录中的用户名脱敏；可执行源码、canonical 条件和科学产物生成逻辑均未改变。因此，没有仅因这次隐私重写而重新运行完整训练。映射关系记录在 [`canonical/canonical-v1/privacy-redaction-provenance.json`](canonical/canonical-v1/privacy-redaction-provenance.json)。

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
- **完整复现 canonical v1 时：** 神经运行系统可使用的 wgpu 兼容 GPU

下面的基本验证不会执行 canonical 实验本身；完整运行 `canonical/canonical-v1/reproduce.sh` 时，神经运行系统的校准步骤会明确使用 GPU 后端。当前基准结果在 macOS / Apple Metal GPU 上生成。由源数据确定性生成的静态文件会通过哈希值核对，但我们不保证包含异步执行的 GPU / MuJoCo 轨迹在不同硬件之间逐位完全一致。

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
```

大型外部数据和实验生成物不会直接提交到 Git。

### 基本验证

安装完成后，无需下载 MaleCNS 数据或执行完整的 canonical 实验，也可以验证源码检出是否处于基本可用状态。

```bash
uv run python -m unittest discover -s tests -v
cargo test --workspace
bash -n canonical/canonical-v1/reproduce.sh
```

前两条命令分别运行 Python 侧的运行定义与调度测试，以及 Rust 神经运行系统测试。最后一条命令不会执行 canonical 实验，只检查复现脚本的 shell 入口是否具有有效语法。

## 检查点发布

检查点文件不会放入 GitHub 源码仓库。Hugging Face 上已经准备了 virtual-fly 的项目概览页面：

- [Hugging Face: `shute2004/virtual-fly`](https://huggingface.co/shute2004/virtual-fly)

下面四个检查点仓库要发布的内容已经准备完成，但**仓库本身尚未创建或公开**。在 GitHub 仓库公开之前，会先创建这些仓库并核对文件和哈希值：

- `virtual-fly-initial-YYYYMMDD` — canonical v1 的初始检查点
- `virtual-fly-trained-YYYYMMDD` — canonical v1 完成 6 个训练回合后的检查点
- `virtual-fly-video-before` — 已公开 Before / After 视频中 Before 一侧实际使用的历史检查点
- `virtual-fly-video-after` — 同一视频中 After 一侧实际使用的历史检查点

前两个带日期的仓库名只在实际发布日期确定后才替换 `YYYYMMDD`。视频用 Before / After 与 **canonical v1 的初始检查点和训练后检查点属于不同历史系**。v240 / v966 等内部版本号不会出现在公开仓库名中，只保留在来源记录里。MaleCNS 原始数据、FlyBody 资源、完整轨迹、全部视频帧以及无关的开发中间产物，不会仅因为本地存在就复制到 Hugging Face。

发布结构、模型卡模板、必要文件清单、归属信息和检查点哈希统一记录在 [`release/huggingface/`](release/huggingface/README.md)。

## 复现 canonical v1

复现脚本会一次完成：从官方来源重新获取 MaleCNS 数据、重新生成快照和派生产物、进行 6 个回合的基准学习、验证初始和最终检查点、执行固定评估并生成来源记录。

```bash
bash canonical/canonical-v1/reproduce.sh
```

大型输出默认保存在仓库之外：

```text
${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions/
```

可通过 `VF_CANONICAL_OUTPUT_ROOT=/path/to/output` 指定其他保存位置。公开版复现脚本会在隐私脱敏后的公开等价提交 `f9c86c9` 上创建独立 clean worktree 并执行科学计算。2026 年 9 月 19 日已记录的实验仍明确归属于原始 `7fa464a`，不会被描述成是在重写后的 SHA 上执行。

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

引用信息见 [`CITATION.cff`](CITATION.cff)。v0.1.0 首次发布只使用 GitHub 和 Hugging Face，不使用 Zenodo / DOI。将来如果需要固定的学术存档版本，再另行考虑 Zenodo。详情见 [`docs/internal/en/release-and-zenodo.md`](docs/internal/en/release-and-zenodo.md)。

## 许可证

`virtual-fly` 自有的源代码和文档采用 [MIT License](LICENSE)。外部数据、外部软件、外部模型资源以及包含第三方材料的生成内容，仍分别受其原始许可条款约束。详情见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
