# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` 是一个研究项目：以公开的成年雄性黑腹果蝇（*Drosophila melanogaster*）MaleCNS 连接组作为初始状态，让神经活动、神经调制和局部突触可塑性随时间演化，并与 FlyBody / MuJoCo 身体和物理环境形成闭环。

它**不是**“用果蝇连接组训练一个人工神经网络”。当前 production path 不使用反向传播、梯度下降、Q-learning、policy gradient、外部神经网络控制器，也不包含手写的障碍物攻略逻辑。与学习相关的权重变化来自神经 runtime 内部的局部活动、eligibility 和多巴胺能调制。

> **项目状态：** canonical experiment v1 以及 end-to-end reproducer 已完成。canonical run 证明了当前代码下完整可追踪的执行链，并确认局部可塑性能够改变存储的 MaleCNS 权重；但这个短 canonical run **没有显示出类别意义上的行为改善**。

## Canonical result

对外公开时的基准结果位于 [`canonical/canonical-v1/`](canonical/canonical-v1/README.md)，并刻意与早期开发 lineage 分离。

| 项目 | Canonical v1 |
|---|---|
| 科学执行代码 | `7fa464aad7269d34f46f1171080b51e095d1d811`（clean） |
| MaleCNS snapshot | 166,700 neurons / 25,582,938 directed edges |
| DAN semantics | 公开 `class=DAN` annotation + dopamine consensus；338 modulators |
| Body / environment | v7 / v7 |
| Vision | direct-ray，13 rays/ommatidium |
| Haltere input | 97-neuron timing subset，gain `0.05`，`interaction-load-v2` |
| Training | 6 episodes，population 2，async shared weights，boundary-band |
| Global weight version | v0 → v6 |
| Aggregate neural step | 484 |
| 数值发生精确变化的存储 edge | 2,163,179 |
| Frozen initial evaluation | 通过 1 个 gate，随后在 control step 120 撞到下一个 gate |
| Frozen final evaluation | 通过 1 个 gate，随后在 control step 120 撞到下一个 gate |

Frozen evaluation 同时关闭 plasticity 和任务事件触发的 DAN stimulation（`reward_current=0`, `aversive_current=0`）。因此 canonical v1 证明的是：**在当前局部可塑性语义下，存储权重发生了变化**；它并不证明行为改善、泛化或长期学习稳定性。

2026-09-19，我们从 clean `7fa464a` 重新执行了完整 end-to-end reproducer。所有由 source 派生的静态 artifact hash 都与 reference 一致；训练到达 global v6；恰好 2,163,179 个存储 edge 发生变化；initial/final checkpoint 均可重新加载；两次 frozen evaluation 的结果也与 reference 一致。

详细资料：

- [`canonical/canonical-v1/reference-report.md`](canonical/canonical-v1/reference-report.md) — canonical 结果简报
- [`canonical/canonical-v1/reference-manifest.json`](canonical/canonical-v1/reference-manifest.json) — 完整 provenance
- [`docs/results.zh-CN.md`](docs/results.zh-CN.md) — canonical 与 historical result 的边界
- [`docs/reproducibility-fixes-2026-09-19.md`](docs/reproducibility-fixes-2026-09-19.md) — provenance / semantics 审计与修复

## Historical visualization

![Historical v240 to v966 comparison](docs/assets/historical-v240-v966-before-after.jpg)

上图来自用于发布展示的**历史** Before/After 可视化（`v240 → v966`）。它适合直观展示身体和神经 viewer 的闭环，但它**不是 canonical experiment 的证据**。v240/v960/v966/v1704 属于较早的 lineage，其 provenance 不同，部分语义也不同。

完整视频与 raw playback 不作为大型 Git blob 提交，而应作为带有明确 historical 标签的 GitHub Release asset 发布。其 hash 与 provenance 记录在 [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json)。

## 当前实际实现的内容

当前 production path：

```text
Flyppy physical world
        ↓
FlyBody compound-eye geometry + local direct rays
        ↓
released MaleCNS R1-R6 body-ID currents
        ↓
whole-MaleCNS neural dynamics
        ↓
local eligibility + class-DAN-mediated plasticity
        ↓
individual released motor-neuron spikes
        ↓
whole-body peripheral muscle state
        ↓
FlyBody / MuJoCo physical actuation
        ↓
Flyppy physical world
```

通过 gate 或发生 collision 等任务事件不会直接写入权重。事件只会转换为对选定真实 DAN 群体的电流刺激，之后的突触变化由 neural runtime 内的局部规则计算。

当前的重要边界：

- MaleCNS source snapshot 是 immutable 的；
- 视觉输入保留局部 retinotopic R1-R6 identity，不在 CNS 外加入 obstacle classifier；
- 运动输出维持 individual released motor-neuron ID，不使用 population-average action decoder；
- viewer 仅用于 observer，不向 training 反馈状态；
- population training 共享一个 global weight state，但 episode-local 的 membrane/spike/refractory/trace/modulation/eligibility 会在 episode 之间 reset；
- 当前 population checkpoint 持久化 global weights，并不是完整持续保存的生物个体状态。

实现细节见 [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md) 与 [`docs/code-structure.md`](docs/code-structure.md)。

## 本项目没有声称什么

`virtual-fly` 是一个具有生物学依据的计算重建项目，并不声称当前模拟器已经完整复制了真实果蝇。

尤其是 canonical v1 **没有**证明：

- 6 个 episode 后出现类别意义上的行为改善；
- 任务泛化；
- 长期学习稳定性；
- 每一个神经元、突触、感觉器官和飞行肌肉都具有完整的生物物理保真度；
- historical v240/v960/v966/v1704 与当前 canonical semantics 等价。

项目会尽可能区分 upstream 观测数据、文献来源、推断、工程假设与 calibration。

## Installation

### 环境要求

- Python `>=3.12,<3.15`
- [`uv`](https://docs.astral.sh/uv/)
- Rust toolchain / Cargo
- 可运行 MuJoCo 的本地环境

当前 canonical reference 在 macOS / Apple Metal GPU 上生成。source-derived static artifact 使用 hash 进行验证，但不保证不同硬件上的 async GPU/MuJoCo trajectory bit-identical。

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
```

大型 upstream dataset 与生成的 experiment artifact 不存入 Git。

## 重现 Canonical v1

canonical reproducer 会自动执行：fresh official source 获取、snapshot/derived artifact 重建、6 episode canonical training、initial/final checkpoint 验证、frozen evaluation 和 provenance manifest 生成。

```bash
bash canonical/canonical-v1/reproduce.sh
```

大型输出 bundle 默认写到 repository 之外：

```text
${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions/
```

可通过 `VF_CANONICAL_OUTPUT_ROOT=/path/to/output` 指定其他路径。科学计算部分始终在固定到 clean `7fa464a` 的 isolated worktree 中执行。

## 当前 development training

通用 production entry point：

```bash
bash scripts/dev/train_flyppy_population.sh
```

`scripts/dev/train_flyppy_v3_population.sh` 保留为 compatibility wrapper。development run 与持续更新的 report 不会自动成为 canonical result。

## Repository 结构

```text
canonical/      canonical experiment package 与 reference provenance
crates/         Rust neural runtime / runner
docs/           architecture、科学契约、reproducibility 与历史资料
reports/        小型 diagnostics 与 historical development report
scripts/        data preparation、analysis、compatibility CLI、launcher
src/            当前 Python package (`virtual_fly`)
tests/          semantics / scheduling / runtime / reproducibility tests
visualization/  observer-only neural viewer
artifacts/      本地大型 data/checkpoint/video；不进入 Git
release/        release asset manifest；大型文件本体不进入 Git
```

文档入口见 [`docs/README.zh-CN.md`](docs/README.zh-CN.md)。

## 结果、provenance 与大型数据

historical development result 作为开发来历保留，但不会被追溯性地重新标记为 canonical。边界写在 [`docs/results.zh-CN.md`](docs/results.zh-CN.md)。

MaleCNS raw、snapshot、checkpoint、trajectory、rendered video、build cache 等大型文件不进入 Git，只追踪小型 manifest、hash 和 report。

## Citation

引用 metadata 位于 [`CITATION.cff`](CITATION.cff)。用于长期归档的 GitHub Release 应从 publication branch 的 tag 创建并与 Zenodo 集成。详见 [`docs/release-and-zenodo.md`](docs/release-and-zenodo.md)。

## License

`virtual-fly` 自有 source code 与 documentation 使用 [MIT License](LICENSE)。upstream dataset、software、model asset，以及包含第三方 material 的 generated media 仍遵循各自的使用条款。详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
