# 为 virtual-fly 做贡献

[English](CONTRIBUTING.md) · [日本語](CONTRIBUTING.ja.md) · [简体中文](CONTRIBUTING.zh-CN.md)

`virtual-fly` 是研究软件。欢迎能够保持“实测生物学 / 文献来源 / 推断 / 工程假设 / calibration”之间边界的贡献。

进行科学或 runtime 修改前，请先阅读：

- [`README.zh-CN.md`](README.zh-CN.md)
- [`AGENTS.md`](AGENTS.md)
- [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md)
- [`docs/results.zh-CN.md`](docs/results.zh-CN.md)
- [`docs/requirements.md`](docs/requirements.md)

## Branch 方针

`main` 是 stable / publication branch。可以使用 `dev/flyppy-v3` 等 development branch，但公开 milestone 应集成回 `main`，不要让 default branch 长期停留在旧状态。

适合合并的条件：

- 相关 tests 通过；
- provenance / semantic contract 已记录；
- 没有隐藏 blocking regression；
- 临时 generated artifact 没有混入 source control；
- publication claim 与该 lineage 实际产生的 evidence 一致。

## Scientific invariant

除非进行明确的 project-level 设计变更，否则不要引入：

- 作为 nervous-system learner 的 backpropagation / gradient descent；
- Q-learning / policy gradient / actor-critic / 外部 learned policy；
- 手写 Flyppy obstacle policy；
- scalar reward 直接更新 weight；
- 用只保留最终答案的外部 feature extractor/classifier 替换已知局部感觉回路。

canonical architecture 中的学习路径是：

```text
sensory stimulation
    → neural dynamics
    → behavior
    → neuromodulatory stimulation
    → local plasticity
    → changed nervous system
```

## Provenance label

应尽可能区分：

- `observed`
- `literature`
- `inferred`
- `assumed`
- `calibrated`

不要把 inferred/calibrated boundary 描述成直接观测到的生物学事实。

## Historical / canonical result

v240/v960/v966/v1704 等 historical artifact 与 diagnostic 为了 provenance 被保留。不要把它们重新标记成 current canonical result。

公开结果 claim 应遵循 [`docs/results.zh-CN.md`](docs/results.zh-CN.md) 与 [`canonical/canonical-v1/`](canonical/canonical-v1/)。

## 修改范围

尽量让一个 change 只有一个可验证目的，例如：

- MaleCNS data semantics
- neural state stepping
- plasticity
- checkpoint format
- sensory transduction
- motor/peripheral mapping
- body physics
- reproducibility/reporting

修改 neural dynamics、plasticity、neuromodulation、sensory transduction、CNS→body mapping、reinforcement stimulation target 或 checkpoint semantics 时，应在同一个 change 中更新相关文档。

## Testing

新的 computational core change 应按需要加入最小组合：

- unit test
- deterministic/reference test
- checkpoint round-trip
- semantic contract test
- 必要的 backend parity check

不能为了性能优化而静默改变科学语义。

## Data / generated artifact

不要把大型外部 dataset 或生成 artifact commit 到 Git。

应放在 Git 外的例子：

- raw MaleCNS download
- large normalized snapshot
- checkpoint
- trajectory
- rendered MP4
- build cache / profiling trace

改为追踪小型 manifest、hash、config 和 report。

## Documentation language

对外公开 documentation 以英文为 default。主要公开入口最好同时提供日文和简体中文版本。

内部 research/development note 可以保留日文，以维持原始 context。不要仅为了表面统一而批量翻译 historical record。

## Pull request / commit

有科学含义的修改应说明：

- 改了什么；
- 为什么修改；
- 属于 observed/literature/inferred/assumed/calibrated 中的哪一种；
- 哪些历史 experiment 会变得不兼容；
- 如何测试。

## License

贡献到本 repository 的原创内容可按照 repository 的 [MIT License](LICENSE) 分发。只有在第三方 code/data/asset 的条款允许，并且能够保留必要 attribution/notice 时，才应提交这些内容。
