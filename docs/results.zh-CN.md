# 结果与 provenance 边界

[English](results.md) · [日本語](results.ja.md) · [简体中文](results.zh-CN.md)

本文规定哪些结果可以称为“当前 canonical result”，哪些结果只作为 historical development evidence 保留。

## 1. Canonical experiment v1

Canonical v1 是对外公开时的基准实验。其科学执行部分固定在 clean Git commit：

```text
7fa464aad7269d34f46f1171080b51e095d1d811
```

简要结果见 [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md)，完整 provenance 见 [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json)。

### 条件

- fresh 获取 official MaleCNS v1.0 source；
- fresh 构建 current-semantics MaleCNS snapshot；
- 166,700 neurons / 25,582,938 directed edges；
- class-DAN semantics：公开 `class=DAN` annotation + dopamine consensus；
- body v7 / environment v7；
- direct-ray vision，13 rays/ommatidium；
- 97-neuron haltere timing subset，gain 0.05，`interaction-load-v2`；
- population 2，async shared weights；
- boundary-band curriculum；
- 6 training episodes；
- global-weights-only checkpoint semantics。

### Canonical 中实际观测到的结果

- global weight version：v0 → v6；
- aggregate neural step：484；
- 数值精确发生变化的存储 edge：**2,163,179 / 25,582,938**；
- `|Δw| > 1e-7`：1,461,896 edges；
- `|Δw| > 1e-6`：536,060 edges；
- strengthened：961,611 edges；
- weakened：1,201,568 edges；
- maximum `|Δw|`：0.0018288875。

Frozen evaluation 关闭 plasticity 和任务事件触发的 DAN current。initial v0 与 final v6 都通过 1 个 gate，然后在 control step 120 撞到下一个 gate。

**解释：** canonical v1 证明了一个 provenance 可追踪的 closed-loop run，其中当前局部可塑性实现改变了存储的 MaleCNS 权重；但这个短实验**没有显示类别意义上的行为改善**，也没有证明泛化或长期学习稳定性。

## 2. End-to-end 重现验证

2026-09-19，我们从 clean `7fa464a` isolated checkout 开始，完整执行了 `canonical/canonical-v1/reproduce.sh`。

已验证：

- fresh MaleCNS source download；
- fresh snapshot 构建；
- fresh derived artifact 构建；
- 所有记录的 static artifact hash 均 byte-for-byte 一致；
- neural calibration 从 stable scales 0.004–0.007 中选择 0.005；
- canonical 6 episode training 完成；
- global v0 → v6；
- 精确改变 2,163,179 个 edge，与 reference 一致；
- initial/final checkpoint 均能在新的 bridge process 中重新加载；
- frozen initial/final evaluation 均与 reference outcome 一致；
- 最终 pinned scientific worktree 保持 clean。

大型 verification bundle 只作为临时数据存在。验证后，仅保留轻量 manifest/report 在 Git repository 外。当前 reproduction script 默认把大型 verification run 写到 repository 外。

## 3. Historical development results

v240、v960、v966、v1704 以及相关 posting/diagnostic run 作为 **historical result** 保留。

这些结果对于理解开发过程有科学价值，但不能与 canonical v1 互换。[`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) 的 provenance audit 记录了重要 lineage 差异，包括旧 dopamine-source semantics、pre-commit provenance、mixed v966 lineage，以及与当前 frozen canonical comparison 不同的 evaluation 条件。

不得把 historical behavior 描述成由 current class-DAN canonical semantics 生成的结果。

## 4. Historical Before/After media

用于公开展示的 historical comparison：

- **before:** global weight version v240；
- **after:** global weight version v966；
- body/environment：v7/v7；
- gate 2 center：13.703125 mm；
- evaluation plasticity：off；
- before outcome：1 gate；
- after outcome：2 gates。

这份 media 适合作为 historical learned-state comparison 的可视化，但不是 canonical behavioral result。

Git repository 只追踪一张小型 representative image 和 release-asset manifest。完整 MP4 与 raw playback JSON 留在 Git 之外，并应在 GitHub Release 中明确标记为 historical。见 [`../release/release-assets-v0.1.0.json`](../release/release-assets-v0.1.0.json)。

## 5. README、Release、论文与公开帖文中的 claim 规则

Canonical v1 支持以下表述：

- 使用公开 MaleCNS connectome 作为初始神经结构；
- closed-loop 视觉/机械感觉输入进入 neural runtime；
- individual released motor-neuron output 通过 peripheral model 驱动物理身体；
- 任务结果刺激选定的真实 DAN population，而不是直接提供 scalar weight-update target；
- 当前局部可塑性实现能够改变存储的 CNS 权重；
- canonical data/provenance path 可以 end-to-end 重现。

Canonical v1 **不支持**以下表述：

- “canonical v1 学会了解决 Flyppy”；
- “canonical v1 改善了行为”；
- “当前 simulator 已完整复制真实果蝇”；
- “historical v966 behavior 已在 current canonical semantics 下重现”；
- “当前模型已经证明一般性学习能力或长期稳定性”。

展示 historical media 时，应明确标记 historical，并链接到本文或 canonical reference report。
