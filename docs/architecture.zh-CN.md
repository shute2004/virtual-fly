# 当前 architecture

[English](architecture.md) · [日本語](architecture.ja.md) · [简体中文](architecture.zh-CN.md)

本文描述当前 production / canonical path **实际实现的 architecture**。它不是 roadmap，也不会把当前语义追溯性地套用到 historical artifact 上。

## 1. System boundary

`virtual-fly` 将职责分为四部分：

1. **神经系统** — MaleCNS dynamics、modulation、eligibility、局部可塑性；
2. **感觉/外周 embodiment** — 局部物理感受转换与 motor neuron→muscle state；
3. **身体与环境** — FlyBody / MuJoCo 与 Flyppy geometry；
4. **实验 orchestration** — reset、curriculum 条件、persistence、provenance、report。

实验层可以判断物理任务事件何时发生，以及该事件刺激哪个神经群体，但不会计算 target action 或“正确的” synaptic weight。

```text
physical scene
    ↓
local sensory transduction
    ↓
MaleCNS neural runtime
    ↓
individual motor-neuron activity
    ↓
peripheral muscle state
    ↓
FlyBody / MuJoCo
    ↓
physical scene
```

## 2. Canonical closed loop

当前 canonical Flyppy path：

```text
Flyppy v7 physical environment
        ↓
FlyBody compound-eye geometry
        ↓
local direct-ray sampling (K=13)
        ↓
released MaleCNS R1-R6 body-ID currents
        ↓
whole-MaleCNS signed neural dynamics
        ↓
local eligibility + dopaminergic modulation
        ↓
individual released motor-neuron spikes
        ↓
WholeBodyPeriphery
        ↓
FlyBody v7 physical actuation
        ↓
MuJoCo state transition
        ↓
Flyppy event detection
        ├─ gate passage → selected reward-associated DAN current
        └─ collision    → selected aversive-associated DAN current
```

这条路径中不存在外部神经网络或 action policy。

## 3. MaleCNS source 与 mutable state

公开 MaleCNS data 被视为初始结构 snapshot，而不是可写的 training file。

```text
immutable source snapshot
        +
mutable functional weight state
        +
short-term neural/plasticity state
        =
current simulated nervous system
```

Canonical v1 使用：

- 166,700 neurons；
- 25,582,938 directed connection edges；
- released neurotransmitter / annotation information；
- class-DAN = dopamine consensus 且公开 annotation 为 `class=DAN`。

学习过程不会原地修改 source file。

## 4. Neural runtime

大型 neural runtime 由 `crates/` 下的 Rust 实现。Python 负责 orchestration 与身体集成，Rust process 负责 whole-CNS state evolution。

当前 activity semantics 区分 silent、depolarizing activity event 和 hyperpolarizing activity deviation，避免把抑制/graded pathway 压缩成单一 unsigned spike bit。

主要职责：

- sparse MaleCNS graph storage；
- state propagation；
- modulation state；
- eligibility dynamics；
- local weight update；
- shared-weight population transaction；
- checkpoint load/save。

GPU 是计算 backend，不是独立的 learned controller。

## 5. 局部可塑性

production plasticity 使用局部规则。概念上，edge update 依赖局部 eligibility 与 postsynaptic neuromodulation：

```text
pre/post activity history
        ↓
eligibility
        +
local dopaminergic modulation
        ↓
local bounded synaptic update
```

环境不会把 `reward = +1/-1`、Q-value、target action 或 policy loss 传给 weight optimizer。

PlasticFastGraph 的作用是减少需要扫描/更新的 edge，而不是替换为另一种 learning rule。

## 6. Neuromodulatory event

任务事件会被转换为神经刺激。

Canonical class-DAN semantics 使用公开的真实 DAN body ID。reward-associated / aversive-associated subset 是**实验刺激条件**，并不声称相应 label 中每个 neuron 在所有自然情境下都表达统一 scalar reward。

```text
gate passage
    ↓
current into selected reward-associated DAN subset

collision
    ↓
current into selected aversive-associated DAN subset
```

Canonical frozen evaluation 将两种 task-triggered DAN current 都设为 0。

## 7. Vision boundary

production vision 不在 CNS 外部进行 obstacle classification。

```text
MuJoCo / FlyBody eye geometry
        ↓
per-ommatidium local direct rays
        ↓
local photoreceptor transduction/adaptation
        ↓
corresponding released R1-R6 body IDs
        ↓
MaleCNS visual circuitry
```

空间 identity 一直保留到 CNS。不会在外部完成 edge detection、object recognition、motion label 或 global spatial pooling 后把“答案”注入 CNS。

## 8. Haltere boundary

haltere path 从物理身体提取局部 mechanical signal，并映射到 MaleCNS sensory afferent。

Canonical v1 使用推断的 97-neuron timing subset（left 46 / right 51）、current gain 0.05、`interaction-load-v2`。

这是明确的 `inferred` sensory boundary，不能描述为完整 haltere physiology 的直接观测。

## 9. Motor boundary

production behavior 不再由 DNg02 population-average decoder 生成。

```text
released motor-neuron body ID activity
        ↓
WholeBodyPeriphery
        ↓
individual motor-unit / muscle activation state
        ↓
versioned FlyBody adapter
        ↓
physical force/torque in MuJoCo
```

peripheral/body seam 含有工程近似，但不会读取障碍物位置、reward state 或 desired action。

旧 DNg02 aggregate 路径只作为 legacy / diagnostic 保留。

## 10. Body version lineage

body version 编号是 historical label，不是简单的线性继承：

```text
v3
└─ v4
   ├─ v5
   ├─ v6
   └─ v7

v8 = v6 measured steering + v7 neutral trim
```

因此，**v7 不是“v6 再加一些改良”**。Canonical v1 使用 v7，是因为它被固定为 canonical condition，而不是因为它包含 v5/v6 的全部 mechanics。

## 11. Environment v7

Flyppy environment version 同样是明确的实验条件。v7 加入物理 side wall，防止 3D body 从 gate 侧面绕行后仍被计为通过。

gate pass 判定不把 thorax 中心当作零尺寸点，而会在 wall plane 周围考虑 full-body geometry。

## 12. Population training

production population trainer 共享一个 global weight state。

每个 slot：

1. 从特定 global weight version 开始 episode；
2. 累积 episode-local plasticity operation；
3. 在 `async` 模式下独立结束；
4. 将 transaction rebase 到最新 global weights；
5. commit 并生成下一个 global weight version。

这**不是 weight averaging**。

由于 completion order 可能变化，async shared-weight training 不被声明为与 serial single-fly training bitwise 等价。source weight version 与 staleness 会记录在 provenance 中。

## 13. Curriculum boundary

curriculum code 选择 reset / experience condition，但不会选择 motor action。

boundary-band curriculum 将 focus/easier experience 与 full-course evaluation logic 分离。curriculum success 是 experiment scheduling state，不是 scalar synaptic reward API。

## 14. Checkpoint boundary

当前 population persistence semantics 为 `global-weights-only-v1`。

跨 episode 保留 learned global weights；membrane、spike、refractory、activity trace、modulation、eligibility 等 episode-local state 会 reset。

因此，当前 population checkpoint **不能**描述为完整持久化的生物个体状态。

磁盘上的 checkpoint container 仍有兼容性数组，但公开语义遵循 weights-only contract，而不是 raw file layout。

## 15. Frozen evaluation

可公开的 weight comparison 至少需要区分：

- plasticity OFF；
- task-triggered DAN stimulation OFF；
- initial / final stored weight state；
- 相同的 body/environment/sensory condition。

Canonical v1 满足这些条件。部分旧 fixed evaluation 并未关闭 event-triggered DAN current，这也是它们保留为 historical 而不是 canonical evidence 的原因之一。

## 16. Observer / viewer boundary

3D body viewer 与 neural viewer 都是 observer-only。

没有 viewer 时，production training 不需要持续生成 viewer snapshot。viewer attach/detach 不得改变 neural、physical、plasticity 或 curriculum state。

## 17. 当前 code ownership

```text
src/virtual_fly/embodiment/   body factory、versioned body seam、retina、haltere、periphery
src/virtual_fly/runtime/      neural bridge、body worker、packed process、telemetry
src/virtual_fly/training/     config、curriculum、population scheduling、checkpoint semantics
src/virtual_fly/playback/     frozen playback/evaluation orchestration
src/virtual_fly/reporting/    stable report/history output
src/virtual_fly/reproducibility.py
                              provenance/hash/semantic validation
src/virtual_fly/semantics.py  compatibility-sensitive semantic identifiers
crates/                       Rust neural runtime / CLI runner
scripts/                      launcher、data preparation、analysis、legacy diagnostic
```

开发者 module map 见 [`code-structure.md`](code-structure.md)。

## 18. Historical code / artifact

historical script、report、artifact 具有 provenance 价值，但它们的存在并不意味着属于 current production path。

新的 production code 应依赖 `src/virtual_fly/`，而不是 historical script module compatibility shim。

historical experiment result 保留原始 semantics/provenance label。见 [`results.zh-CN.md`](results.zh-CN.md)。

## 19. 当前不能视为 canonical biology 已实现的内容

不应从当前 architecture 推断出：

- 每个 neuron / synapse 都具有完整 conductance-level biophysics；
- 完整 receptor-specific neurotransmitter dynamics；
- canonical v1 中已经验证 structural synapse formation/pruning；
- 完整 peripheral sensory physiology；
- 完整 flight-muscle mechanics；
- training episode 之间保留完整 persistent neural state。

这些是明确的 model boundary，而不是隐藏组件。
