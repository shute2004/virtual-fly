# 翼运动神经元与外周肌肉边界

[English](../en/wing-motor-boundary.md) · [日本語](../ja/wing-motor-boundary.md) · [简体中文](wing-motor-boundary.md)

## 1. 目的

为了避免在 CNS 外部放置行为决策器，MaleCNS 到身体之间的边界被下沉到公开数据中的单个翼运动神经元。

不再使用以下路径：

```text
DNg02 群体活动
  -> 平均放电率
  -> 左右差与整体平均
  -> 翼振幅
```

当前路径是：

```text
FlyBody 眼部
  -> 向 R1-R6 body ID 注入局部电流
  -> 整个 MaleCNS
  -> VNC 运动前回路
  -> 单个翼运动神经元放电
  -> 单个运动单位 / 神经肌肉状态
  -> 已识别翼肌
  -> 虚拟肌肉产生力矩
  -> FlyBody / MuJoCo
```

这不是根据运动神经元群体活动向量选择行为的解码器，而是从神经到肌肉、再从肌肉到关节力学的外周过程。

## 2. MaleCNS 翼运动神经元清单与使用范围

`scripts/data/prepare_wing_motor_map.py` 从 MaleCNS v1.0 中按 body ID 提取符合以下条件的公开神经元：

```text
superclass = vnc_motor
subclass   = wm
```

在 2026-09-14 检查的数据中：

- 共 67 个神经元；
- 左侧 33 / 右侧 34；
- 26 种已注释类型；
- `identified` 38；
- `identified_group` 22；
- `identified_variable` 2；
- `putative` 2；
- `unknown` 3。

最初的学习边界只使用标记为 `identified` 和 `identified_group` 的 60 个神经元。其余 7 个不会通过猜测目标来接入。

代表性排除项包括：

- `MNwm35 -> iii4`：`putative`；
- `MNwm36`：目标肌肉未解决；
- `tpn`：对 tp1 / tp2 的神经支配可变化；
- 类型尚未确定的公开翼运动神经元。

对于 DLM / DVM 这类分组类型，也不会在没有额外证据的情况下，仅凭 body ID 顺序把细胞分配给单个肌纤维。

## 3. 已实现的外周层

### 3.1 从单个运动神经元放电到运动单位状态

`scripts/embodiment/wing_muscle_periphery.py`：

- 从神经桥接层逐个读取 60 个 body ID 的放电；
- 为每个 body ID 保持独立激活状态；
- 对主动力肌、直接操舵肌和间接控制肌使用不同时间常数；
- 如果任何请求的 body ID 缺失，则直接失败停止，而不是继续执行；
- 不构造群体平均放电率，也不构造左右平均。

多个运动单位汇聚到同一肌群时，先独立更新每个运动单位，再通过带饱和的募集过程合并为该肌肉的力输出。这表示外周力量汇聚，而不是行为解码。

当前时间常数和每次放电的募集增量属于 `calibrated` 的初始校准参数，不被描述为 MaleCNS 中直接测得的生理常数。

### 3.2 异步主动力肌

DLM / DVM 是异步型间接飞行肌。因此模型不会假设运动神经元逐次直接指挥约 200 Hz 的每一次振翅。

`scripts/embodiment/flybody_muscle_adapter.py` 把 DLM / DVM 的低频激活转换成胸部飞行振荡系统可使用的主动力。振翅相位是用于近似伸展激活和胸部共振的外周机械状态，而不是 CNS 选择的行为。

为了表示每个实验阶段一开始已经处于飞行状态，只有主动力肌具有初始激活。这属于实验初始条件，与前进初速度和初始振翅相位类似，不是每个时间步由外部发送的运动命令。

### 3.3 从操舵肌到力矩

FlyBody 当前没有可直接连接的解剖学翼肌 MJCF 肌肉元素，因此第一阶段使用通过 `qfrc_applied` 施力的虚拟肌肉层。

目前只有定性作用方向约束较强的直接操舵肌会产生力矩：

- b1：偏向增加振幅；
- b2：偏向增加振幅；
- b3：与 basalar pair 拮抗；
- i1：偏向减小振幅。

数值增益标记为 `calibrated`，并不被当作实测力臂。

其他已识别操舵肌和间接控制肌仍保留激活状态，但不会通过猜测作用方向来人为生成力矩。

### 3.4 连接到 FlyBody

旧 DNg02 转换器只保留用于历史物理诊断，不再属于当前 Flyppy 学习路径。

`FlyBodyMuscleAdapter` 在每个物理时间步把继承的 6 个理想化位置执行器设置到当前角度，使它们保持中立，从而阻止旧解析位置命令单独产生行为。随后再把虚拟肌肉力矩施加到翼自由度上。

## 4. 奖赏与厌恶相关刺激

神经调制性质的结果刺激与运动输出边界是两套不同机制。

外部代码唯一可以直接送入神经运行系统的结果相关输入，是对真实 DAN body ID 的电流刺激。

当前 Flyppy 实验使用：

```text
通过门洞
  -> 向公开 PAM01（PAM-gamma5）DAN 注入电流

碰撞
  -> 向公开 PPL1 厌恶相关集合
     （PPL101/PPL103/PPL106）注入电流
```

不存在 `reward=+1`、`punishment=-1`、单一标量奖励或直接目标权重。刺激之后，多巴胺状态和可塑性在 MaleCNS 连接上由神经运行系统内部演化产生。

PAM01 内部也可能具有生理异质性，因此项目不会声称所有 PAM01 细胞在所有自然情境下都表示完全相同的奖励信号。PPL1 集合也同样只被视为使用真实公开 DAN 的实验刺激条件。

仅使用 PPL101 时，部分真实碰撞轨迹中，事件前资格度与其投射目标没有重叠，因此额外可塑性变化为 0。扩展到 PPL101 / PPL103 / PPL106 的 6 细胞集合后，在相同轨迹上有 4,935 条可塑性边出现事件特异差异。因此采用这一 6 细胞集合，作为避免依赖单一 compartment 的最小改动；没有加入外部标量惩罚或目标权重。

## 5. 学习开始前的检查

`scripts/dev/train_flyppy.sh` 在启动前重新生成并检查：

1. 奖赏、厌恶刺激及诊断用神经群；
2. 单个翼运动神经元映射；
3. R1-R6 视网膜位置映射；
4. 翼神经肌肉边界的基本运行检查；
5. Flyppy 闭环学习输入。

当学习路径切换到单个运动神经元边界后，旧的 DNg02 群体平均保护逻辑被移除。

## 6. 剩余限制

当前边界在以下部分有数据或文献支持：

```text
真实 MaleCNS -> 单个运动神经元 -> 解剖学上已识别的肌肉
```

但“肌肉 -> FlyBody 翼关节”仍然是虚拟肌肉近似。

要提高生物真实性，需要加入每块肌肉的附着点、力臂、激活—力量关系，并在 FlyBody 内增加解剖学肌肉模型。未知数值不能为了提高任务成绩而倒推填写。

## 7. 参考资料

- Lesser et al., *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife, version of record 2026.
  - https://elifesciences.org/articles/96084
- Ehrhardt et al., *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/
- Azevedo et al., *Synaptic architecture of leg and wing premotor control networks in Drosophila*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312524/
- *How tp1, an indirect wing steering muscle, stabilizes Drosophila's flight*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/
- Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature 2025.
  - https://www.nature.com/articles/s41586-025-09029-4

## 8. 来源类别

- `observed`：MaleCNS body ID、类型、左右和公开 CNS 连接。
- `literature`：翼运动神经元与肌肉目标、主动力肌/直接操舵肌/间接控制肌分类、DLM / DVM 异步飞行肌性质、b1 / b2 / b3 / i1 的定性作用。
- `putative`：MNwm35 -> iii4。
- `unknown`：MNwm36、未定型运动神经元，以及分组类型内部与单个肌纤维之间的对应。
- `calibrated`：外周激活时间常数、每次放电募集量、胸部振荡耦合、虚拟肌肉增益和力矩上限。
