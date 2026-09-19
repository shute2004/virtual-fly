# 基础资料与外部资源

[English](../en/references.md) · [日本語](../ja/references.md) · [简体中文](references.md)

本文是项目所使用的一次文献、官方数据集和主要外部软件的索引入口。

实际使用外部资料时，不只记录 URL，还应在实验来源记录中保存论文或数据集版本、许可和获取日期。

## 1. MaleCNS

### Male CNS Connectome project

- 项目：https://male-cns.janelia.org/
- 下载 / 程序化访问：https://male-cns.janelia.org/download/
- 数据集：`male-cns:v1.0`
- neuPrint：https://neuprint.janelia.org/
- 论文附属仓库：https://github.com/flyconnectome/2025malecns

官方下载页面同时提供 `neuprint-python` API 和批量数据下载。数据集采用 CC-BY 许可。

### 论文

Berg et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature.

- https://www.nature.com/articles/s41586-026-10735-w

这是把脑和腹神经索视为连续中枢神经系统连接组的主要参考资料。

### MaleCNS 视叶的视网膜对应关系

官方注释 `assignedOlHex1` / `assignedOlHex2` 被用作视叶柱的视网膜对应坐标。视觉输入实现不能把任意 body ID 顺序或数组下标当作空间位置。

MaleCNS 官方媒体还公开了从 R1-R6 到下行神经元的视觉—运动通路示例。

- https://male-cns.janelia.org/media/

## 2. 果蝇早期视觉系统

### 完整视觉系统连接组

Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature.

- https://www.nature.com/articles/s41586-025-08746-0

该研究明确指出，lamina 并未完全包含在成像体积中，因此重建出的 Lai 和 R1-R6 数量低于真实生物总量。`virtual-fly` 不会人为补造缺失的 R1-R6，也不会强行让每个柱都拥有六个细胞。只有 MaleCNS 公开数据中实际存在的 R1-R6 才标记为 `observed`。

### 神经叠加

Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

来自不同相邻小眼、但共享同一视觉轴的 R1-R6 会汇聚到同一个 lamina cartridge。实现视网膜对应输入时，这是重要的布线原则，不能简单把同一个小眼中的六个细胞平均成一个点。

Juusola et al. 的电生理方法概述：

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

该资料可用于理解 lamina cartridge 的局部视网膜对应处理，以及 R1-R6 向 L1-L3 等目标发送组胺能输出的机制。

### 方向选择性与柱状组织

Fisher et al. (2015/2016), *Complementary mechanisms create direction selectivity in the fly*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5 的方向选择性由视叶柱状阵列上的局部回路形成。因此 `virtual-fly` 不会在 CNS 外部先计算类似 T4/T5 的运动特征再注入下游，而是尽量让这些响应从 R1-R6 开始，通过公开神经回路产生。

## 3. FlyBody

- 仓库：https://github.com/TuragaLab/flybody
- MuJoCo Menagerie 中的副本：https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

FlyBody 是果蝇三维解剖身体和 MuJoCo 物理模型的主要来源之一。

重要：本项目不会把 FlyBody 附带的强化学习策略用作 `virtual-fly` 的学习机制。项目使用身体形状、物理、执行机构等资源，而运动命令来自虚拟 CNS。

## 4. FlyGym / NeuroMechFly

- 文档：https://neuromechfly.org/
- 安装：https://neuromechfly.org/installation/
- 教程：https://neuromechfly.org/tutorials/

这些项目用于参考感觉整合、身体模拟、MuJoCo 集成、FlyBody 使用和 GPU 执行方式。

FlyGym 2.x 与旧版 API 不兼容，因此必须明确固定依赖版本。

FlyGym 标准复眼 `Retina` 使用合成六角网格，因此不能直接视为 MaleCNS 个体的 body ID 与视网膜位置映射。当前 MaleCNS 边界使用 FlyBody 的原始眼部相机作为局部光源，同时优先采用 MaleCNS 中实际的柱坐标。

## 5. MuJoCo

- 项目：https://mujoco.org/
- 仓库：https://github.com/google-deepmind/mujoco
- Menagerie：https://github.com/google-deepmind/mujoco_menagerie

MuJoCo 是身体物理模拟的主要候选引擎。

## 6. 访问 MaleCNS

官方 Python 访问方式大致如下：

```python
from neuprint import Client

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token="...",
)
```

认证令牌不得提交到代码或配置文件中。

## 7. 仍需整理的一次文献

在继续增加或精化机制之前，至少应整理以下果蝇一次研究：

- 膜电位与放电特性；
- 化学突触传递；
- 电突触；
- 神经递质与受体；
- 蘑菇体可塑性；
- PAM / PPL1 等多巴胺能回路；
- 奖赏性与厌恶性条件学习；
- STDP 等活动依赖可塑性；
- 稳态可塑性；
- 结构可塑性；
- 自发运动；
- 飞行中央模式和下行控制；
- 翼运动神经元与飞行肌；
- 复眼光感受转换；
- 本体感觉与机械感觉。

实现参数应优先使用果蝇特异的证据，而不是仅仅因为方便就采用一般神经科学中的数值。

## 8. 使用参考资料的规则

1. 优先使用一次论文和官方数据，而不是博客或解说文章。
2. 使用其他物种的数值时，应按情况标记为 `assumed` 或 `inferred`。
3. 不自动跟随数据集更新，每次实验固定明确版本。
4. 把实测连接组与由其产生的可变虚拟个体状态分开保存。
5. 不得在未说明的情况下，用外部软件附带的强化学习或神经网络控制器替代 CNS。
6. 不得用 CNS 外部的便利特征工程替代真实局部回路承担的特征提取。
