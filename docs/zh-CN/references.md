# 科学参考文献与实现依据对应表

[English](../en/references.md) · [日本語](../ja/references.md) · [简体中文](references.md)

`virtual-fly`并不是把某一篇论文直接翻译成代码，而是综合使用了连接组学、视觉神经科学、电生理、运动回路、飞行肌肉和身体力学等多个领域的一手研究与官方数据。

本文档不仅列出参考资料，还明确说明**每篇文献具体支撑了virtual-fly中的哪类实现判断**。这里优先整理与当前实现直接相关的主要文献，而不是声称覆盖开发过程中查阅过的全部资料。被列入这里，并不意味着virtual-fly中的所有数值参数都由该论文直接测得。推断值、假设值和校准值仍会在实现和实验来源记录中单独标注。

## 当前实现的主要科学依据

| 文献 | 状态 | 领域 | 在virtual-fly中主要提供的依据 |
|---|---|---|---|
| Berg et al. (2026), *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell | 已同行评审 | MaleCNS | 将成年雄性果蝇脑与腹神经索作为连续CNS连接组处理；全CNS连接与注释 |
| Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature | 已同行评审 | 视觉 | 视叶组织、lamina与R1-R6重建范围的限制，以及不凭空补造缺失光感受器的判断 |
| Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*, Cell | 已同行评审 | 视觉 | 神经叠加，以及相邻小眼的R1-R6与lamina cartridge之间的对应关系 |
| Juusola et al. (2016), *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*, JoVE | 已同行评审 | 视觉 / 电生理 | R1-R6和lamina中间神经元对局部光刺激的生理响应 |
| Haag et al. (2016), *Complementary mechanisms create direction selectivity in the fly*, eLife | 已同行评审 | 视觉 | 方向选择性应从视叶局部回路中产生，而不是由CNS外部的运动分类器直接给出 |
| Lesser et al. (2024), *Synaptic architecture of leg and wing premotor control networks in Drosophila*, Nature | 已同行评审 | 运动控制 | 翼前运动网络的模块结构、运动神经元招募方式，以及腿与翼前运动网络的差异 |
| Cheong et al. (2026), *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife | 已同行评审・正式版本 | 运动控制 | 成年雄性腹神经索中下降输入→前运动回路→运动输出的组织方式 |
| Ehrhardt et al. (2025), *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster* | 预印本 | 运动控制 | 翼前运动回路的细胞类型级组织，以及运动神经元和肌肉关系的解释 |
| Teoh et al. (2025), *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight* | 预印本 | 飞行控制 | 间接转向肌tp1对飞行稳定和翼铰链力学的作用 |
| Vaxenburg et al. (2025), *Whole-body physics simulation of fruit fly locomotion*, Nature | 已同行评审 | 身体 / 物理 | FlyBody全身几何、MuJoCo身体模型，以及飞行和步行相关的身体力学 |

## 1. MaleCNS与成年雄性CNS连接组

### Berg et al. (2026)

**Berg, S., Beckett, I. R., Costa, M. et al.** *Sexual dimorphism in the complete Drosophila male central nervous system connectome*. Cell 189(18), 5504–5526.e15 (2026).

- DOI: https://doi.org/10.1016/j.cell.2026.08.015
- 论文: https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
- MaleCNS官方项目: https://male-cns.janelia.org/
- 下载与程序访问: https://male-cns.janelia.org/download/
- 本项目使用的数据集: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- 论文附属仓库: https://github.com/flyconnectome/2025malecns

这是把成年雄性果蝇脑与腹神经索视为连续CNS连接组时最核心的科学来源。`virtual-fly`把公开MaleCNS作为`t = 0`时的神经结构，之后产生的功能性权重变化则与原始数据分开保存。

解析视叶的视网膜对应关系时，还会使用MaleCNS官方注释中的`assignedOlHex1` / `assignedOlHex2`。官方站点也提供了从R1-R6到下降神经元的视觉—运动通路示例。

- https://male-cns.janelia.org/media/

## 2. 复眼与视叶回路

### Nern et al. (2025)

**Nern, A., Loesche, F., Takemura, S.-y. et al.** *Connectome-driven neural inventory of a complete visual system*. Nature 641, 1225–1237 (2025).

- DOI: https://doi.org/10.1038/s41586-025-08746-0
- 论文: https://www.nature.com/articles/s41586-025-08746-0

该工作用于解释视觉系统连接组及其覆盖范围的限制。特别是，lamina和R1-R6的部分缺失被视为数据边界，而不是允许人为补造不存在光感受器的理由。只有公开数据中实际存在的细胞才被视为已观测对象。

### Langen et al. (2015)

**Langen, M., Agi, E. et al.** *The Developmental Rules of Neural Superposition in Drosophila*. Cell 162, 120–133 (2015).

- DOI: https://doi.org/10.1016/j.cell.2015.05.055
- 开放论文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

这是神经叠加的重要依据。来自相邻小眼、但共享同一视觉轴的R1-R6会汇聚到同一个lamina cartridge。因此，当前实现不会简单地把一个小眼中的6个细胞平均成单一信号。

### Juusola et al. (2016)

**Juusola, M., Dau, A., Zheng, L. & Rien, D.** *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*. Journal of Visualized Experiments, issue 112, 54142 (2016).

- DOI: https://doi.org/10.3791/54142
- 开放论文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

该资料用于理解R1-R6光感受器和lamina中间神经元对局部光刺激的电生理响应。

### Haag et al. (2016)

**Haag, J., Arenz, A., Serbe, E., Gabbiani, F. & Borst, A.** *Complementary mechanisms create direction selectivity in the fly*. eLife 5, e17421 (2016).

- DOI: https://doi.org/10.7554/eLife.17421
- 开放论文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

该论文是理解T4/T5方向选择性如何在视叶局部回路中形成的重要依据之一。因此，当前视觉输入不会先在CNS外部计算“向上运动”“向下运动”等标签，再把答案注入神经系统。

## 3. 翼前运动回路与运动神经元

### Lesser et al. (2024)

**Lesser, E., Azevedo, A. W., Phelps, J. S. et al.** *Synaptic architecture of leg and wing premotor control networks in Drosophila*. Nature 631, 369–377 (2024).

- DOI: https://doi.org/10.1038/s41586-024-07600-z
- 论文: https://www.nature.com/articles/s41586-024-07600-z

该工作为翼前运动网络的模块化组织、运动神经元招募结构以及腿和翼之间的差异提供了依据。`virtual-fly`不会先把一组翼运动神经元平均后再选择动作，而是保留从单个运动神经元到外周肌肉的路径。

### Cheong et al. (2026)

**Cheong, H. S. J., Eichler, K., Stürner, T. et al.** *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*. eLife, version of record (2026).

- DOI: https://doi.org/10.7554/eLife.96084.3
- 论文: https://elifesciences.org/articles/96084

这是确认成年雄性腹神经索中下降神经元输入如何经过前运动回路到达运动神经元的重要来源。

### Ehrhardt et al. (2025・预印本)

**Ehrhardt, E., Whitehead, S. C. et al.** *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*. bioRxiv preprint, version 3 (2025).

- DOI: https://doi.org/10.1101/2023.05.31.542897
- 开放记录: https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/

该预印本作为辅助依据，用于解释翼前运动回路的细胞类型级组织，以及识别翼运动神经元和肌肉之间的关系。它会明确作为预印本处理，而不会伪装成已同行评审的最终论文。

## 4. 飞行肌肉、转向与全身力学

### Teoh et al. (2025・预印本)

**Teoh, H. K., Biswas, D., Leung, A. et al.** *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight*. bioRxiv preprint, version 2 (2025).

- DOI: https://doi.org/10.1101/2025.11.02.686144
- 开放记录: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/

该预印本用于解释间接转向肌tp1在飞行稳定中的作用，以及相关翼铰链力学。它同样与已同行评审论文分开标注。

### Vaxenburg et al. (2025)

**Vaxenburg, R., Siwanowicz, I., Merel, J. et al.** *Whole-body physics simulation of fruit fly locomotion*. Nature 643, 1312–1320 (2025).

- DOI: https://doi.org/10.1038/s41586-025-09029-4
- 论文: https://www.nature.com/articles/s41586-025-09029-4
- FlyBody仓库: https://github.com/TuragaLab/flybody — Apache-2.0
- 配套数据集: https://janelia.figshare.com/articles/dataset/25309105 — version 4，GPL 3.0+
- MuJoCo Menagerie中的FlyBody: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

这是解剖学详细的FlyBody全身模型与MuJoCo身体物理的主要科学依据。原论文为了展示步行和飞行使用了强化学习，但`virtual-fly`只采用身体几何、物理和执行结构，不把论文中的强化学习策略当作虚拟CNS。canonical v1 还会从 Figshare 配套数据集中单独获取实测基准翼拍模式；该数据集的 GPL 3.0+ 与 FlyBody 代码仓库的 Apache-2.0 是两个不同的许可边界。

## 5. 主要官方数据与外部软件

以下内容不能替代上面的科学论文，但它们是当前实现的重要官方数据源和依赖项。

- **MaleCNS** — https://male-cns.janelia.org/
- **neuPrint** — https://neuprint.janelia.org/
- **FlyBody** — https://github.com/TuragaLab/flybody
- **FlyGym / NeuroMechFly文档** — https://neuromechfly.org/
- **MuJoCo** — https://mujoco.org/
- **MuJoCo仓库** — https://github.com/google-deepmind/mujoco
- **MuJoCo Menagerie** — https://github.com/google-deepmind/mujoco_menagerie

FlyGym标准复眼`Retina`使用合成六角网格，因此不能直接当作MaleCNS个体中body ID与真实视网膜位置的观测对应。当前感觉边界优先使用MaleCNS的视网膜对应注释和公开连接。

## 6. 文献使用与来源记录规则

1. 优先使用一手论文和官方数据，而不是二手解说。
2. 明确记录每篇论文具体支撑哪一个实现边界，不能把一篇论文当作所有无关参数的统一依据。
3. 区分已同行评审论文、预印本、官方数据和软件文档。
4. 没有直接从文献测得的数值，应按情况标记为`inferred`、`assumed`或`calibrated`。
5. 数据集和外部软件按实验固定版本，不静默跟随上游更新。
6. 将实测源数据与会随时间变化的虚拟个体状态分开保存。
7. 不得把外部软件附带的强化学习策略或神经网络控制器悄悄替代虚拟CNS。
