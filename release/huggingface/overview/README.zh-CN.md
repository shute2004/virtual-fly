# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` 是一个研究项目：以成年雄性果蝇公开的 MaleCNS 连接组作为初始状态，让神经活动、神经调制和局部突触可塑性随时间演化，并与 FlyBody / MuJoCo 身体及物理环境形成闭环。

这个 Hugging Face 仓库用于介绍 `virtual-fly`，并说明首次发布前已经准备好的检查点仓库结构。源代码、科学说明、canonical 实验清单以及完整复现脚本位于 GitHub：

- GitHub: https://github.com/shute2004/virtual-fly
- 英文文档: https://github.com/shute2004/virtual-fly/tree/main/docs/en
- 日本語 README: https://github.com/shute2004/virtual-fly/blob/main/README.ja.md
- 简体中文 README: https://github.com/shute2004/virtual-fly/blob/main/README.zh-CN.md

## 检查点仓库结构

准备发布的检查点分为两个明确不同的系谱。下面四个检查点仓库目前尚未创建或公开。**不要把视频用 Before / After 检查点当作 canonical v1 的初始 / 学习后检查点。**

### Canonical v1

canonical 系列仓库名中的日期只在实际发布日期确定后填写。

| 检查点 | 仓库 | 含义 |
|---|---|---|
| 初始 | `shute2004/virtual-fly-initial-YYYYMMDD` | canonical v1 的初始检查点，global weight version 0 |
| 学习后 | `shute2004/virtual-fly-trained-YYYYMMDD` | canonical v1 完成 6 个训练回合后的检查点，global weight version 6 |

canonical v1 的科学计算部分固定在 Git commit `7fa464aad7269d34f46f1171080b51e095d1d811`。这次较短的 canonical 实验中，25,582,938 条存储边里有 2,163,179 条发生了权重变化。但在关闭可塑性和任务触发 DAN 刺激的固定评估中，初始状态和学习后状态的分类结果相同。

因此 canonical v1 证明的是：**在当前局部可塑性定义下，保存的 MaleCNS 权重确实发生了变化**。它并不声称已经证明了明确的行为改善、泛化能力或长期学习稳定性。

### 历史 Before / After 视频

| 检查点 | 仓库 | 来源角色 |
|---|---|---|
| Before | `shute2004/virtual-fly-video-before` | 公开 Before / After 对比视频中 Before 一侧实际使用的历史检查点 |
| After | `shute2004/virtual-fly-video-after` | 同一视频中 After 一侧实际使用的历史检查点 |

这些检查点的内部开发版本分别是 `v240` 和 `v966`，但这些编号不会出现在公开仓库名中，只保留在来源记录里。Before 一侧的 v240 不是未经学习的初始状态，而是已经经历 240 个历史训练回合的检查点；回放不是留出评估，而且在关闭可塑性时仍保留了任务事件触发的 PAM 刺激。v966 也来自混合了多个早期条件的开发系。因此，它们**不是 canonical v1 的实验依据**。

## Hugging Face 中保存什么

正式发布后，每个检查点仓库只保存检查点本体，以及直接需要的来源记录、哈希值、归属信息和模型卡。它们不会复制整个研究工作区。

默认不保存：

- MaleCNS 原始数据
- 完整的标准化 MaleCNS 快照
- FlyBody / FlyGym / MuJoCo 源码或身体资源
- 完整轨迹目录
- 完整渲染帧目录
- 与本次公开无关的开发中检查点
- 构建缓存或虚拟环境

对于上游数据和由其生成的静态数据，MaleCNS 官方来源与 GitHub 上的 canonical 复现脚本是正式的获取和重建路径。

## 许可证与归属

`virtual-fly` 原创源代码和文档使用 MIT License。检查点权重以 MaleCNS 官方站点按 CC BY 4.0 发布的 `male-cns:v1.0` 为基础，因此检查点仓库标记为 `cc-by-4.0`，并明确保留 MaleCNS 归属信息。

FlyBody、FlyGym 和 MuJoCo 继续适用各自的上游许可证，不会仅为方便而复制进检查点仓库。

更详细的许可证边界见 GitHub 仓库中的 `THIRD_PARTY_NOTICES.md`。

## 复现

当前 canonical 实验和完整的源码级来源记录以 GitHub 为准：

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
bash canonical/canonical-v1/reproduce.sh
```

Hugging Face 上的各检查点仓库用于分发检查点文件。canonical v1 的科学定义仍以 GitHub 中的 canonical 包为准。
