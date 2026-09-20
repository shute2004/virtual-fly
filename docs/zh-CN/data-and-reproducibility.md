# 数据来源与可重现性

[English](../en/data-and-reproducibility.md) · [日本語](../ja/data-and-reproducibility.md) · [简体中文](data-and-reproducibility.md)

## 1. 基准数据

初始基准采用 HHMI Janelia 等机构公开的成年雄性果蝇全中枢神经系统连接组 `male-cns:v1.0`。

官方同时提供 neuPrint API 和批量下载。数据集本体采用 CC-BY 许可。

获取数据时至少记录：

- 数据集名称；
- 数据集版本；
- 来源 URL；
- 获取日期；
- 上游数据许可；
- 上游引用信息；
- 各文件哈希值；
- 转换工具版本；
- 转换配置。

## 2. 不直接修改原始数据

外部数据作为只读的源快照处理。

```text
上游原始数据
      ↓
已验证的本地快照
      ↓
归一化后的静态内部数据
      ↓
每个虚拟个体自己的可变状态
```

学习和可塑性产生的变化不得写回原始连接组。

## 3. 数据分层

### A 层：上游数据

下载得到的原始文件或官方 API 返回结果。

### B 层：归一化后的静态数据

转换为项目内部格式的只读数据，例如：

- 神经元列表；
- 突触或聚合连接；
- 注释；
- 形态数据引用；
- 来源记录。

### C 层：虚拟个体状态

每个虚拟个体都可以独立变化的状态：

- 神经状态；
- 突触功能状态；
- 结构变化差分；
- 神经调制状态；
- 身体状态；
- 实验历史。

## 4. 来源类别

值、参数和连接在可行时应标注来源类别。

| 类别 | 含义 |
|---|---|
| `observed` | 直接实测或上游数据中直接存在 |
| `literature` | 从文献采用 |
| `inferred` | 根据实测信息推断 |
| `assumed` | 为填补未知部分而作出的假设 |
| `calibrated` | 通过数值或实验校准得到 |

论文、README 和可视化中不得混淆这些类别。

## 5. 每次实验的来源记录

每次实验都应生成机器可读的来源记录。例如：

```yaml
experiment_id: exp-...
fly_id: fly-...
parent_checkpoint: ...
code_commit: ...
connectome:
  dataset: male-cns:v1.0
  normalized_snapshot_hash: ...
neural_model:
  name: ...
  version: ...
plasticity_model:
  name: ...
  version: ...
body:
  model: flybody
  version: ...
environment:
  name: flyppy
  config_hash: ...
rng:
  seed: ...
```

这些英文标识是机器可读字段名，人类阅读的说明文字不需要模仿这种写法。

## 6. 可重现性等级

### R0：可以准确确定实验条件

能够确定输入数据、配置和代码版本。

### R1：在同一执行环境中确定性重现

使用相同执行方式和相同环境时，可以重现相同结果。

### R2：不同执行方式之间保持一致

CPU、GPU、WebGPU 等不同执行方式在规定误差范围内得到等价结果。

### R3：统计意义上的重现

即使存在非确定性，多次运行仍支持相同的统计结论。

并非所有实验都必须满足 R2，但每次实验应明确说明达到哪个等级。

## 7. 外部资源

默认不把外部资源直接纳入本仓库。主要包括：

- MaleCNS；
- FlyBody；
- FlyGym / NeuroMechFly；
- MuJoCo。

确有需要时，应提供获取脚本并固定所使用的版本。

## 8. 大型文件

以下文件不纳入 Git 管理：

- 原始连接组数据；
- 大型 EM 数据；
- 大型网格模型；
- 检查点；
- 轨迹数据；
- 渲染视频；
- 详细性能分析记录。

Git 中只保留重新获取或重新生成所需的小型来源记录、配置、哈希值和摘要报告。

## 9. Hugging Face 检查点发布

供第三方下载的检查点文件与 GitHub 源码仓库分开发布。Hugging Face 的 `shute2004/virtual-fly` 项目概览页面链接到四个已验证的检查点仓库。

发布结构明确区分四类仓库：

- canonical v1 初始检查点：[`virtual-fly-initial-20260920`](https://huggingface.co/shute2004/virtual-fly-initial-20260920)
- canonical v1 训练后检查点：[`virtual-fly-trained-20260920`](https://huggingface.co/shute2004/virtual-fly-trained-20260920)
- 公开视频 Before 一侧的历史检查点：[`virtual-fly-video-before`](https://huggingface.co/shute2004/virtual-fly-video-before)
- 公开视频 After 一侧的历史检查点：[`virtual-fly-video-after`](https://huggingface.co/shute2004/virtual-fly-video-after)

canonical 系列的发布日期为 2026 年 9 月 20 日。视频用历史检查点即使能被同一运行系统读取，也不能解释为 canonical v1 的初始检查点或训练后检查点。

每个检查点仓库只包含原生检查点目录，以及 `README.md`、归属说明、来源记录和 SHA-256 列表。除非直接用于使用或验证该检查点，否则不上传完整轨迹、全部视频帧、原始数据、无关检查点或构建产物。

检查点权重来源于 MaleCNS `male-cns:v1.0`，官方来源以 CC BY 4.0 发布。MaleCNS 原始数据不会复制到 Hugging Face，而是从官方来源或 canonical 获取／复现流程中取得。FlyBody / FlyGym / MuJoCo 资源也不会仅为了方便而复制进检查点仓库。

已准备好的模型卡、精确检查点哈希和必要文件清单记录在 [`../../release/huggingface/`](../../release/huggingface/README.md)。

## 10. 历史资料中的本地路径

少量历史开发记录和历史报告仍保留原始运行、异常记录或报告生成时使用的绝对本地路径结构。首次公开前，为保护隐私，我们仅将其中恰好14个文件里的本地操作系统用户名替换为`<local-user>`；除此之外，周围路径、历史结果数值和科学内容均未改动。原始blob与公开blob的对应关系记录在[`../../canonical/canonical-v1/privacy-redaction-provenance.json`](../../canonical/canonical-v1/privacy-redaction-provenance.json)中，且不会暴露被删去的用户名。这些路径不是当前代码的依赖，也不是安装要求；当前公开文档和执行命令使用仓库相对路径或可移植路径。

## 11. 公开浏览器实验

浏览器返回的结果应视为位于可信执行边界之外。

每个结果至少应包含：

- 实验包版本；
- 初始状态哈希；
- 配置哈希；
- 执行方式；
- 结果哈希；
- 摘要指标。

重要结果应在服务器侧或多个独立客户端上重新执行确认。

## 12. 个人数据

公开实验不应收集神经科学实验所不需要的个人信息或浏览器识别信息。

如果分布式计算确实需要执行环境信息，也必须明确说明用途和保存范围。
