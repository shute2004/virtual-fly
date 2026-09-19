# 代码结构

[English](../en/code-structure.md) · [日本語](../ja/code-structure.md) · [简体中文](code-structure.md)

本文说明当前实现中面向开发者的职责边界。它并不意味着历史成果物都是由当前代码生成的。

## 当前训练执行路径

```text
scripts/dev/train_flyppy_population.sh
  -> scripts/embodiment/train_flyppy_population_packed.py   （轻量CLI层）
  -> virtual_fly.training.population_packed                 （执行方式选择与准备）
  -> virtual_fly.training.population_process                （实验阶段整体调度）
       -> virtual_fly.training.flyppy_config                （CLI、配置与来源验证）
       -> virtual_fly.training.curriculum*                  （只负责初始条件调度）
       -> virtual_fly.training.checkpointing                （只保存共享权重）
       -> virtual_fly.runtime.neural_bridge                 （与Rust神经运行系统通信）
       -> virtual_fly.runtime.packed_*                      （MuJoCo进程边界）
            -> virtual_fly.embodiment.factory               （构建身体与感觉层）
                 -> virtual_fly.embodiment.body             （带版本的FlyBody连接层）
                 -> virtual_fly.embodiment.retina/haltere   （感觉转换）
                 -> virtual_fly.embodiment.periphery        （单个运动神经元的外周状态）
```

`train_flyppy_v3_population.sh` 这个文件名只为兼容旧入口而保留。当前通用入口是 `train_flyppy_population.sh`。

## 科学兼容性约定

`virtual_fly.semantics` 为会影响兼容性的规则提供明确标识，例如：

- MaleCNS / class-DAN 的定义；
- 神经活动与可塑性规则版本；
- 多实例训练中检查点和神经步数的含义；
- 平衡器相关派生产物的类别；
- 直接射线视觉所使用的 K 值；
- 身体版本及其非线性的历史继承关系。

`virtual_fly.reproducibility` 负责验证和记录这些约定，并不定义另一套学习规则。

## 身体版本谱系

身体版本号是历史标签，不是简单的线性继承关系：

```text
v3
└─ v4
   ├─ v5
   ├─ v6
   └─ v7

v8 = v6 + v7 的 neutral trim
```

实现主体位于 `virtual_fly.embodiment.body`。`scripts/embodiment/flybody_v*_adapter.py` 中的兼容文件只是重新导出包内实现。

## 运行时处理与纯配置逻辑

- `virtual_fly.embodiment`：身体模型、环境、感觉器官、外周状态；
- `virtual_fly.training`：经历条件调度、共享权重训练、保存与恢复；
- `virtual_fly.runtime`：子进程、GPU、观察数据等带副作用的边界；
- `virtual_fly.playback`：固定条件下的回放与评估；
- `virtual_fly.reporting`：稳定的机器可读报告与历史记录；
- `virtual_fly.reproducibility`：哈希、来源与规则验证。

在可行的情况下，纯调度和配置函数会与 MuJoCo、GPU 和子进程生命周期分离，这样可以在不启动模拟器的情况下进行单元测试。

## 历史实现与诊断代码

`virtual_fly.legacy` 保存当前执行路径已经不使用的历史机制。DNg02 群体解码器是目前最明确的例子。

为了重现实验或调查旧结果，`scripts/analysis` 和 `scripts/embodiment` 中仍保留一些历史及诊断脚本。其中不少仍会导入旧脚本模块名，这些名字只作为兼容层存在，不应成为新实现的依赖入口。

主要例子：

- `train_flyppy_curriculum.py`：历史串行训练器；
- `train_flyppy_population.py`：同一进程内运行的参考/诊断用多实例训练器；
- `evaluate_flyppy.py`：历史或诊断用评估器；
- `scripts/analysis/*`：探针、审计、基准测试和报告导出，不是行为控制器。

DLM / DVM 聚合表示以及当前身体连接中仍然存在的其他校准/手写外周力学，不会仅仅因为是工程近似就被归类为历史实现。它们仍属于当前物理规则的一部分。

## Python 与 Rust 的边界

Python 负责实验调度、身体集成、感觉转换、经历条件调度和来源记录。Rust 负责大型 MaleCNS 神经运行系统，包括神经动力学、神经调制、资格度和权重更新。

二者通过指定 body ID 的电流与事件，以及明确的检查点和更新协议进行通信。可视化只读。

## 身体相关派生产物

当前执行路径生成的身体相关派生产物不能覆盖历史实验曾经使用的文件。

尤其是中立配平文件被明确分开：

- 历史 v7：`artifacts/embodiment/wing-pattern-neutral-trim-v1.*`
- 当前通过来源验证的版本：`artifacts/derived/wing-pattern-neutral-trim-v1.*`

只有当当前版本需要的两个文件都不存在时，`train_flyppy_population.sh` 才会生成它们，并通过类型化身体配置明确传入路径。重放历史实验时，可以在显式允许历史成果物的条件下指定旧文件。

v7 的配平数值本身没有改变。路径分离的目的不是改变身体力学，而是保护来源信息。
