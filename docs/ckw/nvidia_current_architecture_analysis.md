# aiconfigurator 当前架构与 NVIDIA 依赖分析

> 说明：本文基于当前代码仓库现状分析，重点梳理框架中和 NVIDIA GPU 绑定较深的模块，为后续适配华为昇腾 910B 系列提供输入。

## 1. 结论先说

当前 `aiconfigurator` 的整体架构虽然抽象成了：

- System spec（系统规格）
- Perf database / silicon data（性能数据库）
- SDK 估算链路（内存、延迟、调度/并行估算）
- Generator（部署配置生成）

但它的默认实现和大量默认假设，确实是以 **NVIDIA GPU + CUDA + NCCL + TensorRT-LLM / vLLM / SGLang** 为主线设计的。

换句话说：

- 框架不是“只支持 NVIDIA”
- 但它当前主要为 NVIDIA 系统做了深度建模和产品化支持
- 如果要支持昇腾 910B，需要做“平台级适配”，而不仅仅是“加一个新 GPU 型号”

## 2. 当前平台现状：基本只成熟支持 NVIDIA 系统

README 已经明确，当前支持矩阵主要覆盖以下 NVIDIA 系统：

- h100_sxm
- h200_sxm
- a100_sxm

以及一批 Estimate-only / 部分支持的 NVIDIA 型号：

- h100_pcie
- a100_pcie
- l4
- a30

同时 `systems/data/` 下实际存在的系统目录也基本都是 NVIDIA GPU 系统，例如：

- aic-core/src/aiconfigurator_core/systems/data/h200_sxm/
- aic-core/src/aiconfigurator_core/systems/data/h100_sxm/
- aic-core/src/aiconfigurator_core/systems/data/a100_sxm/

这意味着：当前产品的“真实性能数据库”和“默认估算模型”几乎完全围绕 NVIDIA GPU 构建。

## 3. 当前架构中和 NVIDIA 绑定最深的模块

### 3.1 System spec 已经 NVIDIA 化
每个系统规格 YAML 里定义了：GPU 显存、带宽、Tensor Core 算力、节点内互联、节点间互联、NCCL 相关内存、SM version。

对于昇腾 910B，这意味着至少 SM version、NCCL、NVLink/NVSwitch 拓扑、CUDA graph、Tensor Core 算力表达这些概念需要映射或重定义。

### 3.2 性能数据库高度 NVIDIA 化
当前 silicon data 目录结构是：`systems/data/<system>/<op_family>/<framework>/<version>/`

其中通信数据基本是 NCCL，说明当前通信模型、通信延迟建模、通信带宽估算默认绑定 NCCL。

如果要支持 910B，理论上应新增或替换为 HCCL，或昇腾实际使用的集合通信库。

### 3.3 Generator 默认模板默认面向 NVIDIA/CUDA
`src/aiconfigurator/generator/config/backend_templates/` 中的启动模板直接使用了 CUDA_VISIBLE_DEVICES、trtllm、vllm、sglang，以及 NVIDIA 常见部署链路相关参数。

`src/aiconfigurator/generator/facts/hardware.yaml` 已经存在平台级硬件 profile，说明 generator 已经具备一定“平台 profile 抽象能力”，但目前主要还是 NVIDIA 平台。

### 3.4 SDK 估算链路默认假设 NVIDIA 硬件语义
在 SDK 侧，内存和性能估算主要依赖 system_spec 里的 gpu mem_capacity、misc nccl_mem、misc other_mem、misc nccl_version。

也就是说，当前框架的内存与通信模型，默认是以 NVIDIA GPU + NCCL 为参照系建立的。

## 4. 当前架构的“可扩展点”

虽然当前是 NVIDIA 主线，但框架在设计上并非完全不可扩展，主要体现在以下几点：

- system spec 是可插拔的
- silicon data 路径是可扩展的
- generator 有 profile 抽象能力

## 5. 为什么说“适配 910B 不是简单加型号”

如果只是加一个 `--system ascend_910b`，技术上并不难。

难的是当前框架很多核心估算都依赖：NVIDIA 性能语义、CUDA 生态推理后端、NCCL 通信库、CUDA graph / CUDA runtime 特征、Tensor Core FLOPS 表达。

而昇腾 910B 生态通常是：CANN、HCCL、MindIE / MindSpore Serving / 其它昇腾推理栈、不依赖 CUDA、Tensor Core / SM 表达不直接等价。

因此真正适配不是“补一个 YAML”，而是“让平台模型从 NVIDIA 假设中抽离出来”。

## 6. 我对当前框架的整体判断

当前框架可以分为四层来看：

1. 数据采集层（collector）：当前主要围绕 NVIDIA + CUDA + trtllm/vllm/sglang 生态
2. 系统规格层（system spec）：当前字段名与定义已经 NVIDIA 化
3. 估算层（SDK）：内存/延迟/通信模型主要按 NVIDIA + NCCL 经验建模
4. 生成层（generator）：部署配置默认面向 NVIDIA GPU 推理栈

如果要支持昇腾 910B，我认为正确的做法不是硬塞进现有 NVIDIA 语义，而是先做一次“平台抽象层增强”，再补齐昇腾数据。

## 7. 小结

当前 `aiconfigurator` 整体架构是：

- 可扩展
- 但当前深度绑定 NVIDIA GPU 生态
- 已有系统规格、硅基性能数据、通信数据、模板、硬件 profile 都以 NVIDIA 为主要假设
- 适配昇腾 910B 属于“新增一个异构平台”的工作量，而不是“新增一个 NVIDIA 子型号”

后续适配重点应放在：system spec 语义扩展、通信库从 NCCL 扩展到 HCCL、backend 从 CUDA 生态扩展到昇腾推理栈、generator profile 增加 Ascend 硬件抽象、silicon data / estimation model 补齐 910B 实测数据。