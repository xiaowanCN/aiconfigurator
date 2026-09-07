基于2026.09.02为最新拉取分支分析：

# 昇腾 910B 系列适配指南

> 从 NVIDIA 默认框架升级为多厂商平台估算框架的详细改造方案。

## 1. 引言

### 1.1 文档目的

本文档说明如何将 aiconfigurator 适配到华为昇腾 910B2/910B4 系列硬件。适配的本质不是简单新增一个 `--system ascend_910b`，而是让框架从"以 NVIDIA 为主线"演进为"可支持多厂商加速卡的平台估算框架"。

### 1.2 适配目标

- **短期目标**：支持 estimate-only 模式的昇腾 910B 性能估算（无需实测数据）
- **中期目标**：接入 MindIE 推理栈，支持昇腾后端的部署配置生成
- **长期目标**：完整的 Collector 数据收集 + 实测校准闭环

## 2. 昇腾 910B 硬件特性分析

### 2.1 硬件规格对比表

| 规格 | B300 SXM (NVIDIA) | 910B2 (昇腾) | 910B4 (昇腾) |
|------|-------------------|--------------|--------------|
| **工艺** | 4nm (TSMC) | 7nm (SMIC) | 7nm (SMIC) |
| **HBM** | HBM3e 288GB | HBM2e 64GB | HBM2e 64GB |
| **显存带宽** | 7.75 TB/s | ~1.6 TB/s | ~1.6 TB/s |
| **FP16 算力** | 4.5 PFLOPS | 320 TFLOPS | 400 TFLOPS |
| **FP32 算力** | N/A | 80 TFLOPS | 100 TFLOPS |
| **INT8 算力** | 153.5 TOPS | 640 TOPS | 800 TOPS |
| **互联** | NVLink 5 (900 GB/s) | HCCS 2.0 (~56 GB/s) | HCCS 3.0 (~100 GB/s) |
| **节点间** | CX8 InfiniBand (100 GB/s) | RoCE / HCCS | RoCE / HCCS |
| **节点内 GPU 数** | 8 | 8 | 8 |
| **功耗** | 1100W | 400W | 450W |
| **SM 版本** | sm100 | N/A (Ascend Core) | N/A (Ascend Core) |
| **通信库** | NCCL 2.27 | HCCL | HCCL |
| **推理框架** | TRT-LLM / vLLM / SGLang | MindIE | MindIE |
| **编程模型** | CUDA | CANN (Ascend Computing Language) | CANN |

### 2.2 关键差异分析

#### 2.2.1 无 NVLink / NVSwitch

昇腾 910B 使用 HCCS（Huawei Collective Communication Switch）替代 NVLink：
- HCCS 2.0 带宽约 56 GB/s（910B2），HCCS 3.0 约 100 GB/s（910B4）
- 相比 NVLink 5 的 900 GB/s 有数量级差距
- **影响**：通信密集型操作（AllReduce、AllToAll）延迟显著增加

#### 2.2.2 无 FP8 / FP4 原生支持

910B 系列不支持 FP8/FP4 Tensor Core：
- 仅支持 FP16/BF16/INT8 矩阵运算
- **影响**：无法使用 FP8/FP4 量化加速，INT8 成为唯一低精度选项

#### 2.2.3 无 CUDA / NCCL

昇腾使用完全不同的软件栈：
- **CANN** 替代 CUDA（算子编译器 + 运行时）
- **HCCL** 替代 NCCL（集合通信库）
- **MindIE** 替代 TRT-LLM/vLLM/SGLang（推理框架）
- **影响**：所有 CUDA 特定假设需要替换

#### 2.2.4 内存模型差异

- 昇腾的内存管理不同于 CUDA（无 `cudaMalloc` 语义）
- HCCL 内存开销模型不同于 NCCL
- CANN 运行时内存开销不同于 CUDA/cuBLAS
- **影响**：`_get_memory_usage()` 中的内存估算公式需要重写

### 2.3 对估算模型的影响

| 估算维度 | 影响程度 | 说明 |
|----------|----------|------|
| GEMM 延迟 | 高 | 算力表达不同，需要昇腾的矩阵运算基准 |
| 注意力延迟 | 高 | CANN 算子实现不同于 CUDA kernel |
| MoE 延迟 | 高 | 通信带宽差异导致 AllToAll 延迟剧增 |
| 内存估算 | 高 | HCCL 内存、CANN 运行时开销需要实测 |
| 通信延迟 | 高 | HCCS 带宽远低于 NVLink |
| KV Cache | 中 | 量化方式可能不同（INT8 vs FP8） |
| 投机解码 | 低 | 框架层差异，与硬件关系较小 |

## 3. 现有架构适配性分析

### 3.1 可复用的抽象

| 抽象 | 位置 | 复用性 |
|------|------|--------|
| **SystemSpec** | `systems/*.yaml` | 高：YAML schema 可直接扩展 |
| **PerfDatabase** | `perf_database.py` | 高：parquet 数据格式通用 |
| **Backend 工厂** | `backends/factory.py` | 高：可新增 MindIE 后端 |
| **Sweep 框架** | `sweep.py` | 高：并行枚举逻辑通用 |
| **Pareto 分析** | `picking.py` | 高：择优逻辑与硬件无关 |
| **Generator 框架** | `generator/` | 中：需要新增模板和规则 |
| **Collector 框架** | `collector/` | 低：高度依赖 CUDA 生态 |

### 3.2 需要改造的硬编码

| 硬编码 | 位置 | 改造内容 |
|--------|------|----------|
| `sm_version` | `systems/*.yaml` | 替换为 vendor/arch 字段 |
| `nccl_mem` | `systems/*.yaml`, `memory.py` | 抽象为 `comm_mem` |
| `nccl_version` | `systems/*.yaml` | 抽象为 `comm_version` |
| `CUDA_VISIBLE_DEVICES` | `generator/templates/` | 替换为设备发现机制 |
| `nccl` 引用 | `operations/communication.py` | 抽象为 `comm_backend` |
| NVIDIA 算力语义 | `systems/*.yaml` | 扩展为 vendor-neutral 表达 |

### 3.3 B60 (Intel XPU) 适配模式参考

项目已有的 B60 适配提供了重要参考：

```yaml
# systems/b60.yaml
data_dir: data/b60
gpu:
  mem_bw: 456000000000
  mem_capacity: 25769803776
  bfloat16_tc_flops: 98304000000000
  int8_tc_flops: 197000000000000
  fp8_tc_flops: 98304000000000  # B60 无原生 FP8，回退到 BF16
node:
  intra_node_bw: 28400000000   # PCIe 无 NVLink
misc:
  nccl_mem: {...}
  nccl_version: '2.23'
  oneccl_version: '2021.15.9'   # Intel oneCCL 版本
```

B60 的适配模式：
- `fp8_tc_flops` 回退到 BF16 值（无原生 FP8）
- `intra_node_bw` 使用 PCIe 带宽（无 NVLink）
- 新增 `oneccl_version` 字段
- Collector 中有 `_xpu` 后缀的收集器（`collect_gemm_xpu.py`）

## 4. 适配方案设计

### 4.1 总体策略

采用**平台抽象增强 + 昇腾增量补齐**的策略：

```
Phase 1: 平台抽象层 + 最小闭环
  → 系统 YAML 扩展、vendor 字段、estimate-only 入口

Phase 2: 推理栈适配 + 数据补齐
  → MindIE 后端、HCCL 通信、Collector 扩展

Phase 3: 产品化闭环
  → 实测数据校准、Generator 模板、支持矩阵
```

### 4.2 Phase 1: 平台抽象层 + 最小闭环

**目标**：让 `aiconfigurator cli default --system ascend_910b2 --backend mindie ...` 能跑通，输出基于算力公式估算的结果。

**范围**：
- 新增 `ascend_910b2.yaml` 和 `ascend_910b4.yaml`
- 扩展 SystemSpec 字段支持 vendor/arch
- 创建 estimate-only 的 PerfDatabase（无实测数据，纯公式估算）
- 新增 MindIE 后端骨架

### 4.3 Phase 2: 推理栈适配 + 数据补齐

**目标**：MindIE 后端能生成正确的部署配置，Collector 能在昇腾设备上收集数据。

**范围**：
- HCCL 通信模型实现
- MindIE 后端完整实现
- Generator 昇腾模板和硬件 profile
- Collector 昇腾设备探测

### 4.4 Phase 3: 产品化闭环

**目标**：完整的实测数据校准 + 支持矩阵 + 文档。

**范围**：
- 昇腾设备上的 Collector 数据收集
- 估算模型校准（silicon data + regression）
- Support Matrix 扩展
- 用户文档

## 5. 详细改造步骤

### Epic A: 系统语义与规格层

**目标**：新增昇腾系统 YAML，扩展字段支持多厂商。

#### A.1 新增 ascend_910b2.yaml

```yaml
# systems/ascend_910b2.yaml
data_dir: data/ascend_910b2  # estimate-only: 无实测数据时留空
vendor: huawei                # 新增: 厂商标识
arch: ascend_v0               # 新增: 架构标识

gpu:
  mem_bw: 1600000000000       # 1.6 TB/s HBM2e
  mem_capacity: 68719476736   # 64GB
  bfloat16_tc_flops: 320000000000000  # 320 TFLOPS
  int8_tc_flops: 640000000000000      # 640 TOPS
  # fp8_tc_flops: 不支持，省略
  # fp4_tc_flops: 不支持，省略
  power: 400
  # sm_version: 不适用，用 arch 替代

node:
  num_gpus_per_node: 8
  inter_node_bw: 50000000000  # RoCE 50GB/s
  intra_node_bw: 56000000000  # HCCS 2.0 ~56GB/s
  pcie_bw: 64000000000        # PCIe Gen5 x16
  p2p_latency: 0.00001        # 10μs (需实测校准)

misc:
  comm_backend: hccl          # 新增: 通信库标识
  comm_mem:                   # 替代 nccl_mem
    1: 0
    2: 268435456              # 256MB (需实测校准)
    4: 314572800              # 300MB
    8: 314572800
  other_mem: 2147483648       # 2GB CANN 运行时 (需实测校准)
  comm_version: '1.0'         # HCCL 版本
```

#### A.2 SystemSpec 字段扩展

在 `system_spec.py` 中添加 vendor/arch 支持：

```python
@dataclass
class SystemSpec:
    # 现有字段...
    vendor: str = "nvidia"     # 新增: nvidia/huawei/intel/amd
    arch: str = ""             # 新增: sm100/ascend_v0/...
```

在 `is_*_system()` 辅助函数中添加：

```python
def is_ascend_system(system_name: str) -> bool:
    spec = load_system_spec(system_name)
    return spec.get("vendor") == "huawei"
```

#### A.3 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `systems/ascend_910b2.yaml` | 新增 | 910B2 系统规格 |
| `systems/ascend_910b4.yaml` | 新增 | 910B4 系统规格 |
| `sdk/system_spec.py` | 修改 | 添加 vendor/arch 字段 |
| `sdk/perf_database.py` | 修改 | `load_system_spec()` 支持新字段 |
| `sdk/common.py` | 修改 | 添加 `is_ascend_system()` |

### Epic B: 通信模型层

**目标**：将通信模型从 NCCL 扩展为可插拔的 comm_backend。

#### B.1 comm_backend 抽象

在 `operations/communication.py` 中抽象通信后端：

```python
class CommunicationBackend(Enum):
    NCCL = "nccl"
    HCCL = "hccl"
    ONECCL = "oneccl"

def get_comm_backend(system_spec: dict) -> CommunicationBackend:
    backend_str = system_spec.get("misc", {}).get("comm_backend", "nccl")
    return CommunicationBackend(backend_str)
```

#### B.2 HCCL 通信模型

HCCL 的 AllReduce/AllToAll 实现不同于 NCCL：
- 使用 Ring 算法而非 NVLS/Tree
- 无 NVLink 直连，依赖 HCCS 交换
- 内存模型不同

```python
def get_hccl_allreduce_latency(
    message_size: int,
    num_gpus: int,
    intra_node_bw: float,
    inter_node_bw: float,
) -> float:
    """HCCL AllReduce 延迟估算"""
    # Ring AllReduce: 2*(n-1)/n * message_size / bandwidth
    ring_factor = 2 * (num_gpus - 1) / num_gpus
    if num_gpus <= 8:  # 节点内
        return ring_factor * message_size / intra_node_bw
    else:  # 跨节点
        return ring_factor * message_size / inter_node_bw
```

#### B.3 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `operations/communication.py` | 修改 | 抽象 comm_backend，新增 HCCL 模型 |
| `operations/moe_comm.py` | 修改 | MoE AllToAll 支持 HCCL |
| `memory.py` | 修改 | `_get_memory_usage()` 使用 `comm_mem` 替代 `nccl_mem` |
| `systems/ascend_910b2.yaml` | 修改 | `misc.comm_mem` 替代 `misc.nccl_mem` |

### Epic C: Perf Database 层

**目标**：支持昇腾系统的性能数据加载，包括 estimate-only 模式。

#### C.1 数据目录骨架

```
systems/data/ascend_910b2/
├── attention/
│   └── mindie/
│       └── 1.0.0/
│           ├── context_attention.parquet   # 空或少量数据
│           └── generation_attention.parquet
├── gemm/
│   └── mindie/
│       └── 1.0.0/
│           └── gemm.parquet
├── moe/
│   └── mindie/
│       └── 1.0.0/
│           └── moe.parquet
└── comm/
    └── hccl/
        └── 1.0.0/
            └── all_reduce.parquet
```

#### C.2 estimate-only 模式

当性能数据目录为空时，回退到基于算力的公式估算：

```python
def get_fallback_latency(op_type, system_spec, shape):
    """无实测数据时的公式估算"""
    if op_type == "gemm":
        flops = system_spec["gpu"]["bfloat16_tc_flops"]
        # GEMM 延迟 = 2 * M * N * K / flops
        return 2 * shape["m"] * shape["n"] * shape["k"] / flops
    elif op_type == "attention":
        # 基于带宽的估算
        bw = system_spec["gpu"]["mem_bw"]
        return shape["bytes"] / bw
    # ...
```

#### C.3 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `systems/data/ascend_910b2/` | 新增 | 空数据目录骨架 |
| `perf_database.py` | 修改 | estimate-only 模式支持 |
| `operations/base.py` | 修改 | 添加公式估算兜底 |

### Epic D: SDK 估算层

**目标**：让内存和延迟估算支持昇腾硬件特性。

#### D.1 算力字段抽象

当前系统 YAML 使用 NVIDIA 特定的算力字段（`fp8_tc_flops`）。需要扩展为更通用的表达：

```python
def get_best_flops(system_spec, quant_mode):
    """根据量化模式选择最佳算力"""
    gpu = system_spec["gpu"]
    if quant_mode == "fp8" and "fp8_tc_flops" in gpu:
        return gpu["fp8_tc_flops"]
    elif quant_mode == "int8" and "int8_tc_flops" in gpu:
        return gpu["int8_tc_flops"]
    elif quant_mode in ("bf16", "fp16"):
        return gpu["bfloat16_tc_flops"]
    # 回退
    return gpu.get("bfloat16_tc_flops", gpu.get("float32_tc_flops"))
```

#### D.2 内存估算适配

替换 NCCL 特定的内存估算：

```python
def _get_memory_usage(self, model, database, ...):
    # ... 现有权重/激活/KV cache 估算 ...

    # 通信库内存 (替换 nccl_mem 引用)
    comm_mem_key = "comm_mem" if "comm_mem" in spec["misc"] else "nccl_mem"
    comm_mem = spec["misc"][comm_mem_key].get(tp_size, 0)

    # 运行时内存 (不再假设 CUDA)
    other_mem = spec["misc"]["other_mem"]

    total = weights + activations + kvcache + comm_mem + other_mem
    return total
```

#### D.3 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `sdk/memory.py` | 修改 | 内存估算支持 comm_mem |
| `backends/base_backend.py` | 修改 | `_get_memory_usage()` 适配 |
| `backends/mindie_backend.py` | 新增 | MindIE 后端实现 |
| `backends/factory.py` | 修改 | 注册 MindIE 后端 |

### Epic E: Generator 层

**目标**：Generator 能生成昇腾设备的部署配置。

#### E.1 Hardware Profile

在 `generator/facts/hardware.yaml` 中添加昇腾 profile：

```yaml
profiles:
  # ... 现有 NVIDIA profiles ...

  ascend_910b2:
    node_selector:
      accelerator: huawei-ascend-910b
    arch: aarch64
    nccl_env: {}  # 不适用
    shared_memory:
      default: 32Gi
    tolerations:
      - key: accelerator
        operator: Equal
        value: huawei-ascend-910b
        effect: NoSchedule
    device_env:
      ASCEND_VISIBLE_DEVICES: "all"
```

#### E.2 设备环境变量

替换 CUDA 特定的环境变量：

| NVIDIA | 昇腾替代 |
|--------|----------|
| `CUDA_VISIBLE_DEVICES` | `ASCEND_VISIBLE_DEVICES` |
| `NCCL_CUMEM_ENABLE` | 不适用 |
| `NCCL_P2P_LEVEL` | 不适用 |
| `NCCL_MNNVL_ENABLE` | 不适用 |

#### E.3 MindIE 模板

创建 `generator/config/backend_templates/mindie/`：

```
backend_templates/mindie/
├── cli_args.j2           # MindIE 启动参数
├── run.sh.j2             # 启动脚本
└── k8s_deploy.yaml.j2    # K8s 部署清单
```

#### E.4 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `generator/facts/hardware.yaml` | 修改 | 添加昇腾 profile |
| `generator/config/backend_templates/mindie/` | 新增 | MindIE 模板 |
| `generator/config/backend_config_mapping.yaml` | 修改 | 添加 MindIE 参数映射 |
| `generator/rule_plugin/mindie.rule` | 新增 | MindIE 计算规则 |
| `generator/config/deployment_config.yaml` | 修改 | 添加昇腾特定参数 |

### Epic F: Collector 层

**目标**：Collector 能在昇腾设备上探测和收集性能数据。

#### F.1 设备探测扩展

```python
# collector/capabilities.py
def detect_device_vendor():
    """检测设备厂商"""
    try:
        import torch
        if torch.cuda.is_available():
            return "nvidia"
    except ImportError:
        pass

    try:
        import torch_npu  # 昇腾 PyTorch 适配层
        if torch_npu.npu.is_available():
            return "huawei"
    except ImportError:
        pass

    return "unknown"
```

#### F.2 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `collector/capabilities.py` | 修改 | 昇腾设备探测 |
| `collector/collect.py` | 修改 | 昇腾收集路径 |
| `collector/mindie/` | 新增 | MindIE 算子收集器 |
| `collector/network/collect_hccl.py` | 新增 | HCCL 通信收集 |

### Epic G: Support Matrix 与文档

**目标**：支持矩阵覆盖昇腾系统。

#### G.1 改造文件清单

| 文件 | 改造类型 | 说明 |
|------|----------|------|
| `systems/support_matrix/` | 修改 | 添加昇腾系统行 |
| `tools/support_matrix/` | 修改 | 生成逻辑支持多厂商 |
| `docs/` | 修改 | 用户文档更新 |

## 6. 代码改造清单

### 6.1 需要新增的文件

| 文件路径 | 优先级 | 说明 |
|----------|--------|------|
| `systems/ascend_910b2.yaml` | P0 | 910B2 系统规格 |
| `systems/ascend_910b4.yaml` | P0 | 910B4 系统规格 |
| `systems/data/ascend_910b2/` | P1 | 数据目录骨架 |
| `backends/mindie_backend.py` | P1 | MindIE 后端 |
| `generator/config/backend_templates/mindie/` | P1 | MindIE 模板 |
| `generator/rule_plugin/mindie.rule` | P1 | MindIE 规则 |
| `collector/mindie/` | P2 | MindIE 收集器 |
| `collector/network/collect_hccl.py` | P2 | HCCL 收集 |

### 6.2 需要修改的文件

| 文件路径 | 优先级 | 改造内容 |
|----------|--------|----------|
| `sdk/system_spec.py` | P0 | vendor/arch 字段 |
| `sdk/common.py` | P0 | is_ascend_system() |
| `operations/communication.py` | P0 | HCCL 通信模型 |
| `memory.py` | P0 | comm_mem 替代 nccl_mem |
| `backends/factory.py` | P1 | 注册 MindIE |
| `backends/base_backend.py` | P1 | 内存估算适配 |
| `perf_database.py` | P1 | estimate-only 模式 |
| `generator/facts/hardware.yaml` | P1 | 昇腾 profile |
| `generator/config/backend_config_mapping.yaml` | P1 | MindIE 映射 |
| `collector/capabilities.py` | P2 | 昇腾设备探测 |
| `systems/support_matrix/` | P2 | 昇腾系统行 |

### 6.3 改造优先级排序

```
P0 (最小闭环): 系统 YAML + vendor 字段 + HCCL 模型 + 内存适配
P1 (推理栈):   MindIE 后端 + Generator 模板 + estimate-only
P2 (数据闭环): Collector + 实测校准 + 支持矩阵
```

## 7. 风险与缓解

### 7.1 高风险项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **MindIE 生态对齐** | MindIE 的 API/配置格式可能与 TRT-LLM/vLLM 差异很大 | 先做 API 调研，设计适配层隔离差异 |
| **通信模型差异** | HCCL 的算法和性能特征与 NCCL 不同 | 实测 HCCL 延迟数据，建立独立的通信模型 |
| **内存管理差异** | CANN 的内存管理不同于 CUDA | 实测 CANN 运行时内存开销，更新 other_mem |

### 7.2 中风险项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **Generator 模板膨胀** | 每个后端需要独立模板 | 保持模板最小化，复用跨后端逻辑 |
| **算力估算精度** | 公式估算可能偏差较大 | 优先收集少量关键算子数据校准 |

### 7.3 低风险项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **文档补齐** | 用户文档滞后 | 与代码同步更新 |
| **测试覆盖** | 新代码缺少测试 | 优先编写集成测试 |

## 8. 首期最小任务集

### 8.1 核心任务

| # | 任务 | 产出 | 验收标准 |
|---|------|------|----------|
| 1 | 创建 ascend_910b2.yaml | 系统 YAML | `load_system_spec("ascend_910b2")` 成功 |
| 2 | 添加 vendor/arch 字段 | system_spec.py 扩展 | `is_ascend_system()` 返回正确 |
| 3 | 实现 HCCL 通信模型 | communication.py 扩展 | 能计算 AllReduce 延迟 |
| 4 | 适配内存估算 | memory.py 修改 | 使用 comm_mem 替代 nccl_mem |
| 5 | 创建 MindIE 后端骨架 | mindie_backend.py | `get_backend("mindie")` 不报错 |
| 6 | 创建 estimate-only 数据目录 | systems/data/ascend_910b2/ | PerfDatabase 加载不报错 |
| 7 | 添加昇腾 hardware profile | hardware.yaml | Generator 能识别 ascend_910b2 |
| 8 | 端到端冒烟测试 | 测试用例 | `aiconfigurator cli default --system ascend_910b2 --backend mindie --model ...` 能输出结果 |

### 8.2 完成后可实现的能力

完成上述 8 项任务后：

```bash
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V3 \
  --total-gpus 8 \
  --system ascend_910b2 \
  --backend mindie \
  --isl 8000 --osl 1000 \
  --ttft 5000 --tpot 50
```

能输出基于算力公式估算的性能预测结果（精度有限，但可作为部署参考）。

## 9. 测试策略

### 9.1 单元测试

```python
# tests/unit/sdk/test_ascend_system.py
def test_load_ascend_system_spec():
    spec = load_system_spec("ascend_910b2")
    assert spec["vendor"] == "huawei"
    assert spec["gpu"]["mem_capacity"] == 68719476736

def test_is_ascend_system():
    assert is_ascend_system("ascend_910b2") is True
    assert is_ascend_system("b300_sxm") is False

def test_hccl_allreduce_latency():
    latency = get_hccl_allreduce_latency(
        message_size=1024*1024*1024,  # 1GB
        num_gpus=8,
        intra_node_bw=56e9,
        inter_node_bw=50e9,
    )
    assert latency > 0
```

### 9.2 集成测试

```python
# tests/integration/test_ascend_default.py
def test_ascend_default_mode():
    """昇腾系统 default 模式端到端测试"""
    result = run_cli_default(
        model="deepseek-ai/DeepSeek-V3",
        system="ascend_910b2",
        backend="mindie",
        total_gpus=8,
        isl=8000, osl=1000,
    )
    assert result.exit_code == 0
    assert len(result.top_configs) > 0
```

### 9.3 回归测试

确保昇腾适配不影响现有 NVIDIA 系统：

```bash
# 运行现有测试套件
pytest tests/unit/ -k "not ascend"
pytest tests/integration/ -k "not ascend"
```

## 10. 附录

### 10.1 昇腾 910B 详细硬件规格

| 参数 | 910B2 | 910B4 |
|------|-------|-------|
| AI Core 数量 | 20 | 25 |
| AI CPU 数量 | 12 | 12 |
| HBM 容量 | 64GB | 64GB |
| HBM 带宽 | 1.6TB/s | 1.6TB/s |
| FP16 算力 | 320 TFLOPS | 400 TFLOPS |
| FP32 算力 | 80 TFLOPS | 100 TFLOPS |
| INT8 算力 | 640 TOPS | 800 TOPS |
| HCCS 版本 | 2.0 | 3.0 |
| HCCS 带宽 | 56GB/s | 100GB/s |
| PCIe 版本 | Gen5 x16 | Gen5 x16 |
| 功耗 | 400W | 450W |

### 10.2 HCCL vs NCCL API 对比

| 功能 | NCCL | HCCL |
|------|------|------|
| 初始化 | `ncclCommInitRank` | `HcclCommInitRootInfo` |
| AllReduce | `ncclAllReduce` | `HcclAllReduce` |
| AllGather | `ncclAllGather` | `HcclAllGather` |
| AllToAll | `ncclAllToAll` | `HcclAllToAll` |
| ReduceScatter | `ncclReduceScatter` | `HcclReduceScatter` |
| Broadcast | `ncclBroadcast` | `HcclBroadcast` |
| 通信组 | `ncclComm_t` | `HcclComm` |
| 数据类型 | `ncclDataType_t` | `HcclDataType` |
| 归约操作 | `ncclRedOp_t` | `HcclReduceOp` |

### 10.3 相关文档链接

- 昇腾 910B 硬件规格：华为官方文档
- CANN 算子开发指南：华为 CANN 文档
- MindIE 推理框架：MindIE 官方仓库
- HCCL 集合通信库：HCCL API 参考
- aiconfigurator 架构剖析：`docs/ckw/architecture_deep_dive.md`
- 已有昇腾适配分析：`docs/ckw/ascend_910b_adaptation_plan.md`
- NVIDIA 依赖分析：`docs/ckw/nvidia_current_architecture_analysis.md`

---

*本文档基于 aiconfigurator v0.12.0 代码库分析。*
