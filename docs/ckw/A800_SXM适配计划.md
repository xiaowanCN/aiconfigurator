# A800 SXM 适配计划（详细版）

> 本文档详细说明如何将 aiconfigurator 适配到 NVIDIA A800 SXM（A100 国内特供版）。

---

## 一、硬件差异分析

### 1.1 A800 vs A100 规格对比

| 规格 | A100 SXM | A800 SXM | 差异影响 |
|------|----------|----------|----------|
| GPU 核心 | GA100 | GA100 | 无差异 |
| HBM 容量 | 80GB HBM2e | 80GB HBM2e | 无差异 |
| 显存带宽 | 2.039 TB/s | 2.039 TB/s | 无差异 |
| BF16 算力 | 312 TFLOPS | 312 TFLOPS | 无差异 |
| INT8 算力 | 624 TOPS | 624 TOPS | 无差异 |
| SM Version | 80 | 80 | 无差异 |
| **NVLink 带宽** | **600 GB/s** (12 links) | **400 GB/s** (8 links) | **唯一差异** |
| PCIe | Gen4 x16 | Gen4 x16 | 无差异 |
| 功耗 | 400W | 400W | 无差异 |

### 1.2 结论

A800 与 A100 **仅 NVLink 互联带宽不同**，算子层面数据可完全复用。这是一个相对简单的适配任务，主要工作是：

1. 创建新的系统配置文件（调整 `intra_node_bw`）
2. 复制 A100 的性能数据
3. 更新版本配置和支持矩阵

---

## 二、适配任务清单

### Task 1: 创建系统配置文件

**文件**: `aic-core/src/aiconfigurator_core/systems/a800_sxm.yaml`

**内容**: 基于 `a100_sxm.yaml` 修改，仅调整 `intra_node_bw`

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

data_dir: data/a800_sxm # relative to systems_dir
gpu:
  mem_bw: 2039000000000 # 2.039TB/s
  mem_bw_empirical_scaling_factor: 0.8 # some nonofficial correction based on observations, you should try to modify based on your own observations
  mem_empirical_constant_latency: 0.000003 # 3us some nonofficial correction based on observations, you should try to modify based on your own observations
  mem_capacity: 85899345920 # 80GiB
  bfloat16_tc_flops: 312000000000000 # 312TFLOPS
  int8_tc_flops: 624000000000000 # 624TFLOPS
  power: 400  # Watt
  sm_version: 80

node:
  num_gpus_per_node: 8
  inter_node_bw: 25000000000  # Byte/s per GPU, single direction, assume 1:1 CX6 per node
  intra_node_bw: 400000000000  # Byte/s per gpu, single direction (A800: 400GB/s, A100: 600GB/s)
  pcie_bw: 32000000000  # Byte/s, single direction, pcie 4.0
  p2p_latency: 0.00001  # 10us some nonofficial correction based on observations, you should try to modify based on your own observations

misc:
  nccl_mem: # some nonofficial correction based on observations, you should try to modify based on your own observations
    1: 0
    2: 358612992 # 342MB
    4: 411041792 # 392MB
    8: 411041792 # 392MB
  other_mem: 3758096384 # increase from 551MB to 3.5GB for safer deployment, this will cover part of the inaccurate mem calc.
  nccl_version: '2.27.3'
```

**关键改动**:
- `intra_node_bw`: 从 `300000000000` (300GB/s) 改为 `400000000000` (400GB/s)

---

### Task 2: 创建数据目录结构

**目录**: `aic-core/src/aiconfigurator_core/systems/data/a800_sxm/`

**策略**: 初始阶段直接复用 A100 数据（算子层面无差异）

需创建的子目录结构：
```
data/a800_sxm/
├── attention/
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           ├── context_attention_perf.parquet
│           └── generation_attention_perf.parquet
├── comm/
│   ├── nccl/
│   │   └── 2.27.3/
│   │       ├── collection_meta.yaml
│   │       └── nccl_perf.parquet
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           └── custom_allreduce_perf.parquet
├── gemm/
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           └── gemm_perf.parquet
├── linear_attention/
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           └── gdn_perf.parquet
├── mla/
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           ├── context_mla_perf.parquet
│           └── generation_mla_perf.parquet
├── mla_bmm/
│   └── vllm/
│       └── 0.14.0/
│           ├── collection_meta.yaml
│           └── mla_bmm_perf.parquet
└── moe/
    └── vllm/
        └── 0.14.0/
            ├── collection_meta.yaml
            └── moe_perf.parquet
```

---

### Task 3: 更新 query_versions.yaml

**文件**: `aic-core/src/aiconfigurator_core/systems/query_versions.yaml`

**改动**: 在 `overrides` 下新增 `a800_sxm` 配置

```yaml
overrides:
  # ... 现有配置 ...
  a100_sxm:
    trtllm: {current: "1.0.0", previous: null}
    sglang: {current: "0.5.10", previous: null}
    vllm: {current: "0.14.0", previous: null}
  a800_sxm:                    # 新增
    trtllm: {current: "1.0.0", previous: null}
    sglang: {current: "0.5.10", previous: null}
    vllm: {current: "0.14.0", previous: null}
```

---

### Task 4: 创建 support_matrix CSV

**文件**: `aic-core/src/aiconfigurator_core/systems/support_matrix/a800_sxm.csv`

**内容**: 复制 `a100_sxm.csv` 并将所有 `a100_sxm` 替换为 `a800_sxm`

**操作步骤**:
```bash
cp systems/support_matrix/a100_sxm.csv systems/support_matrix/a800_sxm.csv
sed -i 's/a100_sxm/a800_sxm/g' systems/support_matrix/a800_sxm.csv
```

---

### Task 5: 更新 support_matrix index

**文件**: `aic-core/src/aiconfigurator_core/systems/support_matrix/index.json`

**改动**: 添加 `a800_sxm` 条目

---

### Task 6: 复制性能数据文件

从 `data/a100_sxm/` 复制到 `data/a800_sxm/`：

| 数据类别 | 源路径 | 目标路径 |
|----------|--------|----------|
| GEMM | `a100_sxm/gemm/vllm/0.14.0/` | `a800_sxm/gemm/vllm/0.14.0/` |
| Attention | `a100_sxm/attention/vllm/0.14.0/` | `a800_sxm/attention/vllm/0.14.0/` |
| MoE | `a100_sxm/moe/vllm/0.14.0/` | `a800_sxm/moe/vllm/0.14.0/` |
| MLA | `a100_sxm/mla/vllm/0.14.0/` | `a800_sxm/mla/vllm/0.14.0/` |
| MLA BMM | `a100_sxm/mla_bmm/vllm/0.14.0/` | `a800_sxm/mla_bmm/vllm/0.14.0/` |
| Comm (NCCL) | `a100_sxm/comm/nccl/2.27.3/` | `a800_sxm/comm/nccl/2.27.3/` |
| Comm (vLLM) | `a100_sxm/comm/vllm/0.14.0/` | `a800_sxm/comm/vllm/0.14.0/` |
| Linear Attn | `a100_sxm/linear_attention/vllm/0.14.0/` | `a800_sxm/linear_attention/vllm/0.14.0/` |

每个目录下需要更新 `collection_meta.yaml` 中的 GPU 名称。

**collection_meta.yaml 模板**:
```yaml
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

schema_version: 1
provenance: legacy
runtime:
  framework: vllm
  version: 0.14.0
tables:
  <table_name>_perf:
    status: complete
```

---

### Task 7: (可选) 通信数据校准

**背景**: AllReduce/AllToAll 等通信操作受 NVLink 带宽影响

**建议**: 在 A800 实机上运行 collector 采集通信数据：
```bash
python3 collect.py --backend vllm --gpu a800_sxm --op-family comm
```

仅需替换 `comm/vllm/0.14.0/custom_allreduce_perf.parquet`

---

## 三、文件变更汇总

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| **新增** | `aic-core/src/aiconfigurator_core/systems/a800_sxm.yaml` | 系统配置文件 |
| **新增** | `aic-core/src/aiconfigurator_core/systems/data/a800_sxm/` | 数据目录及所有子目录 |
| **修改** | `aic-core/src/aiconfigurator_core/systems/query_versions.yaml` | 添加 a800_sxm 版本配置 |
| **新增** | `aic-core/src/aiconfigurator_core/systems/support_matrix/a800_sxm.csv` | 支持矩阵 |
| **修改** | `aic-core/src/aiconfigurator_core/systems/support_matrix/index.json` | 索引更新 |

---

## 四、验证方法

### 4.1 基础验证

完成适配后，运行以下命令验证：

```bash
# 单机小模型验证
aiconfigurator cli default \
  --model Qwen/Qwen2.5-7B \
  --total-gpus 8 \
  --system a800_sxm \
  --backend vllm \
  --backend-version 0.14.0
```

### 4.2 多机验证

```bash
# 多机 MoE 模型验证
aiconfigurator cli default \
  --model Qwen/Qwen3-235B-A22B \
  --total-gpus 64 \
  --system a800_sxm \
  --backend vllm \
  --backend-version 0.14.0
```

### 4.3 对比验证

建议将 A800 和 A100 的估算结果进行对比，观察多机场景下的差异：

```bash
# A100 基准
aiconfigurator cli default \
  --model Qwen/Qwen3-235B-A22B \
  --total-gpus 64 \
  --system a100_sxm \
  --backend vllm \
  --backend-version 0.14.0 \
  --output a100_result.json

# A800 对比
aiconfigurator cli default \
  --model Qwen/Qwen3-235B-A22B \
  --total-gpus 64 \
  --system a800_sxm \
  --backend vllm \
  --backend-version 0.14.0 \
  --output a800_result.json
```

---

## 五、风险评估

| 风险项 | 风险等级 | 说明 | 缓解措施 |
|--------|----------|------|----------|
| 算子数据复用 | **低** | A800 与 A100 算力完全一致，GEMM/Attention/MoE 等算子性能无差异 | 无需特殊处理 |
| 通信数据精度 | **中** | NVLink 带宽差异 (400 vs 600 GB/s) 可能导致多机通信估算偏差 | 在 A800 实机采集通信数据校准 |
| NCCL 行为差异 | **低** | 同版本 NCCL，仅带宽不同 | 无需特殊处理 |

---

## 六、后续优化建议

### 6.1 通信数据校准

在 A800 实机上采集以下通信数据，替换复用的 A100 数据：

- `comm/vllm/0.14.0/custom_allreduce_perf.parquet`
- `comm/nccl/2.27.3/nccl_perf.parquet`

### 6.2 扩展支持

如需支持更多后端（sglang、trtllm），可按相同方式复制对应版本的数据目录。

---

## 七、附录

### 7.1 参考文档

- A100 系统配置: `aic-core/src/aiconfigurator_core/systems/a100_sxm.yaml`
- A100 数据目录: `aic-core/src/aiconfigurator_core/systems/data/a100_sxm/`
- 数据流指南: `docs/ckw/data_flow_and_collection_guide.md`

### 7.2 快速命令参考

```bash
# 复制 A100 数据目录到 A800
cp -r aic-core/src/aiconfigurator_core/systems/data/a100_sxm aic-core/src/aiconfigurator_core/systems/data/a800_sxm

# 更新 collection_meta.yaml 中的 GPU 名称
find aic-core/src/aiconfigurator_core/systems/data/a800_sxm -name "collection_meta.yaml" -exec sed -i 's/A100/A800/g' {} \;
```

---

*文档生成时间: 2026-09-07*
*项目: aiconfigurator*
*目标: A800 SXM + vLLM 0.14.0 后端*
