# Default 模式数据流与采集覆盖指南

本文档详细分析 `aiconfigurator cli default` 命令背后使用的数据，以及如何通过 collector 采集数据进行覆盖。

---

## 目录

1. [Default 模式数据流分析](#default-模式数据流分析)
2. [系统定义文件](#系统定义文件)
3. [性能数据文件](#性能数据文件)
4. [DeepSeek-V4 特殊数据](#deepseek-v4-特殊数据)
5. [B300 vllm 现有数据](#b300-vllm-现有数据)
6. [数据采集覆盖方法](#数据采集覆盖方法)
7. [数据库模式选择](#数据库模式选择)

---

## Default 模式数据流分析

当你运行以下命令时：

```bash
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --total-gpus 64 \
  --system b300_sxm \
  --backend vllm
```

背后的数据加载链路如下：

```
CLI (default mode)
  → build_default_tasks() [创建 agg + disagg Task]
    → Task.__post_init__() [解析模型/后端/版本/量化]
      → get_latest_database_version() [确定版本]
    → Task.run()
      → get_database_view()
        → get_database() [加载 PerfDatabase]
          → 读取 b300_sxm.yaml [系统规格]
          → 遍历 data/b300_sxm/{family}/{backend}/{version}/ [数据目录]
          → 加载 *.parquet 文件 [性能数据]
      → sweep_agg() / sweep_disagg() [执行优化扫描]
```

---

## 系统定义文件

**位置**: `aic-core/src/aiconfigurator_core/systems/b300_sxm.yaml`

```yaml
data_dir: data/b300_sxm  # 相对于 systems_dir 的数据目录

gpu:
  mem_bw: 7750000000000        # 7.75 TB/s
  mem_capacity: 288400343040   # 275040MiB
  bfloat16_tc_flops: 2250000000000000  # 2250 TFLOPS
  fp8_tc_flops: 4500000000000000       # 4.5 PFLOPS
  fp4_tc_flops: 14000000000000000      # 14 PFLOPS
  power: 1100  # Watt
  sm_version: 103  # Blackwell 架构

node:
  num_gpus_per_node: 8
  inter_node_bw: 100000000000   # 100GB/s CX8 XDR
  intra_node_bw: 900000000000   # 900GB/s NVLink
  pcie_bw: 128000000000         # PCIe Gen6

misc:
  nccl_mem:
    1: 0
    2: 358612992   # 342MB
    4: 411041792   # 392MB
    8: 411041792   # 392MB
  other_mem: 3758096384  # 3.5GB
  nccl_version: '2.27'
```

---

## 性能数据文件

### 数据目录结构

**根目录**: `aic-core/src/aiconfigurator_core/systems/data/b300_sxm/`

```
data/b300_sxm/
├── attention/           # 注意力层性能数据
├── comm/                # 通信层性能数据 (NCCL/allreduce)
├── encoder_attention/   # 编码器注意力
├── gemm/                # GEMM 矩阵乘法
├── linear_attention/    # 线性注意力 (GDN/Mamba2)
├── mhc/                 # MHC 模块
├── mla/                 # MLA (Multi-head Latent Attention)
├── mla_bmm/             # MLA BMM
├── moe/                 # MoE (Mixture of Experts)
├── quantize/            # 量化相关
└── sparse_attention/    # 稀疏注意力 (DSA/CSA/HCA)
```

### vllm 后端数据文件

对于 `--backend vllm`，加载 `vllm/0.24.0/` 下的数据：

| 数据类别 | 文件路径 | 说明 |
|----------|----------|------|
| **GEMM** | `gemm/vllm/0.24.0/gemm_perf.parquet` | 矩阵乘法性能 |
| **MoE** | `moe/vllm/0.24.0/moe_perf.parquet` | 混合专家性能 |
| **MLA 预填充** | `mla/vllm/0.24.0/mla_context_module_perf.parquet` | MLA 预填充阶段 |
| **MLA 解码** | `mla/vllm/0.24.0/mla_generation_module_perf.parquet` | MLA 解码阶段 |
| **标准 Attention 预填充** | `attention/vllm/0.24.0/context_attention_perf.parquet` | 标准注意力预填充 |
| **标准 Attention 解码** | `attention/vllm/0.24.0/generation_attention_perf.parquet` | 标准注意力解码 |
| **AllReduce 通信** | `comm/vllm/0.24.0/custom_allreduce_perf.parquet` | AllReduce 通信 |
| **FP8 量化** | `quantize/vllm/0.24.0/computescale_perf.parquet` | FP8 量化开销 |
| **DSA 预填充** | `sparse_attention/vllm/0.24.0/dsa_context_module_perf.parquet` | DSA 预填充 |
| **DSA 解码** | `sparse_attention/vllm/0.24.0/dsa_generation_module_perf.parquet` | DSA 解码 |
| **编码器注意力** | `encoder_attention/vllm/0.24.0/encoder_attention_perf.parquet` | VL 模型编码器 |
| **GDN** | `linear_attention/vllm/0.24.0/gdn_perf.parquet` | GDN/Mamba2 |
| **MHC** | `mhc/vllm/0.24.0/mhc_module_perf.parquet` | MHC 模块 |

### 数据文件格式

每个 parquet 文件包含以下列（以 gemm_perf.parquet 为例）：

| 列名 | 说明 |
|------|------|
| `framework` | 框架名称 (vllm) |
| `version` | 版本号 (0.24.0) |
| `device` | GPU 型号 (NVIDIA B300 SXM6 AC) |
| `op_name` | 操作名称 (gemm) |
| `kernel_source` | 内核来源 |
| `dtype` | 数据类型 (bfloat16, fp8 等) |
| `m`, `n`, `k` | 矩阵维度 |
| `latency` | 延迟 (ms) |
| `backend` | 后端类型 |

---

## DeepSeek-V4 特殊数据

DeepSeek-V4 使用 MLA + MoE + DSA 架构，需要以下额外数据：

### CSA (Cross Sliding Attention)

| 文件 | 说明 |
|------|------|
| `sparse_attention/vllm/0.24.0/dsv4_csa_context_module_perf.parquet` | CSA 预填充 |
| `sparse_attention/vllm/0.24.0/dsv4_csa_generation_module_perf.parquet` | CSA 解码 |

### HCA (Hybrid Chunked Attention)

| 文件 | 说明 |
|------|------|
| `sparse_attention/vllm/0.24.0/dsv4_hca_context_module_perf.parquet` | HCA 预填充 |
| `sparse_attention/vllm/0.24.0/dsv4_hca_generation_module_perf.parquet` | HCA 解码 |
| `sparse_attention/vllm/0.24.0/dsv4_hca_attn_module_perf.parquet` | HCA 注意力模块 |

### 其他 DeepSeek-V4 专用

| 文件 | 说明 |
|------|------|
| `sparse_attention/vllm/0.24.0/dsv4_paged_mqa_logits_module_perf.parquet` | Paged MQA Logits |
| `mla_bmm/vllm/0.24.0/mla_bmm_perf.parquet` | MLA BMM |

---

## B300 vllm 现有数据

当前 B300 vllm 已有数据（`vllm/0.24.0/`）：

| 数据类别 | 目录 | 状态 |
|----------|------|------|
| Attention | `attention/vllm/0.24.0/` | ✅ 已有 |
| Comm | `comm/vllm/0.24.0/` | ✅ 已有 |
| Encoder Attention | `encoder_attention/vllm/0.24.0/` | ✅ 已有 |
| GEMM | `gemm/vllm/0.24.0/` | ✅ 已有 |
| Linear Attention | `linear_attention/vllm/0.24.0/` | ✅ 已有 |
| MHC | `mhc/vllm/0.24.0/` | ✅ 已有 |
| MLA | `mla/vllm/0.24.0/` | ✅ 已有 |
| MoE | `moe/vllm/0.24.0/` | ✅ 已有 |
| Quantize | `quantize/vllm/0.24.0/` | ✅ 已有 |
| Sparse Attention | `sparse_attention/vllm/0.24.0/` | ✅ 已有 (含 dsv4_* 文件) |

---

## 数据采集覆盖方法

### 采集命令

参考 `docs/ckw/b300_data_collection_guide.md`：

```bash
# 进入 collector 容器
docker exec -it collector-vllm-b300 bash
cd /workspace/collector

# 采集 DeepSeek-V4 相关数据
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-architecture DeepseekV4ForCausalLM

# 或全量采集（包含所有模型架构）
python3 collect.py --backend vllm --gpu b300_sxm
```

### 采集的文件映射

| collector 输出文件 | 数据目录目标位置 |
|-------------------|-----------------|
| `gemm_perf.parquet` | `data/b300_sxm/gemm/vllm/{version}/` |
| `moe_perf.parquet` | `data/b300_sxm/moe/vllm/{version}/` |
| `context_mla_perf.parquet` | `data/b300_sxm/mla/vllm/{version}/` |
| `generation_mla_perf.parquet` | `data/b300_sxm/mla/vllm/{version}/` |
| `context_attention_perf.parquet` | `data/b300_sxm/attention/vllm/{version}/` |
| `generation_attention_perf.parquet` | `data/b300_sxm/attention/vllm/{version}/` |
| `custom_allreduce_perf.parquet` | `data/b300_sxm/comm/vllm/{version}/` |
| `computescale_perf.parquet` | `data/b300_sxm/quantize/vllm/{version}/` |
| `dsa_context_module_perf.parquet` | `data/b300_sxm/sparse_attention/vllm/{version}/` |
| `dsa_generation_module_perf.parquet` | `data/b300_sxm/sparse_attention/vllm/{version}/` |
| `dsv4_csa_context_module_perf.parquet` | `data/b300_sxm/sparse_attention/vllm/{version}/` |
| `dsv4_csa_generation_module_perf.parquet` | `data/b300_sxm/sparse_attention/vllm/{version}/` |
| `dsv4_hca_*_module_perf.parquet` | `data/b300_sxm/sparse_attention/vllm/{version}/` |
| `mla_bmm_perf.parquet` | `data/b300_sxm/mla_bmm/vllm/{version}/` |
| `encoder_attention_perf.parquet` | `data/b300_sxm/encoder_attention/vllm/{version}/` |
| `gdn_perf.parquet` | `data/b300_sxm/linear_attention/vllm/{version}/` |
| `mhc_module_perf.parquet` | `data/b300_sxm/mhc/vllm/{version}/` |
| `nccl_perf.parquet` | `data/b300_sxm/comm/vllm/{version}/` |

### 覆盖数据步骤

#### 1. 创建新版本目录

```bash
# 假设采集的 vllm 版本是 0.25.0
VERSION="0.25.0"
BASE_DIR="aic-core/src/aiconfigurator_core/systems/data/b300_sxm"

mkdir -p $BASE_DIR/gemm/vllm/$VERSION
mkdir -p $BASE_DIR/moe/vllm/$VERSION
mkdir -p $BASE_DIR/mla/vllm/$VERSION
mkdir -p $BASE_DIR/attention/vllm/$VERSION
mkdir -p $BASE_DIR/comm/vllm/$VERSION
mkdir -p $BASE_DIR/quantize/vllm/$VERSION
mkdir -p $BASE_DIR/sparse_attention/vllm/$VERSION
mkdir -p $BASE_DIR/encoder_attention/vllm/$VERSION
mkdir -p $BASE_DIR/linear_attention/vllm/$VERSION
mkdir -p $BASE_DIR/mhc/vllm/$VERSION
mkdir -p $BASE_DIR/mla_bmm/vllm/$VERSION
```

#### 2. 复制采集数据

```bash
# 从 collector 输出目录复制到数据目录
COLLECTOR_OUTPUT="/path/to/collector/output"

cp $COLLECTOR_OUTPUT/gemm_perf.parquet $BASE_DIR/gemm/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/moe_perf.parquet $BASE_DIR/moe/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/context_mla_perf.parquet $BASE_DIR/mla/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/generation_mla_perf.parquet $BASE_DIR/mla/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/context_attention_perf.parquet $BASE_DIR/attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/generation_attention_perf.parquet $BASE_DIR/attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/custom_allreduce_perf.parquet $BASE_DIR/comm/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/computescale_perf.parquet $BASE_DIR/quantize/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/dsa_*.parquet $BASE_DIR/sparse_attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/dsv4_*.parquet $BASE_DIR/sparse_attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/encoder_attention_perf.parquet $BASE_DIR/encoder_attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/gdn_perf.parquet $BASE_DIR/linear_attention/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/mhc_module_perf.parquet $BASE_DIR/mhc/vllm/$VERSION/
cp $COLLECTOR_OUTPUT/mla_bmm_perf.parquet $BASE_DIR/mla_bmm/vllm/$VERSION/
```

#### 3. 创建 collection_meta.yaml

为每个数据目录创建元数据文件：

```bash
for dir in gemm moe mla attention comm quantize sparse_attention \
           encoder_attention linear_attention mhc mla_bmm; do
  cat > $BASE_DIR/$dir/vllm/$VERSION/collection_meta.yaml << EOF
collection_date: $(date +%Y-%m-%d)
gpu: NVIDIA B300 SXM
backend: vllm
version: $VERSION
collector_version: 1.0.0
EOF
done
```

#### 4. 更新 Support Matrix

编辑 `aic-core/src/aiconfigurator_core/systems/support_matrix/b300_sxm.csv`，添加新版本：

```csv
backend,version,status
vllm,0.19.0,PASS
vllm,0.22.0,PASS
vllm,0.24.0,PASS
vllm,0.25.0,PASS  # 新增
```

---

## 数据库模式选择

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `SILICON` | 仅使用采集的实测数据 | 有完整采集数据时，精度最高 |
| `HYBRID` | 实测优先 + SOL 回退 | 部分数据缺失时，推荐使用 |
| `EMPIRICAL` | 完全 SOL + 经验因子 | 无采集数据时，精度较低 |

### 使用示例

```bash
# 使用 SILICON 模式（需要完整采集数据）
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --total-gpus 64 \
  --system b300_sxm \
  --backend vllm \
  --backend-version 0.25.0 \
  --database-mode SILICON

# 使用 HYBRID 模式（推荐，采集数据 + SOL 回退）
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --total-gpus 64 \
  --system b300_sxm \
  --backend vllm \
  --database-mode HYBRID

# 使用 EMPIRICAL 模式（无采集数据时）
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --total-gpus 64 \
  --system b300_sxm \
  --backend vllm \
  --database-mode EMPIRICAL
```

### 建议

1. **初次采集**: 先用 `HYBRID` 模式验证，确保采集数据覆盖关键路径
2. **完整采集**: 完成全量采集后，切换到 `SILICON` 模式获取最高精度
3. **数据复用**: 可通过 `reuse.yaml` 声明从旧版本复用数据，减少重复采集

---

## 附录：数据继承机制

### Shared Layer

当 `database_mode=SILICON` 或 `HYBRID` 时启用 shared_layer：

1. **跨版本继承**: 如果 `vllm/0.25.0` 缺少某个 shape 的数据，会从 `vllm/0.24.0` 继承
2. **跨后端继承**: 通过 `op_kernel_source_manifest.yaml` 声明的跨后端数据复用

### reuse.yaml 示例

```yaml
reuse:
  - table: "context_mla"
    from_version: "0.24.0"
    reason: "same kernel"
    approved_by: "team"
  - table: "generation_mla"
    from_version: "0.24.0"
    reason: "same kernel"
```

### TransferKind 策略

| 策略 | 说明 |
|------|------|
| `off` | 无转移 |
| `conservative` | 仅 XSHAPE (同 quant 内跨 shape) |
| `balanced` | XSHAPE + XQUANT (跨 quant) |
| `aggressive` | 所有转移类型 (跨 profile/op) |

---

*文档生成时间: 2026-08-04*
*项目: aiconfigurator*
*目标: B300 SXM + vLLM 后端*
