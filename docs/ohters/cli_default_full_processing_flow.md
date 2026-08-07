# aiconfigurator CLI Default 完整处理流程分析

## 目录

1. [命令概述](#1-命令概述)
2. [整体架构图](#2-整体架构图)
3. [阶段一：入口与参数解析](#3-阶段一入口与参数解析)
4. [阶段二：构建 Task 对象](#4-阶段二构建-task-对象)
5. [阶段三：性能数据库加载](#5-阶段三性能数据库加载)
6. [阶段四：Sweep 搜索执行](#6-阶段四sweep-搜索执行)
7. [阶段五：结果处理与 Pareto 选择](#7-阶段五结果处理与-pareto-选择)
8. [阶段六：输出与保存](#8-阶段六输出与保存)
9. [附录：关键数据结构与常量](#9-附录关键数据结构与常量)

---

## 1. 命令概述

```bash
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --total-gpus 64 \
  --system b300_sxm \
  --backend vllm \
  --isl 256 --osl 256 \
  --ttft 2000 --tpot 26 \
  --top-n 8 \
  --backend-version 0.24.0
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `--model` | `deepseek-ai/DeepSeek-V4-Flash` | HuggingFace 模型路径 |
| `--total-gpus` | `64` | 总 GPU 数量 |
| `--system` | `b300_sxm` | GPU 系统类型 (B300 SXM) |
| `--backend` | `vllm` | 推理后端 |
| `--isl` | `256` | 输入序列长度 (input sequence length) |
| `--osl` | `256` | 输出序列长度 (output sequence length) |
| `--ttft` | `2000` | TTFT SLA 目标 (ms) |
| `--tpot` | `26` | TPOT SLA 目标 (ms) |
| `--top-n` | `8` | 输出前 N 个最优配置 |
| `--backend-version` | `0.24.0` | 性能数据库版本 |

未指定的参数使用默认值：
- `--serving-mode`: `auto` → 同时搜索 agg 和 disagg
- `--database-mode`: `SILICON` (使用实测硅片数据)
- `--free-gpu-memory-fraction`: `1.0`
- `--prefix`: `0`
- `--enable-chunked-prefill`: `False`

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    aiconfigurator cli default                        │
│  --model deepseek-ai/DeepSeek-V4-Flash --total-gpus 64             │
│  --system b300_sxm --backend vllm --isl 256 --osl 256              │
│  --ttft 2000 --tpot 26 --top-n 8 --backend-version 0.24.0          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1: 入口分发 (main.py)                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │ main.py:main │───▶│ _run_cli()   │───▶│ configure_parser()   │    │
│  │  (入口)      │    │  (二级分发)    │    │  (参数解析)           │    │
│  └─────────────┘    └──────────────┘    └──────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ args.mode == "default"
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2: 构建 Task 对象 (build_default_tasks)                       │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ 解析后端版本     │  │ 确定搜索模式     │  │ 构建 parallel 配置   │  │
│  │ backend_version │  │ agg/disagg/afd  │  │ TP/PP/DP/EP/CP      │  │
│  └────────────────┘  └─────────────────┘  └──────────────────────┘  │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ 解析量化模式     │  │ 加载模型配置     │  │ 创建 Task 对象       │  │
│  │ GEMM/MoE/KV    │  │ HF config.json  │  │ serving_mode=agg     │  │
│  └────────────────┘  └─────────────────┘  │ serving_mode=disagg  │  │
│                                            └──────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3: 执行 Sweep (_execute_tasks)                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Task.run()                                                  │    │
│  │  ├─ agg 模式 → sweep_agg()                                  │    │
│  │  │   └─ 三级嵌套: parallel_config × tpot × (batch × ctx)    │    │
│  │  │       └─ backend.run_agg() → 查性能DB → 计算 TTFT/TPOT   │    │
│  │  │                                                          │    │
│  │  └─ disagg 模式 → sweep_disagg()                            │    │
│  │      └─ 预计算 prefill/decode worker 候选 → 速率匹配          │    │
│  │          └─ run_static() → 查性能DB → 计算 per-worker 延迟   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 4: 结果处理 (process_experiment_result)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ SLA 过滤          │  │ Pareto 前沿计算   │  │ Top-N 选择         │ │
│  │ TPOT ≤ 26ms      │  │ 非支配排序        │  │ max(tokens/s/gpu)  │ │
│  │ TTFT ≤ 2000ms    │  │                   │  │                    │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ merge_experiment_results_by_mode()                            │   │
│  │ 合并 agg_vllm 结果 → 选择最优 serving mode                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 5: 输出 (log_final_summary + save_results)                    │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ 终端摘要表格     │  │ Pareto ASCII图  │  │ CSV/YAML 文件保存    │  │
│  │ 最优配置详情     │  │ tokens/s vs SLA │  │ pareto_frontier.png  │  │
│  └────────────────┘  └─────────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 阶段一：入口与参数解析

### 3.1 两级分发机制

```
aiconfigurator cli default --model ... --system ...
     │          │      │
     │          │      └─ mode = "default"
     │          └─ command = "cli"
     └─ 入口点: aiconfigurator.main:main
```

**文件**: `src/aiconfigurator/main.py`

```python
# 第一级: main() 解析 "cli" 子命令
def main(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli_parser = subparsers.add_parser("cli", add_help=False)
    cli_parser.set_defaults(handler=_run_cli)
    args, extras = parser.parse_known_args(argv)
    args.handler(extras)  # -> _run_cli(extras)

# 第二级: _run_cli() 解析 "default" 子命令
def _run_cli(argv):
    parser = argparse.ArgumentParser()
    configure_parser(parser)  # 注册 default/exp/generate/estimate/support/recommend
    args = parser.parse_args(argv)
    main(args)  # -> cli main(args)
```

### 3.2 参数解析结果

对于本命令，解析后的 `args` 对象关键字段：

```python
args = {
    "mode": "default",
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "total_gpus": 64,
    "system": "b300_sxm",
    "backend": "vllm",
    "isl": 256,
    "osl": 256,
    "ttft": 2000.0,
    "tpot": 26.0,
    "top_n": 8,
    "backend_version": "0.24.0",
    # 以下使用默认值
    "serving_mode": "auto",
    "database_mode": "SILICON",
    "free_gpu_memory_fraction": 1.0,
    "prefix": 0,
    "enable_chunked_prefill": False,
    "strict_sla": False,
    "inclusive_tpot": False,
    "nextn": 0,
    "image_height": 0,
    "image_width": 0,
    "num_images": 1,
}
```

### 3.3 输入验证

`_validate_default_mode_inputs()` 检查：
- `--model` 必须提供
- `--system` 必须提供
- `--total-gpus` 在 default 模式下必须提供

---

## 4. 阶段二：构建 Task 对象

### 4.1 流程图

```
build_default_tasks(args)
│
├─ 1. 解析 serving_mode
│   └─ "auto" → modes_to_sweep = ["agg", "disagg"]
│
├─ 2. 确定 backends_to_sweep
│   └─ backend="vllm" (非auto) → ["vllm"]
│   └─ 验证 vllm@0.24.0 在 b300_sxm 上是否可用
│
├─ 3. 加载模型配置
│   └─ 读取 HF config.json → ModelConfig
│   └─ 判断是否 MoE 模型 (DeepSeek-V4-Flash 是 MoE)
│
├─ 4. 解析量化模式
│   └─ HF config 推断 → bfloat16 回退
│   └─ DeepSeek-V4 MoE 特殊处理
│
├─ 5. 构建 global_kwargs (共享工作负载参数)
│   └─ isl=256, osl=256, ttft=2000, tpot=26, total_gpus=64, ...
│
├─ 6. 构建搜索空间
│   ├─ agg: TP=[1,2,4,8], PP=[1], DP=[1], MoE_EP=[1,...]
│   └─ disagg: prefill/decode 各自独立的 TP/PP/DP 候选
│
└─ 7. 创建 Task 对象
    ├─ Task(serving_mode="agg", ...)  → "agg_vllm"
    └─ Task(serving_mode="disagg", ...)  → "disagg_vllm"
```

### 4.2 搜索空间构建

对于 DeepSeek-V4-Flash (MoE 模型) + vllm 后端：

**Agg 模式搜索空间**:
```
TP candidates:  [1, 2, 4, 8]
PP candidates:  [1]
DP candidates:  [1]
MoE TP:         [1]
MoE EP:         [1] (vllm 不支持 WideEP)
CP candidates:  [1]
num_gpu:        [1, 2, 4, 8, 16, 32, 64]  (受 total_gpus=64 约束)
```

**Disagg 模式搜索空间**:
```
Prefill workers:
  TP: [1, 2, 4, 8]
  PP: [1]
  num_gpu_per_replica: [1, 2, 4, 8, 16, 32, 64]

Decode workers:
  TP: [1, 2, 4, 8]
  PP: [1]
  num_gpu_per_replica: [1, 2, 4, 8, 16, 32, 64]

Worker count:
  prefill_num_worker_list: [1..32]
  decode_num_worker_list: [1..32]
  num_gpu_list: [64]  (受 total_gpus 约束)
```

### 4.3 量化模式解析

对于 DeepSeek-V4-Flash，默认解析为：

| 量化维度 | 默认值 | 来源 |
|----------|--------|------|
| `gemm_quant_mode` | `bfloat16` | HF config 无量化信息 → 回退 |
| `moe_quant_mode` | `bfloat16` | 同上 |
| `kvcache_quant_mode` | `bfloat16` | 同上 |
| `fmha_quant_mode` | `bfloat16` | 同上 |
| `comm_quant_mode` | `bfloat16` | 同上 |

> 如果模型 config.json 中包含量化配置 (如 FP8)，则自动推断对应量化模式。

---

## 5. 阶段三：性能数据库加载

### 5.1 数据库结构

```
aic-core/src/aiconfigurator_core/systems/
├── b300_sxm.yaml                    # 系统规格 (GPU 算力/带宽/容量)
└── data/
    └── b300_sxm/
        ├── gemm/
        │   └── vllm/
        │       └── 0.24.0/
        │           └── gemm_perf.parquet
        ├── attention/
        │   └── vllm/
        │       └── 0.24.0/
        │           ├── context_attention_perf.parquet
        │           └── generation_attention_perf.parquet
        ├── moe/
        │   └── vllm/
        │       └── 0.24.0/
        │           └── moe_perf.parquet
        ├── comm/
        │   └── nccl/
        │       └── 2.27.3/
        │           └── nccl_perf.parquet
        └── mla/
            └── vllm/
                └── 0.24.0/
                    ├── context_mla_perf.parquet
                    └── generation_mla_perf.parquet
```

### 5.2 系统规格 (b300_sxm.yaml)

```yaml
gpu:
  mem_bw: 8000000000000           # 8 TB/s HBM 带宽
  mem_capacity: 206158430208      # ~192 GiB
  bfloat16_tc_flops: 4500000000000000   # 4500 TFLOPS
  fp8_tc_flops: 9000000000000000        # 9000 TFLOPS
  fp4_tc_flops: 18000000000000000       # 18000 TFLOPS
node:
  num_gpus_per_node: 8
  inter_node_bw: 100000000000     # 100 GB/s NVLink
  intra_node_bw: 900000000000    # 900 GB/s
misc:
  nccl_mem: {1: 0, 2: 358612992, 4: 411041792, 8: 411041792}
  other_mem: 3758096384
```

### 5.3 性能数据表结构

每个 `.parquet` 文件存储特定算子在不同 shape 下的实测延迟：

**GEMM 表** (`gemm_perf.parquet`):
| 列名 | 含义 |
|------|------|
| `quant_mode` | 量化模式 (bfloat16/fp8/...) |
| `m` | 批量维度 (batch × seq_len) |
| `n` | 输出维度 |
| `k` | 输入维度 |
| `latency` | 延迟 (ms) |
| `power` | 功率 (W) |
| `energy` | 能耗 (mJ) |

**MoE 表** (`moe_perf.parquet`):
| 列名 | 含义 |
|------|------|
| `quant_mode` | 量化模式 |
| `batch_size` | 批量大小 |
| `seq_len` | 序列长度 |
| `num_experts` | 专家数量 |
| `latency` | 延迟 (ms) |

**Context Attention 表** (`context_attention_perf.parquet`):
| 列名 | 含义 |
|------|------|
| `kvcache_quant_mode` | KV 缓存量化 |
| `fmha_quant_mode` | FMHA 量化 |
| `num_heads` | 注意力头数 |
| `seq_len` | 序列长度 |
| `batch` | 批量大小 |
| `latency` | 延迟 (ms) |

### 5.4 插值机制

当查询的 shape 不在实测数据中时，使用四步解析：

```
查询 (m=512, n=7168, k=16384)
│
├─ Step 1: 精确命中?
│   └─ 是 → 直接返回实测值
│
├─ Step 2: 数据范围内插值
│   ├─ Grid 类型 (Attention): 线性/√ 插值
│   └─ Scattered 类型 (GEMM): 最近邻站点 + 曲线插值
│       └─ 在 util (= SOL/latency) 空间做反距离加权
│
├─ Step 3: 超出范围外推
│   └─ 保持边界 util 不变: latency = SOL(query) / median_util
│
└─ Step 4: 无锚点 → 抛出 InterpolationDataNotAvailableError
```

**GEMM 插值公式** (util 空间):
```
对于未知 shape (n, k):
1. 找到最近的 nn_sites=4 个已收集站点 (log 空间距离)
2. 对每个站点 i，在其 m 曲线上评估 latency_i(m_query)
3. 计算 util_i = SOL(n,k,m) / latency_i
4. 反距离加权: avg_util = Σ(util_i / dist_i) / Σ(1/dist_i)
5. 返回 latency = SOL(n,k,m) / avg_util
```

**SOL (Speed-of-Light) 计算**:
```
GEMM SOL:
  sol_math = 2 × m × n × k / tc_flops × 1000    # 计算受限 (ms)
  sol_mem  = dtype_bytes × (m×n + m×k + n×k) / mem_bw × 1000  # 带宽受限 (ms)
  sol_time = max(sol_math, sol_mem)                # Roofline 模型

Attention SOL:
  sol_mem = bytes_accessed / mem_bw × 1000
```

---

## 6. 阶段四：Sweep 搜索执行

这是核心计算阶段。对于本命令，`backend="vllm"` (非 auto)，`serving_mode="auto"`，会创建两个 Task：
- `agg_vllm` → 调用 `sweep_agg()`
- `disagg_vllm` → 调用 `sweep_disagg()`

### 6.1 Agg 模式 Sweep (聚合推理)

#### 6.1.1 三级嵌套搜索结构

```
sweep_agg()
│
├─ Level 1: 遍历 parallel_config (外层循环)
│   for (tp, pp, dp, moe_tp, moe_ep, cp) in parallel_config_list:
│       model = get_model()      # 构建模型算子图
│       backend = get_backend()  # 初始化后端
│
│   ├─ Level 2: 遍历 TPOT 约束
│   │   for tpot_target in [26.0]:  # 单值
│   │
│   │   └─ Level 3: _sweep_one_parallel_agg()
│   │       │
│   │       ├─ 遍历 batch_size:
│   │       │   [1,2,...,15, 16,20,24,28, 32,40,48,56, 64,80,...,240,
│   │       │    256,288,...,480, 512,768, 1024]
│   │       │   (受 max_batch_size=512 截断)
│   │       │
│   │       └─ 遍历 ctx_tokens:
│   │           [256, 512, 768, ..., 8192, 10240, ...] (isl=256 时)
│   │           且必须是 isl 的整数倍 (enable_chunked_prefill=False)
│   │
│   └─ 每个 (batch, ctx) 点:
│       ├─ predict_agg_worker() → backend.run_agg()
│       ├─ 检查 OOM → break
│       ├─ 检查 SLA (TPOT ≤ 26 AND TTFT ≤ 2000)
│       └─ 通过 → 加入结果集
```

#### 6.1.2 backend.run_agg() 核心计算

对于每个 `(parallel_config, batch_size, ctx_tokens)` 点：

**Step 1: 计算调度参数**
```
isl_effective = text_isl + img_ctx_tokens = 256 + 0 = 256
decode_iterations = 1 + max(osl - 1, 0) / decode_tokens_per_iteration
                  = 1 + 255 / 1 = 256
steps_to_finish_ctx = ceil(isl × batch_size / ctx_tokens)
                    = ceil(256 × b / ctx_tokens)
balance_score = isl × b / ctx_tokens / decode_iterations
              = 256 × b / ctx_tokens / 256
```

**Step 2: 确定 mix 步数和 gen-only 步数**
```
if b == 1:
    num_mix_steps = 1
    num_genonly_steps = decode_iterations - 1 = 255

if b > 1:
    if steps_to_finish_ctx >= decode_iterations:
        # 上下文阶段主导
        num_mix_steps = steps_to_finish_ctx
        num_genonly_steps = 0
    else:
        # 上下文先完成
        num_mix_steps = steps_to_finish_ctx
        num_genonly_steps = decode_iterations - num_mix_steps
```

**Step 3: 计算每步延迟**

*Mixed Step (混合步)* — 包含 prefill 和 decode 的前向传播：

```
run_mixed() 使用三趟估算:

Pass 1 - 非注意力算子 (GEMM, MoE, Comm):
    调用 run_static(mode="static_ctx", bs=1, isl=ctx_tokens+decode_query_tokens)
    → 查询所有 context_ops 的性能数据库
    latency_pass1 = Σ op.query(database, x=combined_tokens)

Pass 2 - Context Attention:
    调用 run_static(mode="static_ctx", bs=ceil(ctx_tokens/isl), isl=isl)
    → 按 chunk 缩放
    latency_pass2 = attention_latency × ceil(isl/ctx_tokens)

Pass 3 - Generation Attention:
    调用 run_static(mode="static_gen", bs=num_decode_requests, isl=isl+osl//2, osl=2)
    latency_pass3 = generation_attention_latency

mix_step_latency = latency_pass1 + latency_pass2 + latency_pass3
mix_step_latency *= mix_efficiency  # 默认 1.0
```

*Gen-only Step (纯生成步)*:
```
调用 run_static(mode="static_gen", bs=gen_tokens, isl=isl+osl//2, osl=2)
genonly_step_latency = Σ generation_ops.query(database, ...)
```

**Step 4: 计算 TTFT (首 token 延迟)**

```
                    mix_step_latency
prefill_step_ms = ──────────────────
                    mix_efficiency

                              ⎡ isl ⎤
ttft_per_request  = prefill_step_ms × ⎢ ─── ⎥ + dispatch_overhead
                              ⎣ctx_tokens⎦

queuing_factor = min(2 + (steps_to_finish_ctx - 3) / 20, 4)

ttft = encoder_latency + ttft_per_request × queuing_factor
```

**对于本命令** (isl=256, ctx_tokens=256, b=1):
```
prefill_step_ms = mix_step_latency (假设 ~X ms)
ttft_per_request = X × ceil(256/256) + 0 = X ms
queuing_factor = min(2 + (1-3)/20, 4) = 1.9
ttft = 0 + X × 1.9 = 1.9X ms
```

**Step 5: 计算 TPOT (每 token 延迟)**

```
                    mix_step_latency × num_mix_steps_for_tpot
                 +  genonly_step_latency × num_genonly_steps
tpot = ─────────────────────────────────────────────────────────────
                    (num_mix_steps_for_tpot + num_genonly_steps)
                 ×  decode_tokens_per_iteration
```

**Step 6: 计算吞吐量**

```
total_step_latency = encoder_latency
                   + num_mix_steps × mix_step_latency
                   + num_genonly_steps × genonly_step_latency

                     1000
step_throughput = ────────────── × batch_size × (osl - 1)
                  total_step_latency

output_throughput = throughput_cap(step_throughput)

scale_factor = pp_size × attention_dp_size
output_throughput *= scale_factor  # 跨所有 GPU 的总 tokens/s

tokens_s_gpu = output_throughput / (pp × tp × dp × cp)
request_rate = output_throughput / (osl - 1)
tokens_s_user = 1000.0 / tpot
request_latency = ttft + tpot × max(osl - 1, 0)
```

**Step 7: 内存检查**

```
1. 权重内存:
   weights = Σ op.get_weights() / pp_size

2. 激活内存:
   h = num_heads × head_size
   activations = 2 × num_tokens × h × coeffs[tp_size]

3. KV 缓存:
   # 标准 GQA/MHA:
   kvcache_per_seq = 2 × ceil(num_kv_heads / tp) × head_dim × num_layers × dtype_bytes × seq_tokens
   # MLA (DeepSeek):
   kvcache_per_seq = num_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes × seq_tokens

   kvcache = batch_size × kvcache_per_seq

4. NCCL 和其他开销:
   nccl_mem = system_spec["misc"]["nccl_mem"][tp_size]
   others_mem = system_spec["misc"]["other_mem"]

5. 总内存:
   total_gib = (weights + activations + kvcache + nccl_mem + others_mem) / (1 << 30)

6. OOM 检查:
   # vLLM 使用 OfTotal 模式:
   budget = capacity × fraction × (1 - reserved) × (1 - tolerance) - non_kv
   if kvcache > budget → KV Cache OOM → break 内层循环
```

#### 6.1.3 Agg 数学公式汇总

| 指标 | 公式 | 说明 |
|------|------|------|
| **TTFT** | `encoder + prefill_step × ⌈isl/ctx⌉ × queuing_factor` | 首 token 延迟 |
| **TPOT** | `(mix_steps × mix_lat + gen_steps × gen_lat) / total_steps / decode_tokens_per_iter` | 每 token 延迟 |
| **Throughput** | `1000 / total_step_latency × b × (osl-1)` | 单副本吞吐 (tokens/s) |
| **tokens/s/gpu** | `throughput × pp × dp / (pp × tp × dp × cp)` | 每 GPU 吞吐 |
| **req/s** | `output_throughput / (osl - 1)` | 请求速率 |
| **request_latency** | `ttft + tpot × (osl - 1)` | 端到端延迟 |
| **KV Cache** | `bs × 2 × ⌈kv_heads/tp⌉ × head_dim × layers × dtype × seq` | 标准 GQA |
| **KV Cache (MLA)** | `layers × (kv_lora_rank + qk_rope_head_dim) × dtype × seq` | DeepSeek MLA |
| **Queuing Factor** | `min(2 + (steps-3)/20, 4)` | 批量排队系数 |

---

### 6.2 Disagg 模式 Sweep (分离推理)

#### 6.2.1 整体流程

```
sweep_disagg()
│
├─ Step 1: 预计算 Prefill Worker 候选
│   _get_disagg_worker_candidates(role="prefill")
│   for parallel_config in prefill_configs:
│       for batch_size in [1,2,...,15,16,...,512]:
│           predict_disagg_worker(role="prefill")
│               → backend.run_static(mode="static_ctx")
│               → 查询 context_ops 性能数据库
│               → 计算 TTFT = Σ context_ops.latency
│           if not OOM → 加入 prefill_candidates
│
├─ Step 2: 预计算 Decode Worker 候选
│   _get_disagg_worker_candidates(role="decode")
│   for parallel_config in decode_configs:
│       for batch_size in [1,2,...,512]:
│           predict_disagg_worker(role="decode")
│               → backend.run_static(mode="static_gen")
│               → 查询 generation_ops 性能数据库
│               → 计算 TPOT = Σ generation_ops.latency / (osl-1)
│           if not OOM → 加入 decode_candidates
│
├─ Step 3: 遍历约束对 (ttft_target, tpot_target)
│   for (ttft_c, tpot_c) in [(2000, 26)]:
│
│   └─ _find_best_disagg_under_constraint()
│       ├─ 过滤 prefill: ttft × 1.8 < ttft_target
│       ├─ 过滤 decode: tpot < tpot_target
│       └─ 交叉匹配 → 选择最优 (p_num, d_num)
│
└─ Step 4: 速率匹配 → 输出结果
```

#### 6.2.2 Per-Worker 延迟计算

**Prefill Worker** (`run_static(mode="static_ctx")`):
```
context_latency = Σ for op in context_ops:
    op.query(database, x=batch_size × effective_isl, s=effective_isl)

encoder_latency = Σ encoder_ops (文本模型为 0)

ttft = encoder_latency + context_latency
tpot = 0  (prefill 阶段不产生 decode token)
```

**Decode Worker** (`run_static(mode="static_gen")`):
```
generation_latency = Σ for i in range(0, osl-1, stride):
    for op in generation_ops:
        op.query(database, x=batch_size, s=isl+i+1) × min(stride, osl-1-i)

tpot = generation_latency / (osl - 1)
ttft = 0  (decode 阶段不处理新请求的 prefill)
```

#### 6.2.3 速率匹配 (Rate Matching)

给定一个 prefill worker (p) 和 decode worker (d)，寻找最优的 worker 数量组合：

```
优化目标: max tokens/s/gpu

约束:
  prefill_gpus × p_num + decode_gpus × d_num = total_gpus (64)

计算:
  p_effective = p.seq_s × p_num × 0.9     # prefill 衰减因子 0.9
  d_effective = d.seq_s × d_num × 0.92    # decode 衰减因子 0.92

  system_throughput = min(p_effective, d_effective)  # 瓶颈取最小

  tokens/s = system_throughput × osl
  tokens/s/gpu = tokens/s / total_gpus

搜索算法:
  for d_num in [1..32]:
      for p_num in [1..32]:
          if prefill_gpus × p_num + decode_gpus × d_num != 64:
              continue
          tpg = min(p×p_num×0.9, d×d_num×0.92) / 64
          if tpg > best_tpg:
              best_tpg = tpg
              best_p_num, best_d_num = p_num, d_num
```

#### 6.2.4 Disagg 数学公式汇总

| 指标 | 公式 | 说明 |
|------|------|------|
| **Prefill TTFT** | `Σ context_ops.query(bs×isl, isl)` | 单个 prefill worker 延迟 |
| **Decode TPOT** | `Σ generation_ops.latency / (osl-1)` | 单个 decode worker 每 token 延迟 |
| **p_effective** | `prefill.seq_s × p_num × 0.9` | 有效 prefill 吞吐 (含衰减) |
| **d_effective** | `decode.seq_s × d_num × 0.92` | 有效 decode 吞吐 (含衰减) |
| **system_throughput** | `min(p_effective, d_effective)` | 系统吞吐 (瓶颈) |
| **tokens/s** | `system_throughput × osl` | 总输出 tokens/s |
| **tokens/s/gpu** | `tokens/s / total_gpus` | 每 GPU 效率 |
| **request_latency** | `ttft_p + tpot_d × (osl-1)` | 端到端延迟 |

**衰减因子含义**:
- `prefill_degradation = 0.9`: 多 prefill worker 扩展时，考虑负载均衡不完美、KV 传输开销
- `decode_degradation = 0.92`: 多 decode worker 扩展时，考虑调度开销和网络争用
- 传输延迟不显式建模，通过衰减因子隐式吸收

#### 6.2.5 Worker 匹配示例

假设 (简化):
```
prefill worker: TP=4, PP=1 → 4 GPU/worker, seq_s=100 req/s, ttft=500ms
decode worker:  TP=4, PP=1 → 4 GPU/worker, seq_s=80 req/s,  tpot=20ms
total_gpus = 64
```

搜索最优分配:
```
p_num=8, d_num=8:
  prefill_gpus = 4×8 = 32, decode_gpus = 4×8 = 32, total = 64 ✓
  p_eff = 100 × 8 × 0.9 = 720 req/s
  d_eff = 80 × 8 × 0.92 = 588.8 req/s
  system = min(720, 588.8) = 588.8 req/s
  tokens/s/gpu = 588.8 × 256 / 64 = 2355.2

p_num=12, d_num=4:
  prefill_gpus = 4×12 = 48, decode_gpus = 4×4 = 16, total = 64 ✓
  p_eff = 100 × 12 × 0.9 = 1080 req/s
  d_eff = 80 × 4 × 0.92 = 294.4 req/s
  system = min(1080, 294.4) = 294.4 req/s
  tokens/s/gpu = 294.4 × 256 / 64 = 1177.6

→ 最优: p_num=8, d_num=8 (tokens/s/gpu 更高)
```

---

### 6.3 Agg vs Disagg 选择

两个 sweep 并行执行后，选择吞吐量更高的模式：

```
agg_best_throughput = max(agg_results["tokens/s/gpu"])
disagg_best_throughput = max(disagg_results["tokens/s/gpu"])

if agg_best_throughput > disagg_best_throughput:
    chosen = "agg"
else:
    chosen = "disagg"
```

---

## 7. 阶段五：结果处理与 Pareto 选择

### 7.1 SLA 过滤

```
agg_results_filtered = agg_results[
    (agg_results["tpot"] <= 26) &      # TPOT SLA
    (agg_results["ttft"] <= 2000)      # TTFT SLA
]
```

### 7.2 Cluster 效率调整

```
tokens_s_gpu_cluster = tokens_s_gpu × ⌊total_gpus / num_total_gpus⌋ × num_total_gpus / total_gpus
```

这个公式考虑了整数副本打包：如果配置需要 10 GPU/副本，64 GPU 只能装 6 个副本 (60 GPU)，剩余 4 GPU 浪费。

### 7.3 Pareto 前沿计算

```
def is_pareto(costs):
    # 标准非支配排序 O(n²)
    is_better = np.ones(len(costs), dtype=bool)
    for i, c in enumerate(costs):
        if is_better[i]:
            # 在剩余点中，只要有一个点在所有维度上都 ≥ c，则 c 被支配
            is_better[is_better] = np.any(costs[is_better] > c, axis=1)
            is_better[i] = True
    return is_better

# Pareto 轴:
# X = tokens/s/user (越大越好) 或 request_latency (越小越好)
# Y = tokens/s/gpu_cluster (越大越好)
```

### 7.4 Top-N 选择

```
1. 过滤: tpot ≤ target_tpot
2. 按 parallel 分组 (相同的 tp/pp/dp/moe_ep)
3. 每组取 tokens/s/gpu_cluster 最高的 1 个
4. 全局按 tokens/s/gpu_cluster 降序排列
5. 取前 N=8 个
```

### 7.5 结果合并 (backend="vllm" 非 auto 时跳过)

当 `backend="auto"` 时，会有多组结果 (agg_trtllm, agg_vllm, agg_sglang, ...)。
合并逻辑：

```
agg_experiments = [agg_trtllm, agg_vllm, agg_sglang]
merged_agg = concat(agg_experiments)
merged_agg = merged_agg.sort_values("tokens/s/gpu_cluster", ascending=False).head(top_n)

# 重新计算合并后的 Pareto 前沿
merged_pareto = get_pareto_front(merged_agg)
```

---

## 8. 阶段六：输出与保存

### 8.1 终端输出结构

```
╔══════════════════════════════════════════════════════════════════╗
║  Model: deepseek-ai/DeepSeek-V4-Flash (MoE)                    ║
║  System: b300_sxm | Backend: vllm | Total GPUs: 64             ║
║  ISL: 256 | OSL: 256 | TTFT: 2000ms | TPOT: 26ms              ║
╠══════════════════════════════════════════════════════════════════╣
║  Chosen: agg (vllm)                                             ║
║  Best Throughput: XXXX tokens/s                                 ║
║  agg 1.XXx better than disagg                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Overall Best Configuration:                                    ║
║  ├─ Best Throughput:     XXXX tokens/s                          ║
║  ├─ Per-GPU Throughput:  XXX tokens/s/gpu                       ║
║  ├─ Per-User Throughput: XX tokens/s/user                       ║
║  ├─ Request Rate:        XX req/s                               ║
║  ├─ TTFT:               XXX ms                                  ║
║  ├─ TPOT:               XX ms                                   ║
║  └─ Request Latency:    XXXXX ms                                ║
╠══════════════════════════════════════════════════════════════════╣
║  [Pareto Frontier ASCII Plot]                                   ║
║  tokens/s/gpu_cluster vs tokens/s/user                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Top-8 Configurations:                                          ║
║  Rank │ tokens/s/gpu │ TTFT │ TPOT │ GPUs │ Parallel │ BS       ║
║  ─────┼──────────────┼──────┼──────┼──────┼──────────┼──────    ║
║  1    │ XXX          │ XXX  │ XX   │ XX   │ tp=X pp=X│ XX       ║
║  2    │ ...          │ ...  │ ...  │ ...  │ ...      │ ...      ║
║  ...                                                              ║
╚══════════════════════════════════════════════════════════════════╝
```

### 8.2 保存文件结构

```
results/DeepSeek-V4-Flash_b300_sxm_vllm_isl256_osl256_ttft2000_tpot26_XXXXXX/
├── pareto_frontier.png              # Matplotlib Pareto 图
├── agg_vllm/                        # Agg 模式结果
│   ├── best_config_topn.csv         # Top-8 配置 CSV
│   ├── pareto.csv                   # 完整 Pareto 前沿 CSV
│   ├── exp_config.yaml              # 实验配置 (可复现)
│   ├── top1/                        # Rank-1 部署产物
│   │   ├── generator_config.yaml    # 生成器配置
│   │   ├── per_ops_source.json      # 每个算子的数据来源
│   │   └── <backend_configs>        # 后端特定配置 (如 engine YAML)
│   ├── top2/
│   │   └── ...
│   └── ...top8/
└── disagg_vllm/                     # Disagg 模式结果
    ├── best_config_topn.csv
    ├── pareto.csv
    └── ...
```

### 8.3 输出指标说明

| 指标 | 来源 | 说明 |
|------|------|------|
| `tokens/s/gpu` | sweep 结果 | 每 GPU 吞吐量，主要排序依据 |
| `tokens/s/gpu_cluster` | picking 计算 | 考虑副本打包效率的集群每 GPU 吞吐 |
| `tokens/s/user` | `1000 / tpot` | 每用户每秒 token 数 |
| `tokens/s` (total) | `tokens/s/gpu × total_gpus` | 集群总吞吐 |
| `req/s` | `output_throughput / (osl-1)` | 集群请求速率 |
| `TTFT` | sweep 计算 | 首 token 延迟 (ms) |
| `TPOT` | sweep 计算 | 每 token 延迟 (ms) |
| `request_latency` | `ttft + tpot × (osl-1)` | 端到端延迟 (ms) |
| `concurrency` | `per_replica_concurrency × replicas` | 并发用户数 |
| `total_gpus` | 配置决定 | 实际使用的 GPU 数 |
| `replicas` | `total_gpus / gpus_per_replica` | 副本数 |
| `parallel` | 配置决定 | 并行策略 (tp/pp/dp/etp/ep) |
| `bs` | sweep 决定 | 最优批量大小 |
| `power_w` | 性能数据库 | 功率消耗 (W) |

---

## 9. 附录：关键数据结构与常量

### 9.1 RuntimeConfig

```python
@dataclass
class RuntimeConfig:
    isl: int = 4000           # 输入序列长度
    osl: int = 1000           # 输出序列长度
    ttft: float = 2000.0      # TTFT SLA (ms)
    tpot: float | list = 30.0 # TPOT SLA (ms)，可以是列表
    request_latency: float | None = None
    prefix: int = 0           # 前缀缓存长度
    beam_width: int = 1
    batch_size: int | None = None
    # VL 模型参数
    image_height: int = 0
    image_width: int = 0
    num_images: int = 1
```

### 9.2 ModelConfig

```python
@dataclass
class ModelConfig:
    tp_size: int = 1
    pp_size: int = 1
    attention_dp_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 1
    cp_size: int = 1
    gemm_quant_mode: str = "bfloat16"
    moe_quant_mode: str = "bfloat16"
    kvcache_quant_mode: str = "bfloat16"
    fmha_quant_mode: str = "bfloat16"
    comm_quant_mode: str = "bfloat16"
```

### 9.3 Parallel Config

```python
# 六元组: (tp, pp, dp, moe_tp, moe_ep, cp)
parallel_config = (4, 1, 1, 1, 1, 1)  # 示例: TP=4, 其他=1
```

### 9.4 关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `RATE_MATCH_PREFILL_DEGRADATION` | 0.9 | Disagg prefill 扩展衰减 |
| `RATE_MATCH_DECODE_DEGRADATION` | 0.92 | Disagg decode 扩展衰减 |
| `AUTOSCALE_TTFT_CORRECTION_FACTOR` | 1.8 | Autoscale TTFT 队列修正 |
| `MAX_DECODE_WORKERS_PER_CATEGORY` | 16 | 每组最大 decode 候选数 |
| `MAX_PREFILL_WORKERS` | 32 | 最大 prefill 候选数 |
| `DEFAULT_MAX_BATCH_SIZE` | 512 | 默认最大批量大小 |
| `DEFAULT_CTX_STRIDE` | 512 | 默认 ctx_tokens 步长 |

### 9.5 性能数据库操作类型

| 操作类型 | 查询键 | 用途 |
|----------|--------|------|
| GEMM | (tp, quant, m, n, k) | 矩阵乘法延迟 |
| Context Attention | (tp, fmha_quant, bs, isl) | Prefill 注意力 |
| Generation Attention | (tp, fmha_quant, bs, seq) | Decode 注意力 |
| Context MLA | (tp, bs, isl) | DeepSeek MLA Prefill |
| Generation MLA | (tp, bs, seq) | DeepSeek MLA Decode |
| MoE | (tp, quant, bs, seq, experts) | MoE 层延迟 |
| NCCL | (tp, comm_quant, msg_size) | 通信延迟 |
| Custom AllReduce | (tp, msg_size) | 自定义 AllReduce |

### 9.6 错误处理

| 异常 | 触发条件 | 含义 |
|------|----------|------|
| `InsufficientMemoryError` | 所有并行配置都 OOM | 模型无法放入 GPU 内存 |
| `KVCacheCapacityError` | 模型可放入但 KV 缓存超出预算 | 需要更多 GPU 或更小批量 |
| `NoFeasibleConfigError` | 内存 OK 但无满足 SLA 的配置 | 需要放宽 SLA 约束 |
| `RuntimeError` | 未知异常 | 内部错误 |

---

## 附录 A: 完整数据流图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  HF Config   │     │ System YAML  │     │ CLI Args     │
│  (模型架构)   │     │ (GPU 规格)   │     │ (用户输入)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│              build_default_tasks()                        │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │ ModelConfig  │ │ SearchSpace │ │ RuntimeConfig     │  │
│  │ (量化/并行)  │ │ (TP/PP/BS)  │ │ (ISL/OSL/SLA)    │  │
│  └──────┬──────┘ └──────┬──────┘ └────────┬──────────┘  │
│         └───────────────┼─────────────────┘              │
│                         ▼                                │
│              ┌─────────────────────┐                     │
│              │    Task 对象         │                     │
│              │ serving_mode=agg    │                     │
│              │ serving_mode=disagg │                     │
│              └─────────┬───────────┘                     │
└────────────────────────┼─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Perf DB    │ │ Perf DB    │ │ Perf DB    │
   │ b300_sxm   │ │ b300_sxm   │ │ b300_sxm   │
   │ vllm/0.24.0│ │ vllm/0.24.0│ │ vllm/0.24.0│
   │ GEMM ops   │ │ Attn ops   │ │ MoE ops    │
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────────────────────────────────────────────────┐
│                    Task.run()                             │
│                                                          │
│  ┌─── sweep_agg() ──────────────────────────────────┐   │
│  │  for parallel_config:                             │   │
│  │    for batch_size:                                │   │
│  │      for ctx_tokens:                              │   │
│  │        backend.run_agg()                          │   │
│  │          ├─ run_mixed() → 3-pass 混合步延迟       │   │
│  │          ├─ genonly_step → 纯生成步延迟           │   │
│  │          ├─ TTFT = prefill_step × ⌈isl/ctx⌉ × Q  │   │
│  │          ├─ TPOT = weighted_avg(mix, genonly)     │   │
│  │          ├─ throughput = 1000/total_latency × b   │   │
│  │          └─ memory_check → OOM?                   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─── sweep_disagg() ───────────────────────────────┐   │
│  │  prefill_workers = enumerate(prefill_configs)     │   │
│  │  decode_workers  = enumerate(decode_configs)      │   │
│  │  for (ttft, tpot) constraint:                     │   │
│  │    filter prefill: ttft×1.8 < target              │   │
│  │    filter decode:  tpot < target                  │   │
│  │    cross_product → rate_match(p_num, d_num)       │   │
│  │      system = min(p×p_num×0.9, d×d_num×0.92)     │   │
│  │      tokens/s/gpu = system×osl / total_gpus       │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  process_experiment_result()                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ SLA 过滤      │ │ Pareto 前沿  │ │ Top-N 选择       │ │
│  │ TPOT≤26      │ │ 非支配排序   │ │ 按 gpu_cluster   │ │
│  │ TTFT≤2000    │ │              │ │ 排序取前 8       │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
│                                                          │
│  chosen_exp = max(agg_throughput, disagg_throughput)     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  log_final_summary() + save_results()                    │
│  ┌────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │ 终端摘要    │ │ Pareto 图   │ │ CSV/YAML/PNG 保存   │ │
│  │ Top-8 表格 │ │ ASCII plot  │ │ 部署配置生成        │ │
│  └────────────┘ └─────────────┘ └─────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```
