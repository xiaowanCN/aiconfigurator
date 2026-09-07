基于2026.09.02为最新拉取分支分析：

# aiconfigurator 架构深度剖析

> 以 CLI 命令为切入点，追踪从用户输入到性能预测输出的完整执行路径。

## 1. 引言与文档约定

### 1.1 文档目的

本文以一条典型 CLI 命令为线索，逐层剖析 aiconfigurator 的内部架构：

```bash
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --total-gpus 8 \
  --system b300_sxm \
  --backend vllm \
  --isl 8000 --osl 1000 \
  --ttft 3000 --tpot 30 \
  --top-n 16 \
  --backend-version 0.24.0 \
  --target-concurrency 2048
```

### 1.2 项目定位

aiconfigurator 是 NVIDIA Dynamo 生态中的 GPU 部署配置优化工具。它通过性能建模（而非实际部署）来预测不同并行策略、批处理大小、量化模式下的推理延迟和吞吐量，帮助用户在部署前选择最优配置。

### 1.3 术语表

| 术语 | 含义 |
|------|------|
| **Agg** | Aggregated serving，聚合模式，prefill 和 decode 在同一 worker |
| **Disagg** | Disaggregated serving，分离模式，prefill 和 decode 在不同 worker |
| **AFD** | Attention-FFN-Decoupled，注意力与 FFN 解耦的分离模式 |
| **TP** | Tensor Parallelism，张量并行 |
| **PP** | Pipeline Parallelism，流水线并行 |
| **DP** | Data Parallelism，数据并行（attention_dp_size） |
| **EP** | Expert Parallelism，专家并行（MoE 模型专用） |
| **TTFT** | Time To First Token，首 token 延迟 |
| **TPOT** | Time Per Output Token，每 token 延迟 |
| **ISL** | Input Sequence Length，输入序列长度 |
| **OSL** | Output Sequence Length，输出序列长度 |
| **SOL** | Speed of Light，理论峰值性能 |
| **PerfDatabase** | 性能数据库，存储算子级延迟数据 |
| **MoE** | Mixture of Experts，混合专家模型 |
| **MTP** | Multi-Token Prediction，多 token 预测（投机解码） |
| **KV Cache** | Key-Value Cache，注意力键值缓存 |

## 2. 项目总体架构

### 2.1 仓库结构概览

项目采用**双包架构**，分为四个核心模块：

```
aiconfigurator/
├── aic-core/                          # 模块 1: 性能建模核心 (aiconfigurator-core)
│   ├── src/aiconfigurator_core/       #   Python 核心 SDK
│   │   ├── sdk/                       #     性能建模逻辑
│   │   │   ├── backends/              #       后端抽象 (vllm/sglang/trtllm)
│   │   │   ├── models/                #       LLM 模型定义
│   │   │   ├── operations/            #       算子延迟模型
│   │   │   ├── perf_database.py       #       性能数据库加载
│   │   │   ├── engine.py              #       引擎仿真核心
│   │   │   └── memory.py              #       内存建模
│   │   ├── systems/                   #     系统规格 YAML + 性能数据 parquet
│   │   └── model_configs/             #     模型配置 JSON (100+ 模型)
│   └── rust/aiconfigurator-core/      #   Rust 扩展 (PyO3)
│       └── src/                       #     算子引擎步进估算 (高性能)
│
├── src/aiconfigurator/                # 模块 2: 上层编排 (aiconfigurator)
│   ├── main.py                        #   CLI 入口点
│   ├── cli/                           #   CLI 参数解析与路由
│   ├── sdk/                           #   SDK 兼容层 (task_v2, sweep, picking)
│   └── generator/                     #   部署配置生成器
│
├── collector/                         # 模块 3: GPU 性能数据收集 (独立)
│   ├── sglang/                        #   SGLang 算子收集
│   ├── trtllm/                        #   TRT-LLM 算子收集
│   ├── vllm/                          #   vLLM 算子收集
│   └── network/                       #   网络通信收集
│
└── tools/                             # 模块 4: 工具链
    ├── support_matrix/                #   支持矩阵生成
    ├── perf_database/                 #   性能数据库工具
    └── generator_validator/           #   Generator 输出校验
```

### 2.2 双包架构

| 包 | PyPI 名 | 构建方式 | 职责 |
|---|---|---|---|
| `aic-core/` | `aiconfigurator-core` | maturin (Python + Rust) | 性能建模核心，可独立分发 |
| `src/aiconfigurator/` | `aiconfigurator` | setuptools | 上层编排，依赖 aic-core |

`aiconfigurator-core` 是可独立分发的建模核心，包含 Rust 扩展模块 `aiconfigurator_core._aiconfigurator_core`（通过 PyO3 编译）。上层包通过 `aiconfigurator-core` 的 Python API 调用核心功能。

### 2.3 核心数据流总览

```
用户 CLI 输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ CLI 入口 (main.py → cli/main.py)                            │
│   参数解析 → 模式分发 (default/exp/generate)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Task 构建 (task_v2.py)                                       │
│   build_default_tasks() → Task dataclass                     │
│   __post_init__: 模型解析、量化模式、并行候选、版本路由          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 配置空间搜索 (sweep.py)                                       │
│   sweep_agg(): 并行 × batch × ctx_tokens 枚举                │
│   sweep_disagg(): prefill_parallel × decode_parallel 匹配    │
│     │                                                        │
│     ├── predict_agg_worker() / predict_disagg_worker()       │
│     │     │                                                  │
│     │     ▼                                                  │
│     │   后端工厂 (factory.py) → BaseBackend                   │
│     │     │                                                  │
│     │     ▼                                                  │
│     │   PerfDatabase 查询 + Rust 引擎步进估算                 │
│     │     │                                                  │
│     │     ▼                                                  │
│     │   InferenceSummary (TTFT/TPOT/吞吐量/内存)              │
│     │                                                        │
│     ▼                                                        │
│   速率匹配 (disagg) + SLA 过滤                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pareto 分析与择优 (picking.py)                               │
│   get_pareto_front() → 前沿计算                               │
│   pick_default() / pick_autoscale() → 最优配置                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 结果输出 (cli/report_and_save.py)                            │
│   终端表格 + CSV 保存 + Generator 渲染                        │
└─────────────────────────────────────────────────────────────┘
```

## 3. CLI 入口与参数解析

### 3.1 入口点链路

CLI 入口通过 `pyproject.toml` 注册：

```toml
[project.scripts]
aiconfigurator = "aiconfigurator.main:main"
```

调用链路：

```
aiconfigurator.main:main()           # 顶层分发: cli / version
  └── aiconfigurator.cli.main:main() # CLI 子命令解析
      └── configure_parser()         # 注册 default/exp/generate 子命令
```

`main.py:main()` 是顶层入口，处理 `cli` 和 `version` 两个子命令。当用户执行 `aiconfigurator cli default ...` 时：

1. `main.py` 识别 `cli` 子命令，调用 `_run_cli()`
2. `_run_cli()` 先尝试 Generator CLI helper（处理 `generate`/`render-config` 等）
3. 若非 Generator 命令，构造 `cli/main.py` 的 parser 并解析剩余参数
4. `cli_main(cli_args)` 根据 `args.mode` 分发到对应处理函数

### 3.2 两级分发机制

```
aiconfigurator cli <mode> [options]
         │         │
         │         └── 第二级: mode 分发 (default/exp/generate)
         └── 第一级: 子命令分发 (cli/version)
```

`configure_parser()` 在 `cli/main.py` 中注册三个 mode 子命令：

| mode | 函数 | 用途 |
|------|------|------|
| `default` | `build_default_tasks()` + sweep | 快速估算，自动搜索最优配置 |
| `exp` | YAML 实验配置加载 | 精细控制，支持多组实验 |
| `generate` | Generator 渲染管线 | 生成部署配置文件 |

### 3.3 default 模式参数解析

`_add_default_mode_arguments()` 注册 default 模式的核心参数：

| 参数 | 类型 | 含义 |
|------|------|------|
| `--model-path` / `--model` | str (必选) | HuggingFace 模型路径或本地路径 |
| `--total-gpus` | int | 总 GPU 数量 |
| `--system` | str (必选) | 系统名称 (如 b300_sxm, h200_sxm) |
| `--backend` | str | 后端名称 (trtllm/vllm/sglang/auto) |
| `--isl` / `--osl` | int | 输入/输出序列长度 |
| `--ttft` / `--tpot` | float | SLA 约束 (ms) |
| `--top-n` | int | 输出前 N 个最优配置 |
| `--backend-version` | str | 后端版本号 |
| `--target-concurrency` | float | 目标并发数 |
| `--serving-mode` | str | 服务模式 (auto/all/agg/disagg/afd) |

### 3.4 关键参数映射

CLI 参数到内部数据结构的映射：

```
CLI --model-path    → Task.model_path
CLI --system        → Task.system_name
CLI --backend       → Task.backend_name
CLI --isl/--osl     → RuntimeConfig.isl / RuntimeConfig.osl
CLI --ttft/--tpot   → RuntimeConfig.ttft / RuntimeConfig.tpot
CLI --total-gpus    → Task.total_gpus → 并行配置枚举
CLI --backend-version → Task.backend_version → PerfDatabase 版本路由
```

## 4. Task 构建与配置

### 4.1 build_default_tasks() 详解

`default` 模式的核心是 `build_default_tasks()` 函数，它将 CLI 参数转换为一个或多个 `Task` 对象。

流程：
1. 从 CLI 参数构造 `Task` dataclass
2. `Task.__post_init__()` 自动完成：
   - 模型路径验证与 config.json 解析
   - 模型家族识别 (`get_model_family()`)
   - 量化模式推断 (`_infer_quant_modes_from_raw_config()`)
   - 后端版本解析与校验
   - 并行配置候选列表生成 (`enumerate_parallel_config()`)
   - MoE 通信后端解析 (`resolve_model_config_moe_comm()`)
   - 投机解码配置 (`normalize_speculative_decoding()`)

### 4.2 Task 数据类设计

`Task` 是一个扁平 dataclass，遵循 SGLang 风格设计：

```python
@dataclass
class Task:
    # 模型标识
    model_path: str
    system_name: str
    backend_name: str
    backend_version: str | None = None

    # SLA 约束
    isl: int = 8000
    osl: int = 1000
    ttft: float = 3000.0
    tpot: float = 30.0

    # 资源约束
    total_gpus: int | None = None
    target_concurrency: float | None = None

    # 量化模式
    gemm_quant_mode: GEMMQuantMode = GEMMQuantMode.bfloat16
    kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.bfloat16
    # ... 更多量化模式

    # 并行候选
    tp_candidates: list[int] | None = None
    pp_candidates: list[int] | None = None
    # ...

    # 分离模式前缀字段
    prefill_system_name: str | None = None
    decode_system_name: str | None = None
    # ...
```

设计原则：
- **扁平结构**：无嵌套 DefaultMunch，无 deep_merge
- **`__post_init__` 一站式解析**：构造后所有字段都有具体值
- **严格前缀纪律**：disagg 模式下，顶层 worker-spec 字段不可用，必须使用 `prefill_*` / `decode_*` 前缀

### 4.3 `__post_init__()` 配置解析流程

```
Task.__post_init__()
  ├── _validate_model_path()           # 验证模型路径
  ├── get_model_family()               # 识别模型家族 (DEEPSEEKV4, LLAMA, ...)
  ├── _infer_quant_modes_from_raw_config()  # 从 config.json 推断量化模式
  ├── get_latest_database_version()    # 解析后端版本
  ├── enumerate_parallel_config()      # 生成并行配置候选列表
  ├── resolve_model_config_moe_comm()  # 解析 MoE 通信后端
  ├── normalize_speculative_decoding() # 规范化投机解码配置
  └── build_*_parallel_lists()         # 构建 agg/disagg 并行搜索空间
```

## 5. 性能数据库 (PerfDatabase)

### 5.1 系统规格 (SystemSpec) 与 YAML 规范

每个 GPU 系统由一个 YAML 文件定义，位于 `aic-core/src/aiconfigurator_core/systems/`：

```yaml
# b300_sxm.yaml 示例
data_dir: data/b300_sxm          # 性能数据目录 (相对于 systems/)
gpu:
  mem_bw: 7750000000000           # 显存带宽 (7.75 TB/s)
  mem_capacity: 288400343040      # 显存容量 (275 GiB)
  bfloat16_tc_flops: 2250000000000000  # BF16 Tensor Core 算力
  fp8_tc_flops: 4500000000000000       # FP8 Tensor Core 算力
  fp4_tc_flops: 14000000000000000      # FP4 Tensor Core 算力
  power: 1100                     # 功耗 (W)
  sm_version: 103                 # SM 架构版本
node:
  num_gpus_per_node: 8            # 每节点 GPU 数
  inter_node_bw: 100000000000     # 节点间带宽 (100 GB/s)
  intra_node_bw: 3000000000000    # 节点内带宽 (3 TB/s NVLink)
  pcie_bw: 128000000000           # PCIe 带宽
  p2p_latency: 0.00005            # P2P 延迟 (50μs)
misc:
  nccl_mem:                       # NCCL 内存开销 (按 TP 大小)
    1: 0
    2: 358612992
    4: 411041792
    8: 411041792
  other_mem: 3758096384           # CUDA/cuBLAS 等运行时开销 (3.5 GiB)
  nccl_version: '2.27'
```

关键字段说明：

| 字段 | 用途 | 估算模型中的位置 |
|------|------|------------------|
| `gpu.mem_capacity` | OOM 检查阈值 | `_get_memory_usage()` |
| `gpu.mem_bw` | 算子延迟估算 | `Operation.get_latency()` |
| `gpu.*_tc_flops` | GEMM 延迟估算 | `GEMMOperation` |
| `node.intra_node_bw` | 通信延迟估算 | `CommunicationOperation` |
| `misc.nccl_mem[tp]` | NCCL 内存开销 | `_get_memory_usage()` |
| `misc.other_mem` | 运行时内存开销 | `_get_memory_usage()` |

### 5.2 性能数据目录结构

```
aic-core/src/aiconfigurator_core/systems/data/<system>/
├── <op_family>/                    # 算子族 (attention, gemm, moe, ...)
│   └── <backend>/                  # 后端 (vllm, sglang, trtllm)
│       └── <version>/             # 版本 (0.24.0, 0.5.6.post2, ...)
│           └── <data>.parquet     # 算子延迟数据
├── comm/                          # 通信数据 (NCCL)
│   └── <backend>/<version>/
│       └── all_reduce.parquet
└── fpm_forward/                   # FPM 全模型前向数据
    └── <model>/<backend>/<version>/
```

### 5.3 PerfDatabase 加载机制

`PerfDatabase` 是性能数据库的核心类，负责加载和查询算子延迟数据：

```python
class PerfDatabase:
    def __init__(self, system, backend, version, systems_paths=None):
        self.system = system
        self.backend = backend
        self.version = version
        self.system_spec = load_system_spec(system)  # 加载 YAML
        self._load_perf_data()                        # 加载 parquet
```

加载流程：
1. `load_system_spec()` 从 YAML 加载系统规格
2. 版本路由：通过 `query_versions.yaml` 解析 current/previous/next 别名
3. 遍历 `data_dir` 下的 parquet 文件，按 `(op_family, backend, version)` 索引
4. 支持跨版本向后填充（backward fill）和跨后端数据复用

### 5.4 算子数据查询与插值

算子延迟查询通过 `Operation.get_latency()` 完成：

```python
# 查询流程
op = database.get_attention_op("context_attention", model_family="DEEPSEEKV4")
latency_ms = op.get_latency(batch_size=32, s=8000, h=7168, ...)
```

查询机制：
1. 根据算子类型和参数形状在 parquet 中查找匹配行
2. 若精确匹配不存在，使用插值（线性/对数）估算
3. 支持 FPM（Forward Performance Model）回归预测作为兜底
4. Rust 引擎步进估算器提供高性能批量查询

## 6. 后端与性能预测

### 6.1 后端工厂模式

后端通过工厂模式实例化：

```python
# aic-core/src/aiconfigurator_core/sdk/backends/factory.py
def get_backend(backend_name: str) -> BaseBackend:
    backend_map = {
        common.BackendName.trtllm: TRTLLMBackend,
        common.BackendName.sglang: SGLANGBackend,
        common.BackendName.vllm: VLLMBackend,
    }
    return backend_map[common.BackendName[backend_name]]()
```

三个后端的区别：

| 后端 | 特化点 |
|------|--------|
| **TRTLLMBackend** | TRT-LLM 引擎特性：max_num_tokens、KV cache fraction、overlap scheduler |
| **VLLMBackend** | vLLM v1 调度器：serialised prefill、Little's Law 吞吐量上限、gpu_memory_utilization |
| **SGLANGBackend** | SGLang 调度器：chunked prefill、activation overhead、mem_fraction_static |

### 6.2 BaseBackend 核心能力

`BaseBackend`（83KB）是后端抽象基类，提供共享的推理估算逻辑：

**静态推理 (`run_static`)**：
- 用于 disagg 模式的 prefill/decode worker 评估
- 计算 encoder + context + generation 三阶段延迟
- 返回 `InferenceSummary`（含 TTFT、TPOT、吞吐量、内存）

**聚合推理 (`run_agg`)**：
- 用于 agg 模式的连续批处理评估
- 建模 mix step（混合 prefill/decode）和 genonly step（纯 decode）
- 计算 TTFT（含排队因子）、TPOT、吞吐量

**内存估算 (`_get_memory_usage`)**：
```
总内存 = weights + activations + kvcache + nccl_mem + others
```
- `weights`：模型权重（从 model ops 累加）
- `activations`：激活内存（与 batch_size × isl 相关）
- `kvcache`：KV 缓存（与 batch_size × seq_len 相关）
- `nccl_mem`：NCCL 通信缓冲（从 system_spec 查表）
- `others`：CUDA/cuBLAS 运行时开销

### 6.3 性能预测接口

`sweep.py` 中的 `predict_agg_worker()` 和 `predict_disagg_worker()` 是性能预测的统一入口：

```python
def predict_agg_worker(model, backend, database, runtime_config, ctx_tokens, ...):
    """预测单个 agg worker 的性能"""
    return backend.run_agg(model, database, runtime_config, ctx_tokens=ctx_tokens)

def predict_disagg_worker(model, backend, database, runtime_config, role, ...):
    """预测单个 disagg worker (prefill/decode) 的性能"""
    if role == "prefill":
        return backend.run_static(model, database, runtime_config, mode="static_ctx")
    else:  # decode
        return backend.run_static(model, database, runtime_config, mode="static_gen")
```

### 6.4 Rust 引擎步进估算

高性能估算通过 Rust 扩展实现：

```python
# aic-core/src/aiconfigurator_core/sdk/rust_engine_step.py
def estimate_static_latency_breakdown_with_rust(model, database, runtime_config, mode, ...):
    """Rust 编译的引擎步进估算器"""
    # 将 model ops 序列化为 JSON
    # 调用 Rust FFI 进行批量延迟计算
    # 返回 per-op 延迟字典
```

Rust 引擎步进估算器的优势：
- 批量处理所有算子的延迟查询和插值
- 避免 Python 解释器开销
- 支持并行计算

## 7. 配置空间搜索 (Sweep)

### 7.1 sweep_agg() 聚合搜索

`sweep_agg()` 枚举所有可行的 (并行配置, batch_size, ctx_tokens) 组合：

```python
def sweep_agg(model_path, runtime_config, database, backend_name,
              model_config, parallel_config_list, ...):
    for parallel_config in parallel_config_list:  # 遍历并行配置
        for tpot_value in tpot_list:               # 遍历 TPOT 目标
            _sweep_one_parallel_agg(                # 批次 × ctx_tokens 枚举
                model, backend, database,
                runtime_config, ...
            )
```

搜索空间：
- **并行维度**：tp × pp × dp × moe_tp × moe_ep × cp
- **批次维度**：1 → 1024（非均匀步长）
- **ctx_tokens 维度**：0 → max(8192, 2×isl)（自适应步长）

### 7.2 sweep_disagg() 分离搜索与速率匹配

`sweep_disagg()` 的搜索分为三步：

**Step 1: 枚举 prefill worker 候选**
```python
prefill_summary_df = _get_disagg_worker_candidates(
    role="prefill", b_list=prefill_batch_range, ...
)
```

**Step 2: 枚举 decode worker 候选**
```python
decode_summary_df = _get_disagg_worker_candidates(
    role="decode", b_list=decode_batch_range, ...
)
```

**Step 3: 速率匹配**
```python
for (ttft, tpot) in constraint_pairs:
    _find_best_disagg_under_constraint(
        prefill_summary_df, decode_summary_df,
        match_workers=_match_workers,  # 匹配 prefill/decode 数量
    )
```

速率匹配公式：
```
seq/s = min(
    prefill_seq/s × prefill_num_worker × prefill_degradation,
    decode_seq/s  × decode_num_worker  × decode_degradation
)
```

其中 `degradation` 是经验降级因子（prefill=0.9, decode=0.92）。

### 7.3 并行配置枚举

并行配置枚举在 `task_v2.py` 的 `enumerate_parallel_config()` 中完成：

```python
def enumerate_parallel_config(tp_list, pp_list, dp_list,
                               moe_tp_list, moe_ep_list, cp_list):
    """枚举所有合法的 (tp, pp, dp, moe_tp, moe_ep, cp) 组合"""
    for tp in tp_list:
        for pp in pp_list:
            for dp in dp_list:
                for moe_tp in moe_tp_list:
                    for moe_ep in moe_ep_list:
                        for cp in cp_list:
                            if is_valid_combination(tp, pp, dp, moe_tp, moe_ep, cp):
                                yield (tp, pp, dp, moe_tp, moe_ep, cp)
```

合法性约束：
- `tp × pp × dp × cp ≤ total_gpus`
- `moe_tp × moe_ep = tp`（MoE 模型）
- GPU 能被 `tp × pp × dp` 整除

### 7.4 内存可行性检查

每个搜索点都经过 OOM 检查：

```python
summary.set_memory_and_check_oom(
    memory,                          # 预估内存
    database.system_spec["gpu"]["mem_capacity"],  # GPU 显存容量
    free_gpu_memory_fraction=...,    # KV cache 预留比例
    kv_cache_reserved_fraction=...,  # KV cache 保留比例
    kv_cache_tolerance=...,          # 容差
)
```

OOM 检查逻辑：
1. `total_memory > mem_capacity` → 模型 OOM
2. `kvcache > available_for_kv` → KV cache OOM
3. 任一 OOM → 跳过该配置（单调性优化：更大的 batch 也一定 OOM）

## 8. Pareto 分析与择优

### 8.1 Pareto 前沿计算

`picking.py` 中的 `get_pareto_front()` 计算 Pareto 前沿：

```python
def get_pareto_front(df, x_col="tokens/s/gpu", y_col="request_latency"):
    """计算 (吞吐量/gpu, 延迟) 的 Pareto 前沿"""
    # 按吞吐量降序排序
    # 保留延迟递减的点
```

### 8.2 三种择优模式

| 模式 | 函数 | 适用场景 |
|------|------|----------|
| `pick_default` | 从 top-n 中选择 | `--total-gpus` 指定时 |
| `pick_load_match` | 匹配目标请求率 | `--target-request-rate` 指定时 |
| `pick_autoscale` | 自动缩放 GPU 数量 | `--target-concurrency` 指定时 |

`pick_autoscale` 的核心逻辑：
1. 固定 TTFT/TPOT 约束
2. 遍历不同 GPU 数量
3. 对每个 GPU 数量，找到满足 SLA 的最高吞吐量配置
4. 返回最优 (GPU 数量, 配置) 组合

## 9. 结果输出与配置生成

### 9.1 终端输出格式

CLI 输出包含：
- 模型信息摘要
- 系统/后端/版本信息
- top-n 配置表格（tokens/s/gpu、TTFT、TPOT、并行策略、内存占用）
- 详细延迟分解（可选）

### 9.2 Generator 6 阶段渲染管线

当使用 `--generate` 选项时，触发 Generator 渲染管线：

```
Stage 1: Input Parsing        (api.py)
  解析用户输入的 YAML/CLI 参数

Stage 2: Default Application   (rendering/schemas.py)
  从 deployment_config.yaml 应用默认值

Stage 3: Rule Evaluation       (rendering/rule_engine.py)
  通过 .rule 文件计算派生参数
  (max_batch_size, cuda_graph_sizes, ...)

Stage 4: Parameter Mapping     (rendering/engine.py)
  统一参数 → 后端 CLI 标志映射
  (backend_config_mapping.yaml)

Stage 5: Template Rendering    (rendering/engine.py)
  Jinja2 模板渲染
  (backend_templates/<backend>/cli_args.j2)

Stage 6: Artifact Emission     (artifacts.py)
  输出最终文件: k8s_deploy.yaml, run.sh, cli_args, engine_configs
```

### 9.3 Generator 关键配置文件

| 文件 | 用途 |
|------|------|
| `config/deployment_config.yaml` | 输入 schema：~54 个参数、默认值、约束 |
| `config/backend_config_mapping.yaml` | 统一参数 → 后端 CLI 标志映射 |
| `config/backend_version_matrix.yaml` | Dynamo 版本 → 后端版本映射 |
| `config/backend_templates/<backend>/` | Jinja2 模板 (cli_args, run.sh, k8s_deploy) |
| `rule_plugin/<backend>.rule` | 后端特定的计算规则 |

## 10. 模型抽象层

### 10.1 模型定义与注册机制

模型定义在 `aic-core/src/aiconfigurator_core/sdk/models/` 下，通过注册表管理：

```python
# models/__init__.py
_MODEL_REGISTRY = {
    "DEEPSEEK": DeepSeekModel,
    "DEEPSEEKV32": DeepSeekV32Model,
    "DEEPSEEKV4": DeepSeekV4Model,
    "LLAMA": LlamaModel,
    "QWEN35": Qwen35Model,
    # ...
}
```

每个模型类定义：
- `_num_layers`, `_num_heads`, `_head_size`, `_hidden_size`
- `_num_experts`, `_topk`（MoE 模型）
- `context_ops`, `generation_ops`（算子列表）
- `encoder_ops`（VL 模型的视觉编码器）

### 10.2 MoE 模型特殊处理

MoE 模型有额外的并行维度和通信建模：

- **MoE 并行**：`moe_tp_size`（张量并行）和 `moe_ep_size`（专家并行）
- **MoE 通信**：AllToAll 通信（专家分发/汇聚）
- **MoE 后端**：CUTLASS（Hopper）、WIDEEP（Blackwell）、deepep_moe（SGLang WideEP）
- **MoE 工作空间**：额外的 block-scale dispatch workspace 内存

### 10.3 投机解码支持

通过 `nextn` 参数建模 MTP（Multi-Token Prediction）：

```python
# nextn = 每步额外预测的 token 数
# decode_tokens_per_iteration = 1 + nextn
# 激活内存 × (nextn + 1) 用于验证
```

## 11. 关键代码路径索引

### 11.1 执行流程关键文件表

| 阶段 | 文件 | 关键函数/类 |
|------|------|-------------|
| CLI 入口 | `src/aiconfigurator/main.py` | `main()`, `_run_cli()` |
| CLI 参数解析 | `src/aiconfigurator/cli/main.py` | `configure_parser()`, `main()` |
| Task 构建 | `src/aiconfigurator/sdk/task_v2.py` | `Task`, `build_default_tasks()` |
| 配置搜索 | `src/aiconfigurator/sdk/sweep.py` | `sweep_agg()`, `sweep_disagg()` |
| 性能预测 | `src/aiconfigurator/sdk/predict.py` | `predict_agg_worker()`, `predict_disagg_worker()` |
| 后端工厂 | `aic-core/.../backends/factory.py` | `get_backend()` |
| 后端基类 | `aic-core/.../backends/base_backend.py` | `BaseBackend.run_agg()`, `run_static()` |
| 性能数据库 | `aic-core/.../perf_database.py` | `PerfDatabase` |
| Pareto 分析 | `src/aiconfigurator/sdk/picking.py` | `get_pareto_front()`, `pick_default()` |
| 结果输出 | `src/aiconfigurator/cli/report_and_save.py` | `log_final_summary()`, `save_results()` |

### 11.2 性能建模关键文件表

| 模块 | 文件 | 用途 |
|------|------|------|
| 算子基类 | `aic-core/.../operations/base.py` | `Operation.get_latency()` |
| 注意力算子 | `aic-core/.../operations/attention.py` | ContextAttention, GenerationAttention |
| GEMM 算子 | `aic-core/.../operations/gemm.py` | 矩阵乘法延迟 |
| MoE 算子 | `aic-core/.../operations/moe.py` | MoE 专家计算 |
| 通信算子 | `aic-core/.../operations/communication.py` | AllReduce, AllToAll |
| 内存建模 | `aic-core/.../memory.py` | 内存使用估算 |
| 引擎仿真 | `aic-core/.../engine.py` | 引擎步进仿真 |
| Rust 桥接 | `aic-core/.../rust_engine_step.py` | Rust FFI 调用 |

### 11.3 Generator 关键文件表

| 文件 | 用途 |
|------|------|
| `generator/api.py` | Generator API 入口 |
| `generator/rendering/engine.py` | 渲染引擎 + 上下文构建 |
| `generator/rendering/rule_engine.py` | .rule 文件求值器 |
| `generator/rendering/schemas.py` | 默认值应用 |
| `generator/config/deployment_config.yaml` | 输入 schema |
| `generator/config/backend_config_mapping.yaml` | 参数映射 |
| `generator/config/backend_templates/` | Jinja2 模板 |
| `generator/rule_plugin/*.rule` | 后端计算规则 |
| `generator/facts/hardware.yaml` | 硬件 profile 定义 |
| `generator/builders/k8s_builder.py` | K8s manifest 构建器 |

## 12. 扩展点与适配接口

### 12.1 新增系统 (System)

添加新 GPU 系统需要：

1. 创建 `aic-core/src/aiconfigurator_core/systems/<system>.yaml`：
   ```yaml
   data_dir: data/<system>
   gpu:
     mem_bw: <bytes/sec>
     mem_capacity: <bytes>
     bfloat16_tc_flops: <flops>
     # ...
   node:
     num_gpus_per_node: <n>
     intra_node_bw: <bytes/sec>
     # ...
   misc:
     nccl_mem: {1: 0, 2: <bytes>, ...}
     other_mem: <bytes>
   ```

2. 创建性能数据目录 `systems/data/<system>/`（或标记为 estimate-only）

3. 在 `generator/facts/hardware.yaml` 中添加硬件 profile

4. 更新支持矩阵

### 12.2 新增后端 (Backend)

添加新推理后端需要：

1. 创建 `aic-core/.../backends/<backend>_backend.py`：
   ```python
   class NewBackend(BaseBackend):
       def __init__(self):
           super().__init__()
           self.name = "new_backend"
       # 重写需要特化的方法
   ```

2. 在 `factory.py` 中注册

3. 创建 Generator 模板 `generator/config/backend_templates/<backend>/`

4. 创建 `.rule` 文件 `generator/rule_plugin/<backend>.rule`

5. 更新 `backend_config_mapping.yaml`

### 12.3 新增模型 (Model)

添加新 LLM 模型需要：

1. 创建 `aic-core/.../models/<model>.py`：
   ```python
   class NewModel(BaseModel):
       def __init__(self, model_path, model_config, backend_name):
           super().__init__(model_path, model_config, backend_name)
           # 定义算子列表
   ```

2. 在 `models/__init__.py` 中注册模型家族映射

3. 创建模型配置 `model_configs/<model>/config.json`

4. 如需 Collector 支持，添加 `collector/cases/models/<model>.yaml`

### 12.4 平台适配扩展点

适配非 NVIDIA 平台的关键扩展点：

| 层 | 扩展点 | 当前 NVIDIA 依赖 |
|---|---|---|
| System Spec | YAML 字段 | `sm_version`, `nccl_mem`, Tensor Core FLOPS 语义 |
| Perf Database | 数据目录 | NCCL 通信数据、CUDA kernel 数据 |
| SDK 估算 | 内存/通信模型 | NCCL 内存、CUDA 假设 |
| Generator | 硬件 profile/模板 | `CUDA_VISIBLE_DEVICES`、NVIDIA 后端模板 |

## 13. 附录

### 13.1 CLI 命令完整参数列表

```bash
aiconfigurator cli default \
  --model-path <path>           # 必选: HuggingFace 模型路径
  --system <name>               # 必选: 系统名称
  --backend <name>              # 后端: trtllm/vllm/sglang/auto
  --backend-version <ver>       # 后端版本
  --total-gpus <n>              # 总 GPU 数
  --isl <n>                     # 输入序列长度
  --osl <n>                     # 输出序列长度
  --ttft <ms>                   # TTFT SLA 约束
  --tpot <ms>                   # TPOT SLA 约束
  --top-n <n>                   # 输出前 N 个配置
  --target-concurrency <n>      # 目标并发数
  --target-request-rate <n>     # 目标请求率
  --serving-mode <mode>         # auto/all/agg/disagg/afd
  --gemm-quant-mode <mode>      # GEMM 量化模式
  --kvcache-quant-mode <mode>   # KV cache 量化模式
  --nextn <n|auto>              # 投机解码长度
  --attention-backend <name>    # 注意力后端
  --save-dir <path>             # 保存目录
  --debug                       # 调试模式
```

### 13.2 系统 YAML 字段规范

```yaml
# 必选字段
data_dir: <string>              # 性能数据相对路径
gpu:
  mem_bw: <int>                 # 显存带宽 (bytes/sec)
  mem_capacity: <int>           # 显存容量 (bytes)
  bfloat16_tc_flops: <int>      # BF16 Tensor Core 算力 (FLOPS)
  power: <int>                  # 功耗 (W)
node:
  num_gpus_per_node: <int>      # 每节点 GPU 数
  inter_node_bw: <int>          # 节点间带宽 (bytes/sec/GPU)
  intra_node_bw: <int>          # 节点内带宽 (bytes/sec/GPU)
misc:
  nccl_mem: {<tp>: <bytes>}    # NCCL 内存开销
  other_mem: <int>              # 运行时内存开销 (bytes)

# 可选字段
gpu:
  fp8_tc_flops: <int>           # FP8 算力
  fp4_tc_flops: <int>           # FP4 算力
  int8_tc_flops: <int>          # INT8 算力
  sm_version: <int>             # SM 架构版本
  mem_bw_empirical_scaling_factor: <float>  # 带宽经验修正系数
  mem_empirical_constant_latency: <float>   # 常量延迟修正 (sec)
node:
  pcie_bw: <int>                # PCIe 带宽 (bytes/sec)
  p2p_latency: <float>          # P2P 延迟 (sec)
misc:
  nccl_version: <string>        # NCCL 版本
```

### 13.3 性能数据 parquet schema

算子延迟数据的 parquet 文件包含以下列：

| 列名 | 类型 | 含义 |
|------|------|------|
| `batch` | int | 批次大小 |
| `s` | int | 序列长度 |
| `h` | int | 隐藏维度 |
| `num_heads` | int | 注意力头数 |
| `head_size` | int | 每头维度 |
| `latency_ms` | float | 延迟 (毫秒) |
| `energy_wms` | float | 能耗 (瓦·毫秒) |
| `source` | string | 数据来源 (silicon/hybrid/fpm) |

---

*本文档基于 aiconfigurator v0.12.0 代码库分析。*
