# AIConfigurator CLI 六种模式详解

本文档详细介绍 AIConfigurator CLI 的六种运行模式：`default`、`estimate`、`recommend`、`exp`、`generate`、`support`。
链接：https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/cli_user_guide.md
---

## 目录

1. [概述](#概述)
2. [default 模式 - 默认对比分析](#default-模式---默认对比分析)
3. [estimate 模式 - 单点性能估算](#estimate-模式---单点性能估算)
4. [recommend 模式 - GPU数量推荐](#recommend-模式---gpu数量推荐)
5. [exp 模式 - YAML实验配置](#exp-模式---yaml实验配置)
6. [generate 模式 - 朴素配置生成](#generate-模式---朴素配置生成)
7. [support 模式 - 支持矩阵检查](#support-模式---支持矩阵检查)
8. [Python API 接口](#python-api-接口)
9. [模式对比总结](#模式对比总结)

---

## 概述

AIConfigurator 提供六种 CLI 模式，覆盖从快速估算到完整部署配置生成的全流程：

| 模式 | 用途 | 典型场景 |
|------|------|----------|
| `default` | 聚合 vs 分离对比分析 | 给定GPU数量，找最优配置 |
| `estimate` | 单点性能估算 | 预测特定配置的TTFT/TPOT |
| `recommend` | GPU数量推荐 | 根据性能目标规划采购 |
| `exp` | YAML实验配置 | 批量运行多个实验场景 |
| `generate` | 朴素配置生成 | 快速生成基础部署配置 |
| `support` | 支持矩阵检查 | 验证模型/硬件兼容性 |

### 基本调用格式

```bash
aiconfigurator cli <mode> [OPTIONS]
```

---

## default 模式 - 默认对比分析

### 功能说明

`default` 模式是 AIConfigurator 的核心功能，用于在给定 GPU 数量下，对比聚合 (aggregated, agg) 和分离 (disaggregated, disagg) 两种部署模式的性能，找到最优配置。

### 核心逻辑

1. 构建 agg 和 disagg 两种部署模式的 Task
2. 扫描多种并行策略组合（TP/PP/DP/EP）
3. 运行性能估算，生成 Pareto 前沿
4. 按吞吐量排序，返回 Top-N 最优配置

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | HuggingFace 模型路径或本地路径 | 必需 |
| `--system` | GPU 系统类型 (如 `h200_sxm`) | 必需 |
| `--total-gpus` | GPU 总数 | 8 |
| `--backend` | 推理后端 (`trtllm`/`vllm`/`sglang`/`auto`) | `trtllm` |
| `--backend-version` | 后端版本 | 自动选择最新 |
| `--isl` | 输入序列长度 | 4000 |
| `--osl` | 输出序列长度 | 1000 |
| `--ttft` | TTFT SLA 目标 (ms) | 2000.0 |
| `--tpot` | TPOT SLA 目标 (ms) | 30.0 |
| `--top-n` | 返回前 N 个最优配置 | 5 |
| `--save-dir` | 结果保存目录 | 不保存 |

### 使用示例

```bash
# 基本使用：8卡 H200 运行 Qwen3-32B
aiconfigurator cli default \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm \
    --total-gpus 8

# 指定后端和SLA目标
aiconfigurator cli default \
    --model-path deepseek-ai/DeepSeek-V3 \
    --system h200_sxm \
    --total-gpus 32 \
    --backend auto \
    --ttft 600 --tpot 50 \
    --isl 4000 --osl 500 \
    --top-n 3 \
    --save-dir results

# 使用 FP8 量化模型
aiconfigurator cli default \
    --model-path Qwen/Qwen3-32B-FP8 \
    --system h200_sxm \
    --total-gpus 16 \
    --backend trtllm
```

### 输出说明

输出包含：
- **chosen_exp**: 最佳吞吐量的实验名称
- **best_configs**: 各实验的最优配置（DataFrame）
- **pareto_fronts**: Pareto 前沿数据
- **best_throughputs**: 各实验的最佳吞吐量 (tokens/s/gpu_cluster)

---

## estimate 模式 - 单点性能估算

### 功能说明

`estimate` 模式用于预测单个配置的 TTFT、TPOT 和功耗。适合快速评估特定硬件配置的性能表现。

### 估算模式

`estimate` 支持多种子模式：

| 子模式 | 说明 |
|--------|------|
| `agg` | 聚合模式 (IFB 调度) |
| `disagg` | 分离模式 (P/D 分离) |
| `afd` | Attention-FFN 分离 |
| `static` | 静态批处理 |
| `static_ctx` | 静态批处理（仅上下文） |
| `static_gen` | 静态批处理（仅生成） |

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径 | 必需 |
| `--system` | GPU 系统类型 | 必需 |
| `--estimate-mode` | 估算子模式 | `agg` |
| `--backend` | 推理后端 | `trtllm` |
| `--batch-size` | 批处理大小 | 128 |
| `--tp-size` | 张量并行度 | 1 |
| `--pp-size` | 流水线并行度 | 1 |
| `--isl` | 输入序列长度 | 1024 |
| `--osl` | 输出序列长度 | 1024 |
| `--database-mode` | 数据库模式 | `SILICON` |

### 分离模式专用参数

| 参数 | 说明 |
|------|------|
| `--prefill-tp-size` | 预填充 TP 大小 |
| `--prefill-pp-size` | 预填充 PP 大小 |
| `--prefill-batch-size` | 预填充批大小 |
| `--prefill-num-workers` | 预填充 worker 数 |
| `--decode-tp-size` | 解码 TP 大小 |
| `--decode-pp-size` | 解码 PP 大小 |
| `--decode-batch-size` | 解码批大小 |
| `--decode-num-workers` | 解码 worker 数 |

### 使用示例

```bash
# 聚合模式估算
aiconfigurator cli estimate \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm \
    --estimate-mode agg \
    --tp-size 4 \
    --batch-size 64 \
    --isl 2048 --osl 512

# 分离模式估算
aiconfigurator cli estimate \
    --model-path deepseek-ai/DeepSeek-V3 \
    --system h200_sxm \
    --estimate-mode disagg \
    --prefill-tp-size 8 --prefill-batch-size 1 --prefill-num-workers 4 \
    --decode-tp-size 8 --decode-batch-size 256 --decode-num-workers 8

# 静态批处理模式
aiconfigurator cli estimate \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm \
    --estimate-mode static \
    --tp-size 4 --batch-size 32 \
    --isl 4000 --osl 1000
```

### 输出说明

返回 `EstimateResult` 对象，包含：
- **ttft**: 首个 token 延迟 (ms)
- **tpot**: 每 token 延迟 (ms)
- **request_latency**: 端到端请求延迟 (ms)
- **power**: 功耗估算 (W)
- **memory_usage**: 显存使用情况

---

## recommend 模式 - GPU数量推荐

### 功能说明

`recommend` 模式是采购规划工具，根据性能目标反推所需的最少 GPU 数量。与 `default` 模式相反：
- `default`: 给定 GPU 数量，找最优配置
- `recommend`: 给定性能目标，找最少 GPU 数量

### 核心逻辑

1. 从单节点 GPU 数开始搜索
2. 逐步倍增 GPU 预算（2x, 4x, 8x...）
3. 在每个预算下运行 `default` 模式分析
4. 找到满足 SLA 目标的最小 GPU 配置

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径 | 必需 |
| `--system` | GPU 系统类型 | 必需 |
| `--target-request-rate` | 目标请求速率 (req/s) | 与 concurrency 二选一 |
| `--target-concurrency` | 目标并发数 | 与 request-rate 二选一 |
| `--ttft` | TTFT SLA 目标 (ms) | 2000.0 |
| `--tpot` | TPOT SLA 目标 (ms) | 30.0 |
| `--backend` | 推理后端 | `trtllm` |
| `--strict-sla` | 严格 SLA 过滤 | false |

### 使用示例

```bash
# 根据请求速率推荐
aiconfigurator cli recommend \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm \
    --target-request-rate 50.0 \
    --ttft 2000 --tpot 30

# 根据并发数推荐
aiconfigurator cli recommend \
    --model-path deepseek-ai/DeepSeek-V3 \
    --system h200_sxm \
    --target-concurrency 100 \
    --ttft 1000 --tpot 40

# 使用 auto 后端
aiconfigurator cli recommend \
    --model-path Qwen/Qwen3-32B-FP8 \
    --system b200_sxm \
    --target-request-rate 100.0 \
    --backend auto \
    --strict-sla
```

### 输出说明

输出的 `best_configs` 包含额外列：
- **total_gpus_needed**: 所需 GPU 总数
- **replicas_needed**: 所需副本数

结果按 GPU 数量从少到多排序。

---

## exp 模式 - YAML实验配置

### 功能说明

`exp` 模式允许通过 YAML 文件定义一个或多个实验场景，支持批量运行和对比分析。适合复杂场景的系统性评估。

### YAML 配置格式

使用 Flat V2 schema，每个 key 直接映射到 Task 字段：

```yaml
# 可选：指定要运行的实验列表（按顺序执行）
exps:
- experiment_1
- experiment_2

# 实验定义
experiment_1:
  serving_mode: agg           # agg 或 disagg
  model_path: Qwen/Qwen3-32B
  system_name: h200_sxm
  total_gpus: 8
  isl: 4000
  osl: 1000
  ttft: 1000.0
  tpot: 40.0

experiment_2:
  serving_mode: disagg
  total_gpus: 32
  prefill_model_path: Qwen/Qwen3-32B
  prefill_system_name: h200_sxm
  decode_model_path: Qwen/Qwen3-32B
  decode_system_name: h200_sxm
```

### 聚合模式配置项

```yaml
agg_example:
  serving_mode: agg
  model_path: deepseek-ai/DeepSeek-V3
  system_name: h200_sxm
  backend_name: trtllm
  backend_version: 1.3.0rc10
  total_gpus: 8
  isl: 4000
  osl: 1000
  ttft: 1000.0
  tpot: 40.0

  # 可选：量化覆盖
  gemm_quant_mode: fp8_block
  moe_quant_mode: fp8_block
  kvcache_quant_mode: bfloat16

  # 可选：搜索空间
  agg_tp_candidates: [1, 2, 4, 8]
  agg_pp_candidates: [1]
  agg_dp_candidates: [1, 2, 4, 8]
```

### 分离模式配置项

```yaml
disagg_example:
  serving_mode: disagg
  total_gpus: 32
  isl: 4000
  osl: 1000
  ttft: 1000.0
  tpot: 40.0

  # 预填充 worker
  prefill_model_path: deepseek-ai/DeepSeek-V3
  prefill_system_name: h200_sxm
  prefill_backend_name: trtllm
  prefill_tp_candidates: [1, 2, 4, 8]

  # 解码 worker
  decode_model_path: deepseek-ai/DeepSeek-V3
  decode_system_name: h200_sxm
  decode_backend_name: trtllm
  decode_tp_candidates: [1, 2, 4, 8]

  # 编排参数
  num_gpu_per_replica: [8, 16, 24, 32]
  max_prefill_workers: 32
  max_decode_workers: 32
```

### 使用示例

```bash
# 运行单个 YAML 文件中的所有实验
aiconfigurator cli exp --yaml-path experiments.yaml

# 指定保存目录
aiconfigurator cli exp \
    --yaml-path src/aiconfigurator/cli/exps/qwen3_32b_disagg.yaml \
    --save-dir results

# 使用预置实验配置
aiconfigurator cli exp \
    --yaml-path src/aiconfigurator/cli/exps/deepseek_disagg_sglang.yaml
```

### 预置实验配置

项目提供多个预置实验配置，位于 `src/aiconfigurator/cli/exps/`：

| 文件 | 说明 |
|------|------|
| `qwen3_32b_disagg.yaml` | Qwen3-32B 分离模式对比 |
| `h200_qwen3_32b_multi_backends.yaml` | 多后端对比 |
| `deepseek_disagg_sglang.yaml` | DeepSeek 分离模式 |
| `deepseek_wideep_trtllm.yaml` | DeepSeek WideEP |
| `hetero_disagg.yaml` | 异构分离模式 |
| `database_mode_comparison.yaml` | 数据库模式对比 |

---

## generate 模式 - 朴素配置生成

### 功能说明

`generate` 模式生成基础的聚合部署配置，不进行参数扫描优化。适合快速获取一个可用的部署配置作为起点。

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径 | 必需 |
| `--total-gpus` | GPU 总数 | 必需 |
| `--system` | GPU 系统类型 | 必需 |
| `--backend` | 推理后端 | `trtllm` |
| `--save-dir` | 输出目录 | 当前目录 |

### 使用示例

```bash
# 基本使用
aiconfigurator cli generate \
    --model-path Qwen/Qwen3-32B \
    --total-gpus 8 \
    --system h200_sxm

# 指定后端和输出目录
aiconfigurator cli generate \
    --model-path deepseek-ai/DeepSeek-V3 \
    --total-gpus 16 \
    --system h200_sxm \
    --backend vllm \
    --save-dir ./configs

# 生成 FP8 模型配置
aiconfigurator cli generate \
    --model-path Qwen/Qwen3-32B-FP8 \
    --total-gpus 4 \
    --system b200_sxm \
    --backend trtllm
```

### 输出说明

生成的配置包含：
- 基础并行策略（TP/PP）
- 批处理大小建议
- KV Cache 配置
- 引擎构建参数

---

## support 模式 - 支持矩阵检查

### 功能说明

`support` 模式检查特定模型和硬件组合是否被 AIConfigurator 支持。是轻量级的预检查工具，无需运行完整的性能估算。

### 检查逻辑

支持矩阵基于以下维度进行多数投票：
- 模型架构
- GPU 系统类型
- 后端名称
- 后端版本

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径 | 必需 |
| `--system` | GPU 系统类型或 `all` | 必需 |
| `--backend` | 后端名称或 `all` | `trtllm` |
| `--backend-version` | 后端版本过滤 | 最新版本 |

### 使用示例

```bash
# 检查单个系统
aiconfigurator cli support \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm

# 检查所有系统
aiconfigurator cli support \
    --model-path Qwen/Qwen3-32B \
    --system all \
    --backend all

# 指定后端版本
aiconfigurator cli support \
    --model-path deepseek-ai/DeepSeek-V3 \
    --system h200_sxm \
    --backend trtllm \
    --backend-version 1.3.0rc10
```

### 输出说明

返回元组 `(agg_supported, disagg_supported)`：
- `agg_supported`: 聚合模式是否支持
- `disagg_supported`: 分离模式是否支持

---

## Python API 接口

所有模式都提供 Python API 接口，方便程序化调用：

```python
from aiconfigurator.cli import (
    cli_default,
    cli_estimate,
    cli_generate,
    cli_recommend,
    cli_support,
    cli_exp,
)
```

### API 函数签名

#### cli_support

```python
def cli_support(
    model_path: str,
    system: str,
    *,
    backend: str = "trtllm",
    backend_version: str | None = None,
) -> tuple[bool, bool]
```

#### cli_recommend

```python
def cli_recommend(
    model_path: str,
    system: str,
    *,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    ttft: float = 2000.0,
    tpot: float = 30.0,
    backend: str = "trtllm",
    ...
) -> CLIResult
```

#### cli_estimate

```python
def cli_estimate(
    model_path: str,
    system_name: str,
    *,
    mode: str = "agg",
    backend_name: str = "trtllm",
    isl: int = 1024,
    osl: int = 1024,
    batch_size: int = 128,
    tp_size: int = 1,
    ...
) -> EstimateResult
```

### 使用示例

```python
# 检查支持
agg_ok, disagg_ok = cli_support(
    model_path="Qwen/Qwen3-32B",
    system="h200_sxm"
)

# 推荐 GPU 数量
result = cli_recommend(
    model_path="Qwen/Qwen3-32B",
    system="h200_sxm",
    target_request_rate=50.0,
    ttft=2000,
    tpot=30,
)
print(f"Best config: {result.best_configs}")

# 单点估算
estimate_result = cli_estimate(
    model_path="Qwen/Qwen3-32B",
    system_name="h200_sxm",
    mode="agg",
    tp_size=4,
    batch_size=64,
    isl=2048,
    osl=512,
)
print(f"TTFT: {estimate_result.ttft}ms, TPOT: {estimate_result.tpot}ms")
```

---

## 模式对比总结

| 特性 | default | estimate | recommend | exp | generate | support |
|------|---------|----------|-----------|-----|----------|---------|
| 输入 | 模型+GPU数 | 模型+配置 | 模型+目标 | YAML | 模型+GPU数 | 模型+系统 |
| 输出 | Top-N配置 | 性能指标 | 最少GPU数 | 实验结果 | 基础配置 | 支持状态 |
| 优化 | 参数扫描 | 无 | GPU搜索 | 依配置 | 无 | 无 |
| 场景 | 部署规划 | 性能评估 | 采购规划 | 批量实验 | 快速生成 | 兼容性检查 |

### 选择建议

1. **初次评估模型** → `support` 先检查兼容性
2. **规划部署方案** → `default` 对比 agg/disagg
3. **评估特定配置** → `estimate` 单点估算
4. **采购规划** → `recommend` 找最少 GPU
5. **批量对比** → `exp` 定义多场景实验
6. **快速起步** → `generate` 生成基础配置

---

*文档生成时间: 2026-08-04*
*项目: aiconfigurator*
