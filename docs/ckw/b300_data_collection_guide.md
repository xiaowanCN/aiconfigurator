# B300 GPU 性能数据采集完整指南

本文档详细介绍如何在 B300 SXM GPU 机器上使用 aiconfigurator collector 模块采集性能数据。

---

## 目录

0. [Docker 镜像构建方案](#零docker-镜像构建方案)
1. [环境准备](#一环境准备)
2. [数据采集步骤](#二数据采集步骤)
3. [断点续传](#三断点续传)
4. [输出文件说明](#四输出文件说明)
5. [数据验证与入库](#五数据验证与入库)
6. [完整命令参考](#六完整命令参考)
7. [B300 特性说明](#七b300-特性说明)
8. [支持的模型架构](#八支持的模型架构)
9. [注意事项](#九注意事项)

---

## 零、Docker 镜像构建方案

### 0.1 构建概述

`docker/Dockerfile.collector` 定义了三个独立的采集镜像，分别对应三个推理框架后端：

| 镜像名称 | 基础镜像 | 用途 |
|----------|----------|------|
| `collector-sglang` | `lmsysorg/sglang:v0.5.14` | SGLang 后端算子采集 |
| `collector-trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10` | TRT-LLM 后端算子采集 |
| `collector-vllm` | `vllm/vllm-openai:v0.24.0` | vLLM 后端算子采集 |

### 0.2 构建镜像

在项目根目录执行以下命令构建镜像：

```bash
# 进入项目根目录
cd /path/to/aiconfigurator

# 构建 SGLang collector 镜像
docker build -f docker/Dockerfile.collector --target collector-sglang -t collector-sglang .

# 构建 TRT-LLM collector 镜像
docker build -f docker/Dockerfile.collector --target collector-trtllm -t collector-trtllm .

# 构建 vLLM collector 镜像
docker build -f docker/Dockerfile.collector --target collector-vllm -t collector-vllm:v0804.1 .
```

**一键构建所有镜像：**

```bash
# 按顺序构建三个镜像
docker build -f docker/Dockerfile.collector --target collector-sglang -t collector-sglang . && \
docker build -f docker/Dockerfile.collector --target collector-trtllm -t collector-trtllm . && \
docker build -f docker/Dockerfile.collector --target collector-vllm -t collector-vllm:v0804.1 .
```

### 0.3 创建并启动容器

```bash
# vLLM 采集容器（推荐）
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --name collector-vllm-b300 \
  -v $(pwd)/output:/output \
  collector-vllm:v0804.1

# TRT-LLM 采集容器
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --name collector-trtllm-b300 \
  -v $(pwd)/output:/output \
  collector-trtllm

# SGLang 采集容器
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --name collector-sglang-b300 \
  -v $(pwd)/output:/output \
  collector-sglang
```

### 0.4 验证容器

```bash
# 查看运行中的容器
docker ps | grep collector

# 进入容器验证环境
docker exec -it collector-vllm-b300 bash

# 容器内验证 collector 脚本
python3 collect.py --help
```

### 0.5 容器内 GPU 配置

```bash
# 进入容器
docker exec -it collector-vllm-b300 bash

# 启用持久模式
sudo nvidia-smi -pm 1

# 锁定GPU频率到最大值
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
sudo nvidia-smi -ac $mem_freq,$sm_freq

# 验证频率锁定
nvidia-smi -q -i 0 | grep -A 4 "Max Clocks"
```

### 0.6 使用预构建镜像运行采集

```bash
# 方式1: 进入容器手动执行
docker exec -it collector-vllm-b300 bash
cd /workspace/collector
python3 collect.py --backend vllm --gpu b300_sxm --smoke

# 方式2: 直接在容器外执行命令
docker exec collector-vllm-b300 python3 /workspace/collector/collect.py \
  --backend vllm --gpu b300_sxm --smoke

# 方式3: 全量采集（后台运行）
docker exec -d collector-vllm-b300 python3 /workspace/collector/collect.py \
  --backend vllm --gpu b300_sxm --resume --measure_power

# 方式4: 采集输出到挂载目录
docker exec collector-vllm-b300 python3 /workspace/collector/collect.py \
  --backend vllm --gpu b300_sxm --resume
# 输出文件在容器内 /workspace/collector/ 目录
# 可通过挂载的 volume 在宿主机访问
```

### 0.7 清理容器和镜像

```bash
# 停止并删除容器
docker stop collector-sglang-b300 collector-trtllm-b300 collector-vllm-b300
docker rm collector-sglang-b300 collector-trtllm-b300 collector-vllm-b300

# 删除镜像（可选）
docker rmi collector-sglang collector-trtllm collector-vllm
```

---

## 一、环境准备

### 1.1 启动框架容器（传统方式）

如果不使用预构建的 collector 镜像，也可以直接使用框架官方镜像并挂载源代码：

#### vLLM 后端（推荐）

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host \
  --name vllm -v /path/to/aiconfigurator:/workspace \
  vllm/vllm-openai:v0.24.0
```

#### TensorRT-LLM 后端

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host \
  --name trtllm -v /path/to/aiconfigurator:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10
```

#### SGLang 后端

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host \
  --name sglang -v /path/to/aiconfigurator:/workspace \
  lmsysorg/sglang:v0.5.14
```

### 1.2 GPU 配置（容器内执行）

```bash
# 进入容器
docker exec -it vllm bash

# 启用持久模式
sudo nvidia-smi -pm 1

# 锁定GPU频率到最大值
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
sudo nvidia-smi -ac $mem_freq,$sm_freq

# 验证频率锁定
nvidia-smi -q -i 0 | grep -A 4 "Max Clocks"
```

### 1.3 环境要求

- 节点独占，无其他任务干扰
- 冷却系统正常
- 本地存储或 `/tmp/` 目录（避免 NFS 文件锁定问题）

---

## 二、数据采集步骤

### 2.1 查看采集计划（预览）

```bash
cd /workspace/collector

# 查看 B300 全量采集计划
python3 collect.py --backend vllm --gpu b300_sxm --plan-only

# 查看特定模型的采集计划
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-architecture DeepseekV4ForCausalLM --plan-only
```

### 2.2 冒烟测试（验证环境）

```bash
# 快速验证环境是否正常（每op随机采样4个案例）
python3 collect.py --backend vllm --gpu b300_sxm --smoke
```

### 2.3 采集通信数据

```bash
# 设置NCCL测试二进制路径（如有）
export PATH=$PATH:${NCCL_TEST_BIN_PATH}/

# vLLM 后端（跳过NCCL测试）
network/collect_comm.sh --all_reduce_backend vllm --skip-nccl

# TRT-LLM 后端
network/collect_comm.sh

# SGLang 后端
network/collect_comm.sh --all_reduce_backend sglang

# 启用功率监控
network/collect_comm.sh --all_reduce_backend vllm --skip-nccl --measure_power --power_test_duration 2.0
```

通信脚本会自动检测 GPU 数量并测试 2/4/8 GPU 组合，采集：
- NCCL all_gather, alltoall, reduce_scatter, all_reduce (half/int8)
- 自定义 AllReduce（根据所选后端）

### 2.4 全量算子采集

```bash
# B300 全量采集（约30 GPU小时）
python3 collect.py --backend vllm --gpu b300_sxm

# 启用功率监控
python3 collect.py --backend vllm --gpu b300_sxm --measure_power

# 自定义功率测量持续时间
python3 collect.py --backend vllm --gpu b300_sxm --measure_power --power_test_duration_sec 2.0
```

### 2.5 针对特定模型采集

```bash
# DeepSeek-V4
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-path deepseek-ai/DeepSeek-V4

# 或指定架构
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-architecture DeepseekV4ForCausalLM

# Llama 系列
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-architecture LlamaForCausalLM

# Qwen 系列
python3 collect.py --backend vllm --gpu b300_sxm \
  --model-architecture Qwen3MoeForCausalLM
```

### 2.6 选择性采集（修复/补充）

```bash
# 只采集特定操作
python3 collect.py --backend vllm --gpu b300_sxm --ops gemm moe

# 使用过滤器（子字符串匹配，可重复，OR语义）
python3 collect.py --backend vllm --gpu b300_sxm --ops gemm --case-filter "tp=8"

# 限制每op最大案例数
python3 collect.py --backend vllm --gpu b300_sxm --limit 100
```

---

## 三、断点续传

采集过程中如果中断，可以使用检查点恢复：

```bash
# 恢复中断的运行（自动从 .collector_checkpoint/ 读取检查点）
python3 collect.py --backend vllm --gpu b300_sxm --resume

# 重试之前失败的任务
python3 collect.py --backend vllm --gpu b300_sxm --resume --resume-retry-failed

# 自定义检查点目录
python3 collect.py --backend vllm --gpu b300_sxm --resume --checkpoint-dir /path/to/checkpoints
```

---

## 四、输出文件说明

采集完成后，输出文件在当前目录（自动转换为 parquet 格式）：

### 算子性能数据

| 文件名 | 说明 |
|--------|------|
| `gemm_perf.parquet` | GEMM 矩阵乘法 |
| `moe_perf.parquet` | MoE 混合专家 |
| `context_attention_perf.parquet` | 标准注意力预填充 |
| `generation_attention_perf.parquet` | 标准注意力解码 |
| `context_mla_perf.parquet` | MLA 预填充 |
| `generation_mla_perf.parquet` | MLA 解码 |
| `mla_bmm_perf.parquet` | MLA BMM |
| `computescale_perf.parquet` | FP8 量化开销 |
| `dsa_context_module_perf.parquet` | DSA 模块预填充 |
| `dsa_generation_module_perf.parquet` | DSA 模块解码 |
| `encoder_attention_perf.parquet` | 编码器注意力 |
| `gdn_perf.parquet` | GDN 操作 |
| `mhc_module_perf.parquet` | MHC 模块 |

### 通信性能数据

| 文件名 | 说明 |
|--------|------|
| `all_reduce_perf.parquet` | AllReduce 通信 |
| `all_gather_perf.parquet` | AllGather 通信 |
| `alltoall_perf.parquet` | AllToAll 通信 |
| `reduce_scatter_perf.parquet` | ReduceScatter 通信 |

---

## 五、数据验证与入库

```bash
# 采集完成后，将数据移动到系统目录
cp *.parquet /workspace/aic-core/src/aiconfigurator_core/systems/data/b300_sxm/

# 如果需要保留CSV格式，使用 --keep-csv 参数
python3 collect.py --backend vllm --gpu b300_sxm --keep-csv
```

---

## 六、完整命令参考

### collect.py 参数列表

```
python3 collect.py [OPTIONS]

主要参数:
  --backend {trtllm,sglang,vllm}    后端选择（默认: trtllm）
  --gpu GPU_TYPE                     GPU类型（如 b300_sxm）
  --sm SM_VERSION                    SM版本（如 103）
  --ops [OPS ...]                    指定操作列表
  --smoke                            冒烟测试模式
  --plan-only                        仅查看计划，不执行
  --resume                           断点续传
  --resume-retry-failed              恢复时重试失败任务
  --checkpoint-dir DIR               检查点目录
  --measure_power                    启用功率监控
  --power_test_duration_sec SEC      功率测量最小持续时间
  --model-path MODEL_PATH            模型路径
  --model-architecture ARCH          模型架构
  --model-cases YAML_PATH            自定义模型案例YAML
  --model-cases-full                 全量模式
  --case-filter FILTER               案例过滤器（可重复）
  --limit LIMIT                      每op最大案例数
  --shuffle                          随机打乱案例
  --debug                            启用调试日志
  --profile                          性能分析模式
  --keep-csv                         保留CSV不转parquet
```

### 采集操作（Ops）完整列表

#### vLLM 后端支持的操作

| 操作名 | 说明 |
|--------|------|
| `gemm` | GEMM 矩阵乘法 |
| `compute_scale` | FP8 量化开销 |
| `mla_context` | MLA 预填充 |
| `mla_generation` | MLA 解码 |
| `mla_bmm_gen_pre` | MLA BMM gen pre |
| `mla_bmm_gen_post` | MLA BMM gen post |
| `moe` | 混合专家 |
| `attention_context` | 标准注意力预填充 |
| `attention_generation` | 标准注意力解码 |
| `encoder_attention` | 编码器注意力 |
| `dsa_context_module` | DSA 模块预填充 |
| `dsa_generation_module` | DSA 模块解码 |
| `dsv4_csa_context_module` | DeepSeek-V4 CSA 预填充 |
| `dsv4_hca_context_module` | DeepSeek-V4 HCA 预填充 |
| `dsv4_csa_generation_module` | DeepSeek-V4 CSA 解码 |
| `dsv4_hca_generation_module` | DeepSeek-V4 HCA 解码 |
| `gdn` | GDN 操作 |
| `mhc_module` | MHC 模块 |

---

## 七、B300 特性说明

### 硬件规格

| 参数 | 值 |
|------|-----|
| SM Version | 103 |
| 显存带宽 | 7.75 TB/s |
| 显存容量 | ~275 GB (288,400,343,040 bytes) |
| BF16 Tensor Core 算力 | 2,250 TFLOPS |
| FP8 Tensor Core 算力 | 4.5 PFLOPS |
| FP4 Tensor Core 算力 | 14 PFLOPS |
| 功耗 | 1,100W |
| 每节点 GPU 数 | 8 |
| 节点间带宽 | 100 GB/s (CX8 XDR 800Gb/s) |
| 节点内带宽 | 900 GB/s (NVLink 双向 1.8TB/s) |
| NCCL 版本 | 2.27 |

### 支持的数据类型

| 数据类型 | 最低 SM | B300 支持 |
|----------|---------|-----------|
| float16 | 0 | Yes |
| bfloat16 | 80 | Yes |
| fp8 | 89 | Yes |
| fp8_block | 89 | Yes |
| nvfp4 | 100 | Yes |
| w4a16_mxfp4 | 89 | Yes |
| w4a8_mxfp4_mxfp8 | 100 | Yes |
| dsa_context_module | 90 | Yes |
| compute_scale | 89 | Yes |

### 系统定义文件

B300 的系统定义位于：
```
aic-core/src/aiconfigurator_core/systems/b300_sxm.yaml
```

---

## 八、支持的模型架构

以下是已定义案例文件的模型架构（共23个）：

### DeepSeek 系列
- `DeepseekV3ForCausalLM`
- `DeepseekV32ForCausalLM`
- `DeepseekV4ForCausalLM`

### Llama 系列
- `LlamaForCausalLM`
- `Llama4ForCausalLM`

### Qwen 系列
- `Qwen3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `Qwen3VLForCausalLM`
- `Qwen3VLMoeForCausalLM`
- `Qwen35ForCausalLM`
- `Qwen35MoeForCausalLM`
- `Qwen15MoeForCausalLM`

### 其他模型
- `Gemma4ForCausalLM`
- `GlmMoeDsaForCausalLM`
- `GptOssForCausalLM`
- `MiMoForCausalLM`
- `MiMoV2FlashForCausalLM`
- `MiniMaxM2ForCausalLM`
- `MiniMaxM3ForCausalLM`
- `MixtralForCausalLM`
- `NemotronHForCausalLM`
- `DeciLMForCausalLM`
- `KimiK25ForCausalLM`

---

## 九、注意事项

### 环境要求

1. **节点独占**: 采集时确保节点无其他任务干扰，避免数据不准确
2. **NFS 问题**: 避免在 NFS 上采集，`fcntl.flock()` 不可靠，可能导致 worker 死锁。使用本地存储或 `/tmp/`
3. **框架版本**: 必须使用 `framework_manifest.yaml` 中声明的精确版本
4. **Git LFS**: 确保源代码中的 LFS 文件已下载（`git lfs pull`）
5. **冷却时间**: 长时间采集后让 GPU 冷却，避免热节流影响数据

### 采集建议

1. **先冒烟测试**: 全量采集前先用 `--smoke` 验证环境
2. **使用断点续传**: 长时间采集建议使用 `--resume` 支持中断恢复
3. **分批采集**: 可以用 `--ops` 分批采集不同操作，最后合并
4. **功率监控**: 建议启用 `--measure_power` 记录功耗数据

### 常见问题

1. **Worker 死锁**: 检查是否在 NFS 上运行，改用本地目录
2. **框架版本不匹配**: 检查容器内框架版本是否与 `framework_manifest.yaml` 一致
3. **GPU 频率波动**: 确保已锁定 GPU 频率到最大值
4. **显存不足**: 某些大模型案例可能需要调整 batch size

---

## 附录: 快速开始脚本

```bash
#!/bin/bash
# B300 数据采集快速开始脚本

set -e

# 配置
BACKEND="vllm"
GPU="b300_sxm"
CONTAINER_NAME="vllm-collector"
IMAGE="vllm/vllm-openai:v0.24.0"

# 1. 启动容器
echo "启动容器..."
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host \
  --name $CONTAINER_NAME \
  -v $(pwd):/workspace \
  $IMAGE

# 2. 等待容器启动
sleep 5

# 3. 配置 GPU
echo "配置 GPU..."
docker exec $CONTAINER_NAME bash -c "
  sudo nvidia-smi -pm 1
  sm_freq=\$(nvidia-smi -q -i 0 | grep -A 4 'Max Clocks' | grep 'SM ' | grep -o '[0-9]\+ MHz' | grep -o '[0-9]\+')
  mem_freq=\$(nvidia-smi -q -i 0 | grep -A 4 'Max Clocks' | grep 'Memory ' | grep -o '[0-9]\+ MHz' | grep -o '[0-9]\+')
  sudo nvidia-smi -ac \$mem_freq,\$sm_freq
"

# 4. 冒烟测试
echo "运行冒烟测试..."
docker exec $CONTAINER_NAME bash -c "
  cd /workspace/collector
  python3 collect.py --backend $BACKEND --gpu $GPU --smoke
"

# 5. 采集通信数据
echo "采集通信数据..."
docker exec $CONTAINER_NAME bash -c "
  cd /workspace/collector
  network/collect_comm.sh --all_reduce_backend $BACKEND --skip-nccl
"

# 6. 全量采集
echo "开始全量采集..."
docker exec $CONTAINER_NAME bash -c "
  cd /workspace/collector
  python3 collect.py --backend $BACKEND --gpu $GPU --measure_power --resume
"

echo "采集完成！"
```

---

*文档生成时间: 2026-08-04*
*项目: aiconfigurator*
*目标GPU: B300 SXM (SM 103)*
