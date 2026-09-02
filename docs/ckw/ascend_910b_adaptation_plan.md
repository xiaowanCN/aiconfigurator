# 昇腾 910B 系列适配改造方案（基于 aiconfigurator 现状）

> 目标：让 `aiconfigurator` 从“以 NVIDIA 为主”演进为“可支持昇腾 910B 系列”的异构平台估算与配置生成框架。

## 1. 整体判断

适配昇腾 910B 的本质，不是简单新增一个 `--system ascend_910b`。

因为当前框架在以下层面都默认依赖 NVIDIA 体系：

- 系统规格（sm_version、nccl_*、Tensor Core FLOPS 语义）
- 性能数据库（nccl、CUDA 生态 backend 数据）
- SDK 估算模型（NCCL 内存、CUDA 假设）
- Generator（CUDA_VISIBLE_DEVICES、trtllm/vllm/sglang 模板）

因此建议采用“平台抽象增强 + 昇腾增量补齐”的策略，而不是硬塞。

## 2. 推荐改造目标

1. 新增一个独立的平台类型，把框架从 NVIDIA/CUDA/NCCL 默认升级为多平台抽象。
2. 扩展系统规格，新增 ascend_910b.yaml 并扩展字段定义。
3. 替换/扩展通信模型，把 nccl 扩展为 comm_backend（nccl / hccl / oneccl 等）。
4. 适配昇腾后端推理栈，新增 backend namespace（例如 mindie / ascend-mindie / ascend）。

## 3. 详细改造清单

### A. System Spec 层改造
1. 新增 ascend_910b.yaml
2. 重定义/兼容“SM version”语义，避免直接套用 NVIDIA SM

### B. Perf Database 层改造
1. 新增昇腾系统目录
2. 通信表从 NCCL 扩展为通信库抽象
3. 支持 XPU / Ascend kernel source 标记

### C. SDK 估算层改造
1. 内存模型抽象
2. 通信延迟/带宽估算抽象
3. 算力表达抽象
4. 平台判断逻辑扩展

### D. Generator 层改造
1. 新增 Ascend 硬件 profile
2. 模板支持昇腾设备选择逻辑
3. 后端配置映射新增 MindIE（或对应栈）
4. generator 不应再假设 NVIDIA-only

### E. Collector 层改造
1. 新增 Ascend 收集路径
2. 设备探测抽象
3. 失败分类与 capability 语义扩展

### F. 矩阵 / 支持能力层改造
1. support matrix 扩展平台维度
2. 新增 Ascend 系统 CSV

## 4. 推荐实施顺序

1. Phase 1：平台抽象层改造（最关键）
2. Phase 2：Ascend 最小闭环
3. Phase 3：实测数据补齐
4. Phase 4：产品化

## 5. 我对改造难度的判断

- 简单部分：新增系统 YAML、新增硬件 profile、新增一个 estimate-only 系统入口
- 中等部分：generator 设备变量抽象、后端参数映射扩展、system spec 字段 vendor-neutral 化
- 复杂部分：SDK 通信模型从 NCCL 抽象到 HCCL、backend 从 CUDA 生态扩展到 MindIE、collector 从 CUDA 探测扩展到 Ascend 探测、真实 silicon data 补齐与校准

## 6. 落地建议

1. 先做“平台抽象增强”，再做“昇腾支持”
2. 第一阶段先支持“estimate-only 910B”
3. MindIE 适配优先级最高
4. 同步补平台校验和测试

## 7. 最终结论

如果要支持昇腾 910B，框架的改动方案可分为四层：

1. 系统语义层：vendor/arch/comm/算力字段重构
2. 数据层：新增 ascend 910B 的 perf database 与 HCCL 数据
3. 估算层：SDK 内存/通信/并行估算 vendor-neutral 化
4. 生成层：generator 支持 Ascend backend、模板与 profile

一句话总结：适配昇腾 910B 的关键，不是“给现有框架补一个型号”，而是“让 aiconfigurator 从 NVIDIA 默认框架升级为可支持多厂商加速卡的平台估算框架”。