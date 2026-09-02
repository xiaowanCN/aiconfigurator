# 昇腾 910B 适配：详细任务拆解（可执行版）

> 说明：本文档在不修改已有分析文档的前提下，基于 `docs/ckw/ascend_910b_adaptation_plan.md` 的分层方案，进一步拆解为可执行工程任务。

---

## 0. 目标定义

### 业务目标
- 让 `aiconfigurator` 具备昇腾 910B 系列的基础评估能力与配置生成能力。
- 首期目标不是“完全对齐 NVIDIA 全量能力”，而是先做到 **可评估、可生成、可验证、可扩展**。

### 成功标准（Phase 1）
- `--system ascend_910b` 能跑通默认/推荐/估算主流程。
- 系统规格、硬件 profile、generator 模板、SDK 估算链路能识别 Ascend 平台。
- 可产出至少一套 estimate-only 的部署配置生成结果。
- support matrix 能展示 ascend_910b 的支持状态（哪怕初期是受限支持）。

---

## 1. 工作原则

1. **先平台抽象，再产品补齐**
   - 避免在现有 NVIDIA 逻辑中硬编码大量 ascend 条件。
2. **先 estimate-only，再 silicon data**
   - 首期允许基于系统规格和估算公式上线，不依赖完整实测数据。
3. **先最小闭环，再扩展**
   - 优先打通一条端到端链路，再逐步覆盖更多 backend/model/mode。
4. **规则优先、文档同步**
   - 涉及 generator/collector 改动，需遵守 `.claude/rules/` 约束。

---

## 2. 总体分期建议

### Phase 1：平台抽象层 + 最小闭环
- 目标：让 ascend_910b 作为一个“第一等公民系统”进入框架。
- 产出：可估算、可生成、可展示。

### Phase 2：推理栈适配 + 数据补齐
- 目标：补齐 MindIE/Ascend 推理栈、HCCL 通信数据、关键 op 数据。
- 产出：估算结果开始具备实测校准能力。

### Phase 3：产品化闭环
- 目标：support matrix、回归测试、文档、CI 门禁完整。
- 产出：可用于正式评估与对外展示。

---

## 3. 详细任务拆解（按模块）

---

# Epic A：系统语义与规格层

## A1. 新增平台 vendor 标识
- **任务目标**：在系统规格中明确区分 `nvidia` / `ascend`。
- **建议改动范围**：
  - `aic-core/src/aiconfigurator_core/systems/`
  - `src/aiconfigurator/generator/facts/`
  - 相关 SDK 入口
- **具体任务**：
  1. 定义 vendor 字段规范（如 `vendor: ascend`）。
  2. 支持 vendor 级别条件逻辑。
  3. 输出 vendor 规范文档。

## A2. 新增 ascend_910b.yaml 系统规格
- **任务目标**：定义 910B 的硬件能力入口。
- **建议新增文件**：
  - `aic-core/src/aiconfigurator_core/systems/ascend_910b.yaml`
- **必须覆盖字段**：
  - 显存容量
  - 显存带宽
  - BF16/INT8 算力
  - FP8 是否支持
  - 节点内互联（HCCS）
  - 节点间互联（RoCE/IB）
  - 通信库类型（HCCL）
  - 校正因子（可先用默认值）

## A3. 重构 SM version 语义
- **任务目标**：避免用 NVIDIA SM 概念强套 910B。
- **建议方式**：
  - 新增 `platform.arch_version`
  - 或新增 vendor-specific key 兼容层
- **具体任务**：
  1. 定义 ascend 平台版本标识。
  2. 更新 SDK 判断逻辑。
  3. 更新 generator 判断逻辑。

---

# Epic B：通信模型层

## B1. 将 nccl 扩展为 comm_backend
- **任务目标**：支持 `nccl / hccl / oneccl` 等通信后端。
- **建议改动范围**：
  - `aic-core/src/aiconfigurator_core/sdk/`
  - `aic-core/src/aiconfigurator_core/systems/`
- **具体任务**：
  1. 新增 `misc.comm_backend` 字段。
  2. 将 `misc.nccl_mem` 映射到统一的 `comm_mem`。
  3. 兼容老系统（老字段自动转为新抽象）。

## B2. 新增 HCCL 数据目录规范
- **任务目标**：为后续真实数据补齐打基础。
- **建议目录**：
  - `systems/data/ascend_910b/comm/hccl/<version>/`
- **具体任务**：
  1. 定义 HCCL 数据 schema。
  2. 定义 metadata 标准。
  3. 准备 placeholder 数据集。

## B3. 通信估算逻辑抽象
- **任务目标**：通信时延/带宽估算不再默认走 NCCL。
- **具体任务**：
  1. 抽离通信 backend 选择逻辑。
  2. 支持按 vendor/backend 查询 perf data。
  3. 支持 estimate-only fallback。

---

# Epic C：Perf Database 层

## C1. 新增 ascend_910b 目录骨架
- **建议新增目录**：
  - `systems/data/ascend_910b/attention/<backend>/<version>/`
  - `systems/data/ascend_910b/gemm/<backend>/<version>/`
  - `systems/data/ascend_910b/moe/<backend>/<version>/`
- **任务目标**：建立未来数据落盘标准。

## C2. 制定 kernel_source 映射规则
- **任务目标**：为后续 Ascend kernel 分类提供标准。
- **建议映射**：
  - ascend_attn
  - ascend_moe
  - ascend_gemm
  - hccl

## C3. 首期采用 estimate-only 模式
- **任务目标**：先让流程跑通。
- **具体任务**：
  1. 标记 ascend_910b 默认使用 SOL/EMPIRICAL/HYBRID。
  2. 保证没有实测数据时不会直接崩溃。
  3. 输出明确提示。

---

# Epic D：SDK 估算层

## D1. 抽象 vendor/platform 判断逻辑
- **任务目标**：避免 if-else 硬编码。
- **建议新增函数**：
  - `vendor_for_system()`
  - `platform_family_for_system()`
  - `is_ascend_system()`

## D2. 抽象算力字段
- **任务目标**：兼容 NVIDIA + Ascend 差异。
- **建议字段**：
  - bf16_peak_flops
  - int8_peak_flops
  - fp16_peak_flops
- **具体任务**：
  1. 新增字段规范。
  2. 在 estimator 中优先读取通用字段。
  3. 对老字段做兼容映射。

## D3. 抽象内存估算字段
- **任务目标**：统一通信内存表达。
- **建议改造**：
  - `misc.comm_mem`
  - `misc.comm_backend`
  - `misc.other_mem`

## D4. 新增 Ascend 平台校正因子策略
- **任务目标**：提升早期估算合理性。
- **建议做法**：
  - 默认保守系数
  - 可通过 YAML override
  - 可后续被 silicon data 替代

---

# Epic E：Generator 层

## E1. 新增 ascend_910b hardware profile
- **建议新增位置**：
  - `src/aiconfigurator/generator/facts/hardware.yaml`
- **建议内容**：
  - node selector
  - tolerations
  - 环境变量
  - 默认 backend 路由
  - 默认 comm backend

## E2. 模板支持 Ascend 设备选择变量
- **任务目标**：支持非 CUDA 设备选择。
- **建议方案**：
  - 新增 `_device_env` 平台条件分支。
  - 为 ascend 增加：
    - `ASCEND_VISIBLE_DEVICES`
    - 或 vendor 推荐变量
- **建议优先修改模板**：
  - `src/aiconfigurator/generator/config/backend_templates/trtllm/run.sh.j2`
  - `src/aiconfigurator/generator/config/backend_templates/vllm/run.sh.j2`
  - `src/aiconfigurator/generator/config/backend_templates/sglang/run.sh.j2`

## E3. backend mapping 增加 ascend/mindie 项
- **建议改动文件**：
  - `src/aiconfigurator/generator/config/backend_config_mapping.yaml`
- **任务目标**：
  - 为后续 MindIE 参数映射预留结构。

## E4. generator 默认值 vendor-neutral 化
- **重点检查项**：
  - CUDA-specific 默认注释
  - CUDA graph 相关默认逻辑
  - NCCL 默认假设
  - 默认 engine args

---

# Epic F：Collector 层

## F1. 增加 ascend 收集路径
- **建议新增目录**：
  - `collector/mindie/`（或对应的 ascend 收集器目录）
- **任务目标**：
  - 为未来实测数据采集提供入口。

## F2. 设备探测逻辑扩展
- **任务目标**：识别 NPU 环境。
- **建议改造文件**：
  - `collector/capabilities.py`
  - `collector/helper.py`
- **建议逻辑**：
  - CUDA available
  - XPU available
  - Ascend NPU available

## F3. capability 语义扩展
- **任务目标**：表达 ascend 平台 dtype/op 能力边界。
- **建议改造**：
  - `cases/capabilities.yaml`
  - registry unverified 标记体系

---

# Epic G：Support Matrix 与文档层

## G1. 新增 ascend_910b support matrix
- **建议新增文件**：
  - `aic-core/src/aiconfigurator_core/systems/support_matrix/ascend_910b.csv`
- **任务目标**：
  - 进入 support matrix 扫描和展示体系。

## G2. 更新支持矩阵索引
- **建议改动**：
  - `aic-core/src/aiconfigurator_core/systems/support_matrix/index.json`
- **任务目标**：
  - 让构建流程识别新系统。

## G3. 补齐用户文档
- **建议新增文档**：
  - 昇腾支持说明
  - 限制条件说明
  - 已知偏差说明
- **建议位置**：
  - `docs/` 或 `docs/ckw/`

---

# Epic H：验证与质量保障

## H1. 单测覆盖
- **建议测试点**：
  - ascend_910b.yaml 解析
  - vendor 判断
  - comm_backend 选择
  - generator profile 解析
  - estimate-only 路径

## H2. generator validator
- **建议任务**：
  - 增加 ascend/mindie backend 验证入口
  - 校验生成产物结构

## H3. regression gate
- **建议任务**：
  - 增加 ascend 基线 case
  - 防止平台抽象回归

## H4. CI 门禁
- **建议任务**：
  - `pytest -m unit`
  - lint
  - support matrix check
  - generator output snapshot check

---

## 4. 建议里程碑

### M1（2 周）
- 新增 ascend_910b 系统规格
- 新增 vendor 字段
- 新增 generator hardware profile
- 新增 estimate-only 主链路支持

### M2（4 周）
- 抽象 comm_backend
- 抽象 SDK platform/vendor 逻辑
- generator 模板支持 ascend device env
- 新增 ascend support matrix 骨架

### M3（8 周）
- 新增 ascend/mindie backend mapping
- 新增 HCCL 数据目录规范
- 补齐 ascend regression tests
- 输出首版支持矩阵文档

### M4（12 周+）
- 补齐 silicon data
- 精调估算因子
- 产品化发布

---

## 5. 风险清单

### 高风险
- MindIE 生态与现有 backend 抽象不完全对齐。
- 910B 部分特性无法用现有字段直接表达。
- 通信模型差异导致估算偏差明显。

### 中风险
- generator 模板条件分支膨胀。
- 多平台 vendor 判断逻辑复杂度上升。
- 首期 estimate-only 与用户预期差距较大。

### 低风险
- support matrix 增量维护。
- 文档补齐工作量。
- 单测补齐工作量。

---

## 6. 首期最小任务集（建议优先执行）

如果资源有限，建议优先做以下 8 件事：

1. 新增 `ascend_910b.yaml`
2. 新增 vendor/platform 字段定义
3. 新增 `ascend_910b` hardware profile
4. 新增 `vendor_for_system()` 系列判断
5. 抽象 `comm_backend` 字段
6. generator 模板支持 ascend device env
7. 新增 ascend support matrix 骨架
8. 新增 ascend estimate-only regression tests

这 8 项完成后，基本可以拿到“最小闭环”。

---

## 7. 最后结论

这份任务拆解的核心逻辑是：

- **不要把“支持昇腾 910B”理解成一个单点改动**
- 它是一个“平台抽象层工程”
- 最合适的推进方式是：**先抽象，再补齐数据，再产品化**