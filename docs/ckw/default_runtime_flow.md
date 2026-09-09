## 一句话结论

你执行的命令本质上是：先构建 agg / disagg 两组优化任务，再并行做参数空间扫掠，最后在 SLA 约束下挑出吞吐最高的部署配置。

示例命令：

aiconfigurator cli default --model deepseek-ai/DeepSeek-V4-Pro --total-gpus 8 --system b300_sxm --backend vllm \
  --isl 3000 --osl 1000 --ttft 2000 --tpot 26 --top-n 16

## 整体执行脉络

### 1. CLI 入口解析参数

入口定义在 pyproject.toml:92：

aiconfigurator = "aiconfigurator.main:main"

default 子命令参数在 src/aiconfigurator/cli/main.py 中定义，关键函数是：

- _add_default_mode_arguments
- build_default_tasks(...)

--model, --total-gpus, --system, --isl, --osl, --ttft, --tpot 这些都会被解析成统一的优化输入。

### 2. 构建优化任务（Task）

真正把命令参数变成可执行任务的是：

- src/aiconfigurator/cli/main.py:1476 的 build_default_tasks(...)

默认 serving_mode=auto 时，框架会同时构建两类任务：

- agg（聚合 serving）
- disagg（分离 serving）

当 backend=vllm 时，任务名通常会是：

- agg
- disagg

每个任务最终被封装为：

- src/aiconfigurator/sdk/task_v2.py:446 的 Task 数据类

Task 里会固化这些关键输入：

- model_path
- system_name
- backend_name
- isl, osl, prefix
- ttft, tpot
- total_gpus
- database_mode

### 3. 执行所有实验任务

任务构建完成后，CLI 进入执行阶段，核心是：

- src/aiconfigurator/cli/main.py:1949 的 _execute_tasks(...)

这里会逐个执行每个实验；如果开了并行实验，还会并发执行。

单个实验的执行核心是：

- src/aiconfigurator/sdk/task_v2.py:2091 的 Task.run(...)

Task.run() 内部会做三件事：

1. self.validate()
2. 加载性能数据库
3. 根据 serving_mode 分发到对应扫掠函数

对应分支在 src/aiconfigurator/sdk/task_v2.py 约 2115~2149 行：

- agg -> sweep_agg(...)
- disagg -> sweep_disagg(...)

所以 default 模式不是“算一个答案”，而是自动在聚合和分离两种架构里做空间搜索。

### 4. 结果处理与择优

每个实验得到候选结果后，框架会调用：

- src/aiconfigurator/cli/utils.py:17 的 process_experiment_result(...)

这里会根据模式选择：

- pick_default(...)
- 或 pick_load_match(...)

并输出：

- best_config_df
- best_throughput
- pareto_frontier_df
- best_latencies

当 backend=auto 或任务数较多时，还会进一步按模式合并：

- src/aiconfigurator/cli/utils.py:154 的 merge_experiment_results_by_mode(...)

最终选出吞吐最高的实验：

chosen_exp = max(best_throughputs, key=best_throughputs.get)

### 5. 终端输出与可选落盘

最后框架会：

- 打印最终摘要
- 输出 top-N 配置表
- 绘制帕累托结果

如果你加了 --save-dir，还会把结果持久化。

在 Python API 层，对应的是：

- src/aiconfigurator/cli/api.py:155 的 cli_default(...)

它内部也是：

tasks = build_default_tasks(...)
result = _execute_and_wrap_result(tasks, mode="default", top_n=top_n, strict_sla=strict_sla)

最后返回一个 CLIResult：

- chosen_exp
- best_configs
- pareto_fronts
- best_throughputs
- best_latencies

## 你这条命令在做什么

这条命令实际触发的是：

model        = deepseek-ai/DeepSeek-V4-Pro
total_gpus   = 8
system       = b300_sxm
backend      = vllm
isl          = 3000
osl          = 1000
ttft         = 2000 ms
tpot         = 26 ms
top_n        = 16

框架会按这套目标：

1. 构建 agg 和 disagg 两类 Task
2. 基于 vllm 后端和 b300_sxm 的性能数据库做配置扫掠
3. 在所有可行配置里筛掉不满足 TTFT / TPOT 的结果
4. 计算帕累托前沿
5. 在 8 GPU 预算下输出前 16 个最优配置
6. 选出整体最优方案

所以这不是一个静态查询，而是一个“给定 GPU 预算 + SLA 目标，自动找最优推理部署配置”的优化过程。

## 结论：框架流程脉络

整体可以概括为：

1. CLI 参数解析
2. 构建 Task 任务集合
3. 加载性能数据库
4. 执行 sweep 扫掠
5. 计算帕累托与最优配置
6. 输出/落盘结果

其中最关键的分界点是：

- 参数 -> Task：build_default_tasks
- Task -> 搜索结果：Task.run()
- 搜索结果 -> 最优解：process_experiment_result / merge_experiment_results_by_mode

## 关键代码路径

- pyproject.toml:92
- src/aiconfigurator/cli/main.py
  - _add_default_mode_arguments
  - build_default_tasks
  - _execute_tasks
  - log_final_summary
- src/aiconfigurator/cli/api.py:155
  - cli_default
  - _execute_and_wrap_result
- src/aiconfigurator/sdk/task_v2.py
  - Task（约 446 行）
  - Task.run()（约 2091 行）
- src/aiconfigurator/cli/utils.py
  - process_experiment_result
  - merge_experiment_results_by_mode

## 补充说明

- default 模式默认是比较 agg 与 disagg
- recommend 模式是反向问题：给定负载目标，反推最少 GPU
- generate 模式更偏“快速生成朴素配置”，不是完整优化

如果只看一句话：

aiconfigurator cli default 的本质，是在给定模型、机型、后端和 SLA 下，自动搜索并对比聚合/分离部署的最优配置。
