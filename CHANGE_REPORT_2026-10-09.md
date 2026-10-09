# asharp-verl 代码改动报告（2026-10-09）

## 范围与核查方法

本报告记录工作区相对 `bec9ef74`（上游 `v0.7.1`）的改动，供以后追查本次提交。核查时有 21 个已跟踪文件被修改、36 个非忽略文件新增；未把上游历史提交算作本次改动。按**功能单元**分类；同一功能涉及的代码、配置和测试放在一起。实验报告中的训练数字来自仓库已有记录，原始日志、模型和数据保存在仓库外的 `/data1/jty/`，此次未重新运行训练或独立复核这些数字。

## 功能新增

### 1. 策略优势与 critic 回报分离的 GAE 参数

`verl/trainer/ppo/ray_trainer.py` 的 `compute_advantage` 增加 `lam_policy`、`lam_critic`。前者计算 actor 的优势，后者计算 critic 的训练回报；两者相等时只运行一次 GAE，不相等时分别计算。两者均为可选值，未设置时继承旧的 `algorithm.lam`，并检查有效值为 `[0, 1]`。训练循环把配置传入该函数。

`verl/trainer/config/algorithm.py` 定义两个字段；`ppo_trainer.yaml`、`ppo_megatron_trainer.yaml`、四份 `_generated_ppo_*.yaml` 和两份 `tests/trainer/config/legacy_*.yaml` 同步暴露或固化默认值。这使终局奖励场景能够给 actor 使用完整回报传播（例如 λ=1），同时单独选择 critic 目标；默认配置保持旧语义。`tests/trainer/ppo/test_decoupled_gae_on_cpu.py` 覆盖分离计算、旧默认值和非法输入。这里只改变 GAE 路径，其他优势估计器不使用这两个参数。

### 2. 固定策略 critic 预训练与评估流程

`verl/experimental/critic_pretraining/collect_policy_rollouts.py` 从输入 parquet 读取 prompt，调用 vLLM 固定策略采样，可加载策略 LoRA、内置或自定义奖励函数，并把精确的 prompt/response token ID、奖励及采样索引写成逐轨迹 parquet。`pretrain.py` 读取这些轨迹，把终局奖励折扣为每个响应位置的 Monte Carlo 目标；价值预测对齐到响应动作的前一个 token。它支持多卡 Accelerate、全量或 LoRA 训练、梯度累积、断点继续、周期保存和导出。LoRA 导出包含可训练的 `score` head；导出目录还写入 PPO 加载参数元数据。`evaluate.py` 在固定轨迹上输出 EV、误差、偏差、AUC、校准、序列位置指标，并可用相同价值估计比较多个 GAE λ 的位置优势。`README.md` 给出收集、训练、评估和载入方法。

PPO 接入方面，`verl/workers/config/critic.py`、`verl/trainer/config/critic/dp_critic.yaml`、生成配置及 legacy 配置补齐 `critic.model.lora_adapter_path` 的配置声明。实际加载逻辑在基线的 `fsdp_workers.py` 已存在，本次使其成为可配置接口。`tests/experimental/critic_pretraining/test_pretrain_on_cpu.py` 覆盖目标、对齐、截断和统计计算。此功能位于 `experimental`，现有测试没有覆盖完整多 GPU 训练或真实模型导出再载入。

### 3. Qwen2.5-Math LoRA-32 实验的奖励与启动方案

`experiments/qwen25_math_lora32/reward_math_verify.py` 对最后一个完整 `\\boxed{}` 答案给出二值终局奖励：先沿用严格匹配，再用 Math-Verify 判等；没有格式分或部分分。`requirements.txt` 记录评分依赖，`tests/experimental/reward_score/test_math_verify_terminal_on_cpu.py` 覆盖等价答案、错误答案及严格评分兼容性。

同目录的 `run_qwen25_math_15b_math_ppo_lora32.sh` 固定该实验的模型、数据、96×4 逻辑批次、LoRA-32、PPO 步数和验证周期；可通过环境变量调整 3/4 GPU、两个 λ、学习率、critic adapter 和奖励函数。这是特定实验的启动脚本，包含 `/data1/jty/`、`/data2/jty/` 的本机路径，换环境需要修改。`README.md` 说明目录分工。

## 功能完善

### 4. 可重复的 actor/critic 初始化

`verl/workers/fsdp_workers.py` 在构建 actor 或 critic 的模型与 LoRA 时，可选地设置 CPU RNG 种子，完成后恢复原 RNG 状态，减少随机 value head 等初始化对对照实验的干扰。字段见 `verl/workers/config/actor.py`、`critic.py` 及各自 YAML；默认 `null` 保持既有行为。实验启动脚本将两者设为 42。该种子针对初始化阶段；不能据此推断整条训练链完全确定。

### 5. LoRA checkpoint 与 value model 导出

`ray_trainer.py` 在 actor/critic 保存检查点时写 `lora_train_meta.json`，记录张量无法可靠反推的 LoRA rank、alpha 与任务类型。`verl/model_merger/base_model_merger.py` 读取这些值，并为旧检查点提供 `--lora-alpha`、`--lora-task-type` 覆盖；处理 PEFT 的 enum/字符串差异、`modules_to_save.default` 参数及原始模块参数名，确保 critic 的 `score`/`classifier` head 随 adapter 导出。`--is-value-model` 选择 `AutoModelForTokenClassification`，导出配置的 `architectures` 也改成实际 value model 类。`verl/utils/model.py` 与 `fsdp_workers.py` 在 critic 建立时同步该架构字段，防止后续按 causal LM 错误重建。对应测试为 `tests/model_merger/test_lora_critic_state_on_cpu.py`、`tests/trainer/ppo/test_checkpoint_metadata_on_cpu.py` 和 `tests/utils/test_model_on_cpu.py`。

## 问题修复

### 6. critic-only 阶段的保存与下一轮 rollout

`ray_trainer.py` 在跳过 actor 更新的 `critic_warmup` 分支补上按频率、末步或实例到期保存检查点的逻辑，并调用权重更新管理器唤醒/同步休眠的 rollout 副本。此前保存及唤醒位于 actor 更新分支，长时间 critic-only 训练可能没有可恢复检查点，下一轮采样也可能受影响。该逻辑在已有实验记录中用于完整 critic warm-up；此次 CPU 测试没有直接覆盖 Ray/vLLM 生命周期。

### 7. vLLM colocate IPC 路径

`verl/workers/rollout/vllm_rollout/utils.py` 和 `vllm_rollout.py` 两端都把 ZMQ Unix socket 目录改为 `TMPDIR`，未设置时仍为 `/tmp`。两端路径保持一致，适配集群单独指定临时目录的部署。

## 实验用工具

### 8. 通用 PPO 诊断辅助

`verl/experimental/ppo_diagnostics/advantage.py` 按正确/错误样本统计序列前后窗口的优势均值、正优势比例和 token 数；`one_batch.py` 对同一已采样 batch 计算 λ=0.95/0.99/1.0 的诊断指标，并携带 actor/rollout 概率一致性指标。两者为只读工具，未接入生产训练循环。`tests/experimental/ppo_diagnostics/` 中三份测试分别覆盖位置汇总、单 batch 无修改诊断及单步 PPO/3、4 GPU 微批次语义。

### 9. MATH 日志、固定策略和评分分析工具

`experiments/qwen25_math_lora32/analysis/` 下八个脚本构成一个实验调查工具集：`analyze_qwen25_math_ppo.py` 对照 PPO 日志并绘图；`analyze_advantage_diagnostic.py` 整理真实 batch 的位置优势；`analyze_critic_only.py` 统计 critic-only 学习曲线；`compare_critic_holdout.py` 对照不同 critic 检查点的固定 holdout 指标与校准；`analyze_fixed_policy_eval.py` 计算固定策略 pass@1、pass@4 及配对 bootstrap；`audit_math_reward_equivalence.py` 和 `compare_math_scorers.py` 审查严格评分与数学等价评分差异；`check_vllm_lora_refresh.py` 做 vLLM adapter A/B/A 切换检查。这些脚本读取既有日志/结果或本机模型，不是线上训练路径。

## 实验产物

### 10. 已完成调查的文字结论与后续协议

`experiments/qwen25_math_lora32/reports/` 中五份 Markdown 分别保存原 PPO 代码及日志审计、固定策略对照评估、真实 batch 的 credit 诊断、770 步 critic-only 训练与固定 holdout 比较、以及 λ=1 加暖启动 critic 的后续实验协议。其作用是保留当时的数据、判读边界和下一次实验的预设标准，不是模型权重或原始日志。关键记录包括：原 PPO 对照未见稳定 pass@1 提升；真实 batch 的早期优势区分在 λ=0.95 时弱、λ=1 时明显增强；critic-only 的固定 holdout 在 step 693 校准优于 step 770。具体数字和外部产物路径以各原报告为准。仓库中没有纳入 parquet、图片、训练日志或 checkpoint，后查时仍依赖报告列出的外部路径。

## 实验所需一次性用品

### 11. 历史配方、备份及临时数据检查

`experiments/qwen25_math_lora32/archive/gsm8k/reward_boxed.py` 保留已被 MATH 实验取代的 GSM8K 数值 `\\boxed{}` 二值评分器，供历史复现。`examples/ppo_trainer/run_qwen25_math_15b_math_ppo_lora32.sh.orig` 是早期启动脚本快照；其主脚本现位于实验目录。仓库根部 `tmp.py` 是一次性 parquet 检查脚本，用于打印 MATH train/test 的难度与题型分布、原始索引范围；它有本机固定数据路径，不是通用入口。这三项作为本次状态的一部分保留，后续若清理应另起提交并记录原因。

## 验证与限制

- `git diff --check` 通过；新增 Python 模块和测试的 `compileall` 通过。
- 使用 `/data1/jty/anaconda3/envs/asharp-verl/bin/python` 运行上述九组针对性 CPU 测试，结果 **25 passed**、2 条第三方依赖警告。系统默认 Python 运行同一批测试时因缺少 `orjson` 在两个测试的收集阶段失败；这是环境差异，不是测试断言失败。
- 未运行多 GPU、Ray/vLLM 训练、完整 checkpoint 导出/恢复和外部 `/data1/jty/` 实验产物复核。因此初始化、critic-only 生命周期、IPC 和大型预训练流程仍需在目标训练环境中做端到端验证。
