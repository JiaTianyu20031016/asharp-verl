# asharp-verl 变更报告（2026-10-13）

## 范围和基线

本报告记录主分支 `asharp-verl-0.7.1` 自上次提交 `dd3b0fff`（`add critic adapter checkpoint auto save`）之后的工作区变化。本次按功能单元归类，而不是按文件数归类。此前的改动见 `CHANGE_REPORT_2026-10-09.md` 及提交历史；本报告不把它们重新算作新增。

本次纳入四处训练代码修改、一份临时工具替换、一份研究 PDF、实验目录清理和对应的失效测试清理。`experiments/` 中 23 个已跟踪文件的删除在检查开始前已暂存；其他源码修改、`.gitignore` 和 PDF 在检查开始时尚未暂存。

## 问题修复

### 1. 断点恰好在 epoch 边界时的恢复计数

`verl/trainer/ppo/ray_trainer.py` 的 `RayPPOTrainer.fit` 把恢复起始 epoch 从 `global_steps // len(train_dataloader)` 改为 `max(global_steps - 1, 0) // len(train_dataloader)`。正常新训练从第 0 个 epoch 开始不变；检查点步数不是 epoch 长度的整数倍时，商也不变。差异只出现在已完成步数刚好是 epoch 长度整数倍时：现在会重新进入该 epoch，让恢复的 `StatefulDataLoader` 迭代器完成边界状态，再进入下一个新迭代器，避免少运行一个 epoch。检查点仍在 `data.pt` 中保存 dataloader 状态；此处没有修改其存取格式。

边界：如果旧检查点缺少 `data.pt`，加载端本来就会从头启动 dataloader，此公式不能单独保证精确续训。此次没有端到端恢复训练记录。

### 2. Actor/Critic 动态微批次跨 rank 对齐

`verl/workers/actor/dp_actor.py` 与 `verl/workers/critic/dp_critic.py` 在计算 log probability/value 的动态批次路径，以及 PPO 更新的动态批次路径，调用 `prepare_dynamic_batch` 时都显式传入 `torch.distributed.group.WORLD`。此前未传 `dp_group`，各 rank 仅按本地 token 数决定微批次数；现在 `rearrange_micro_batches` 会在 WORLD 上取最大微批次数，使参与同一次分布式前向或反向的 rank 具有相同调用次数。这针对不同 rank 序列长度分布不同时可能出现的 FSDP 集体通信不同步。

该行为仅在动态批次开启、分布式进程组已初始化时生效；静态微批次不受影响。同步采用整个 WORLD，而非更小的并行子组。底层函数随后要求同步后的微批次数不超过本地样本组数；若某 rank 的本地样本数少于别的 rank 所需的微批次数，仍可能触发断言。此次没有多卡实训验证。

### 3. Actor PPO loss 指标按累积权重归一

`verl/workers/actor/dp_actor.py` 在每个实际处理的 micro-batch 上累加已有的 `loss_scale_factor`，返回前用权重总和归一 `actor/pg_loss` 和 `actor/kl_loss`。此前这两个指标把每次加权 loss 直接跨 PPO epoch、mini-batch 求和，日志值会随这些次数扩大；现在报告的是对应权重下的平均值。计算梯度所用的 `loss_scale_factor`、反向传播和优化器步骤没有改动。未启用 KL loss 时，`actor/kl_loss` 仍为零。若未处理任何 micro-batch，现在会明确抛出 `RuntimeError`，而不是返回零值。

这里只归一上述两个标量；`append_to_dict` 收集的其他指标及 Critic 的 `critic/vf_loss` 仍按原来的汇总方式记录，不能把所有日志字段都理解为同一统计口径。

## 实验用工具

### 4. 根目录 `tmp.py` 改为临时 GPU 吞吐测试

原脚本读取本机 MATH train/test parquet，打印难度、题型和样本索引分布。现在它接受 `--gpu`、`--duration`、`--size`，在指定 CUDA 卡上循环执行 FP16 方阵乘法，约每两秒输出按矩阵乘法运算量计算的 TFLOPS，结束时检查结果是否为有限值。默认使用 GPU 0、60 秒、8192×8192 矩阵。它是一次性性能探针，不参与训练、测试或打包；运行需要可用的 CUDA GPU，输出是该脚本负载下的测量值，不代表端到端训练吞吐。

## 实验产物

### 5. Finite-step / Asharp GAE 研究提案 PDF

仓库根目录新增 `finite_step_research_proposal_with_Asharp_GAE.pdf`。文件检查确认它是 549,140 字节、4 页的 PDF；SHA-256 为 `7761b5f2f8e49f80f769d6c74ab7605988b646b648e9d904993934696c559dce`。它不被训练代码引用，也没有对应的参数或配置改动。本报告仅将它登记为独立研究文档，没有把文件名当成已经实现或已经验证的算法结论。当前环境缺少 PDF 文本提取工具，因此未核对正文内容。

## 实验所需一次性用品清理

### 6. 将本机实验目录移出版本跟踪

根目录 `.gitignore` 新增 `/experiments/`，使今后整个根级实验目录默认不进入 Git。与之配套，已暂存删除的 23 个文件分属两个实验功能单元：

- `experiments/qwen25_math_lora32/`：MATH LoRA-32 启动脚本、Math-Verify 终局奖励实现与依赖清单、八个离线分析脚本、五份实验报告和一份历史 GSM8K 评分器。
- `experiments/qwen25_3b_gsm8k_ppo/`：GSM8K 两阶段 warmup/PPO 的启动与串联脚本、进度摘要脚本、说明和局部忽略配置。

这次清理不改动 `verl/experimental/` 下的通用实验代码。被删除文件仍可从 `dd3b0fff` 及更早的 Git 历史读取，但新检出的工作树不再直接提供这两个实验的运行配方、奖励函数和分析报告。旧的 `CHANGE_REPORT_2026-10-09.md` 仍描述当时的实验文件，应结合提交历史阅读，不能视为当前文件清单。

原测试 `tests/experimental/reward_score/test_math_verify_terminal_on_cpu.py` 直接导入本次删除的 MATH 奖励模块；在安装 `math_verify` 的环境里，这会使测试收集失败。随实验评分器清理一并删除这项测试。相应的评分行为不再有仓库内测试覆盖。

## 核查与边界

核查以 `git diff HEAD`、暂存区差异及相关调用链为依据。四个修改过的 Python 文件均通过只读 AST 语法解析，`git diff --check` 与 `git diff --cached --check` 均通过。使用 `/data1/jty/anaconda3/envs/asharp-verl/bin/python` 运行 `tests/utils/test_seqlen_balancing.py` 中两个无分布式 CPU 测试，结果为 **2 passed**，另有一条第三方 `pynvml` 弃用警告。

上述测试验证动态批次的基本拆分与还原，没有覆盖本次新增的 WORLD 跨 rank 同步；也没有运行 PPO 指标、GPU 吞吐或 checkpoint 边界恢复的端到端测试。因此这些运行时行为仍需在目标训练环境验证。
