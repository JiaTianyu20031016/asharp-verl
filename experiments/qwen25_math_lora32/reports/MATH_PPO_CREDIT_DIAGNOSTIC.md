# MATH PPO: real-batch credit-assignment diagnostic (2026-10-09)

This followed the [code/log audit](MATH_PPO_AUDIT.md) and [fixed-policy evaluation](MATH_PPO_FIXED_EVAL.md). It resumed the baseline step-462 actor and critic checkpoint, generated five consecutive training batches (96 prompts × 4 responses each), computed actor/vLLM log-probs, critic values and GAE advantages, and **returned before either optimizer step**. No checkpoint was written. The completed diagnostic log is `/data1/jty/logs/ppo_lora32_math/diagnostic-five-batches-from-baseline-step462.retry.log`.

The five batches had mean reward 0.765 (individual batch range 0.661–0.833) and mean response length 522 tokens. On the **same generated data and same trained critic values**, only the actor GAE lambda changed in the diagnostic calculation. Values below are the mean advantage of correct responses minus that of incorrect responses, averaged across the five batches:

| Actor lambda | First 100 response tokens | Last 100 response tokens |
| ---: | ---: | ---: |
| 0.95 (original) | 0.046 ± 0.025 | 1.069 ± 0.136 |
| 0.99 | 0.195 ± 0.067 | 2.236 ± 0.257 |
| 1.00 | 1.591 ± 0.146 | 1.225 ± 0.153 |

The ± values are standard deviations **across five batches**, not confidence intervals. At lambda 0.95, the correct-answer early advantage averaged only +0.006 and the incorrect-answer early advantage -0.040; early outcome separation was 23 times smaller than at the end of the response. Across all five batches this early gap was small (range 0.028–0.087). At lambda 1, the early correct/incorrect means were +0.299/-1.292. This is direct evidence that the current terminal-reward, long-response, lambda-0.95 GAE configuration supplies little *outcome-discriminating* signal to early reasoning tokens. It does not by itself prove that this is the only cause of flat validation accuracy, nor does it measure the gradient after token-level model Jacobians.

Actor-versus-vLLM probability agreement was high on every batch after the checkpoint weights were sent over the normal IPC path: mean absolute token-probability difference 0.00134, mean Pearson correlation 0.99938. This makes a wholly stale rollout adapter unlikely, although small inference/training numerical differences remain.

The next *single-hypothesis* training test should use actor lambda 1.0 while keeping critic lambda 1.0 and all user-fixed settings unchanged. It is more decisive for early credit than lambda 0.99, which still leaves an 11-fold early/late gap. The newly added opt-in actor/critic initialization seed makes future runs more reproducible; the historical baseline lacked this seed, so comparisons to it are not perfectly paired. Report raw sampled pass@1 and true pass@4 on fixed validation prompts, alongside reward, entropy, length, critic quality and per-position advantage trends. Stop early if no sustained improvement outside sampling variation.

Reproducible analysis: `experiments/qwen25_math_lora32/analysis/analyze_advantage_diagnostic.py`. Raw per-batch CSV, summary CSV and plot: `/data1/jty/analysis/ppo_lora32_math/diagnostic_batches.csv`, `diagnostic_position_summary.csv`, and `diagnostic_advantage_by_position.png`.
