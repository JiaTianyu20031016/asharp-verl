# MATH LoRA-32 PPO: fixed-policy evaluation follow-up

This is an inference-only follow-up to [the code/log audit](MATH_PPO_AUDIT.md). No model parameter was updated and no new PPO training run was started.

The base policy and baseline actor checkpoints at steps 385 and 462 were evaluated on the same 693 filtered MATH test prompts, four samples per prompt, temperature 1, top-p 1, seed 42, and at most 1024 response tokens. A sample prompt's token IDs matched the training chat-template path. The raw per-response parquet files are under `/data1/jty/analysis/ppo_lora32_math/`.

| Policy | Sampled pass@1 | Empirical pass@4 | Length-stopped | Zero-score responses verified symbolically equivalent |
| --- | ---: | ---: | ---: | ---: |
| Base | 67.24% | 79.22% | 8.08% | 38 / 2772 |
| Step 385 | 66.99% | 78.50% | 7.54% | 41 / 2772 |
| Step 462 | 66.88% | 78.79% | 7.32% | 42 / 2772 |

The prompt-paired bootstrap estimate for step 462 minus base is -0.36 percentage points in sampled pass@1, with a 95% interval of [-1.70, +0.97] points; for empirical pass@4 it is -0.43 points, interval [-2.60, +1.73]. Step 385 is likewise indistinguishable from base. With fixed sampling seed, 92.8% of the response strings differed between base and step 462. The exported LoRA adapter therefore affects inference, though this static comparison does not prove that every *in-training IPC weight refresh* was successful.

The fixed-policy pass@4 values are calculated by taking the maximum of the four actual response scores for each prompt. They must not be confused with the training log's bootstrap-with-replacement `best@4/mean` metric.

The standard MATH scorer requires exact normalized boxed-answer strings. A separate local SymPy audit checked whether strict-zero boxed predictions and reference expressions simplify to the same expression. All 38 flagged base-policy pairs were manually reviewed and are truly equivalent: examples include `\frac{175}{2}` versus `87.5`, `58500` versus `58,500`, and an expanded polynomial versus its factored form. This establishes a **lower bound** of 1.37 percentage points of mis-scored base responses. Another 131 boxed-but-zero responses could not be parsed by this audit and remain unclassified. Crediting only the flagged pairs gives sampled pass@1 of 68.61% / 68.47% / 68.40% for base / step 385 / step 462; correcting these clear scorer misses does not reveal a training gain. The diagnostic did not alter training rewards or the primary exact-match metric.

Every length-stopped response in the base-policy fixed evaluation scored zero. The original training logs had 7–8% maximum-length responses, so truncation is a measurable source of zero reward, though not yet proven to be the principal cause of stagnation.

Another comparability limitation in the completed runs: the FSDP critic initialization path did not explicitly seed the random value head. `data.seed=42` seeded the data sampler, not that head; initial critic predictions differed. An opt-in `model_init_seed` was subsequently added for both actor and critic, and the MATH launcher sets each to 42 while restoring the previous CPU RNG state after construction. This change passed config composition on both 3 and 4 GPUs, but no new PPO run has yet exercised it.

An in-process vLLM LoRA-ID A/B/A check gave a useful but qualified result: the engine acknowledged all removals; among 64 prompts, seven or eight greedy outputs changed even when the same adapter was repeated without switching, so exact output restoration is not a valid pass/fail criterion. In the final run, 55 prompts were stable across all four A evaluations (repeat, same-adapter reload, and reload after B); 16 of those 55 changed under adapter B. This supports that vLLM's remove/re-add mechanism can replace a LoRA rather than always retaining the first adapter. It still does not directly test the training-time tensor-IPC transfer path.

A toy batch passed through the exact GAE/whitening implementation used 530-token responses, 284/384 correct outcomes, zero initial critic values and terminal reward. The mean normalized advantage of the first 100 tokens in a correct response was -0.240 at actor lambda 0.95 and -0.579 at 0.99, but +0.593 at 1.0. Thus a smaller actor lambda together with global advantage whitening can discourage even the early tokens of successful long responses when the critic is uninformative. This toy is a mechanism check, not a measurement of actual training advantages after the critic learns. It weakens the earlier assumption that 0.99 alone is necessarily sufficient; the next choice should be based on real per-position advantage diagnostics, with lambda 1.0 considered if needed.

Focused CPU checks for the actor/critic GAE split, one-update PPO clipping semantics and idealized 3/4-GPU microbatch averaging passed (5 tests). Remaining targeted checks before another PPO run: directly verify in-training LoRA tensor-IPC refresh or compare actor/vLLM log-probs immediately after sync; capture real advantage statistics by response position; make critic initialization deterministic for controlled comparisons. Only then select a single training change while preserving the user-fixed settings.

Reproduction scripts: `experiments/qwen25_math_lora32/analysis/analyze_fixed_policy_eval.py`, `experiments/qwen25_math_lora32/analysis/audit_math_reward_equivalence.py`, and `verl/experimental/critic_pretraining/collect_policy_rollouts.py` (accepts rank-32 LoRA adapters explicitly).
