# MATH LoRA-32 PPO: code and log audit (2026-10-09)

No follow-up training was started after either run. Both runs were stopped early because sampled validation accuracy showed no sustained improvement.

## Runs and comparison

The baseline and the critic-warm-up run used Qwen2.5-Math-1.5B-Instruct, MATH-lighteval, rank-32 actor and critic LoRA, 96 prompts per global batch, four sampled responses per prompt, 1024 maximum response tokens, terminal 0/1 reward, no KL reward/loss, actor learning rate 5e-6, critic learning rate 1e-5, and one actor/critic optimizer step per batch. The only intended difference was `trainer.critic_warmup=232`: steps 1–231 updated only the critic. Both had a ten-epoch upper limit; neither reached it.

| Run | Final logged step | Actor updates | Validation sampled pass@1 at step 0 | Best observed | Last complete validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 477 | 477 | 66.27% | 66.96% (step 385) | 66.23% (step 462) |
| Critic warm-up | 544 | 313 | 66.05% | 66.70% (step 539) | 66.70% (step 539) |

The warm-up actor remained frozen through step 231, yet sampled validation pass@1 moved from 66.05% to 65.04%; this roughly one-percentage-point span is a direct measure of evaluation sampling variation in this setup. The last warm-up value is therefore not persuasive evidence of a gain. Train reward averaged 0.7405 in the baseline and 0.7382 in the warm-up run; the easier train split should not be compared directly to test pass@1.

## Code-path findings

1. **The configured one-update regime makes PPO clipping inactive.** `dp_actor.py:update_policy` detects a single minibatch and one epoch and replaces `old_log_prob` with the current forward pass detached. Consequently the ratio is exactly one when the gradient is taken; `actor/ppo_kl`, `actor/pg_clipfrac`, and `actor/pg_clipfrac_lower` are structurally zero. The actor is receiving an on-policy, advantage-weighted policy-gradient update, but clip ratio 0.2 has no effect on that update. Zero clip/KL metrics do not demonstrate stability. This is a consequence of the requested single update per batch, not evidence by itself of a bug.
2. **Actor LoRA parameters really changed.** Comparing baseline actor checkpoints at steps 385 and 462 found all 392 tensors (36,929,536 parameters) changed; relative parameter-delta L2 was about 1.7%. The nonzero actor gradient norm corroborates this. The vLLM synchronization path removes the previous LoRA before adding the new one (`vllm_rollout/utils.py:update_weights_from_ipc`); no source-level stale-adapter defect was found. Runtime equivalence of the sampling model to the latest checkpoint has not yet been directly measured, so this remains a useful focused diagnostic before a new run.
3. **A real critic-warm-up lifecycle bug was fixed and smoke-tested.** The trainer sleeps colocated vLLM replicas after generation. Previously, when actor updates were skipped during warm-up, the usual weight-sync/wake path was skipped too, and the next rollout failed. The trainer now calls `checkpoint_manager.update_weights()` in the critic-only branch. A three-step smoke test passed before the formal run.
4. **The separate GAE lambdas follow the intended tensors.** Policy advantages use lambda 0.95; critic targets use lambda 1.0. Under gamma 1 and terminal-only reward, critic targets are Monte Carlo outcomes. The GAE implementation masks padding. No indexing or value-target mix-up was found in this path.
5. **Reward is strict normalized boxed-answer equality.** Both training and validation route `DigitalLearningGmbH/MATH-lighteval` to `math_reward.compute_score`, which extracts the last `\boxed{}` and compares normalized strings. Algebraically equivalent expressions can still be scored as different. The current logs do not save per-response text, so the false-negative rate cannot be quantified from these logs. This is a reward-quality risk, not a proven cause of the flat curve. Roughly 7–8% of responses hit the 1024-token cap, another possible source of missing boxed answers.
6. **The named `best@4` metric is not empirical pass@4.** Validation uses bootstrap resampling of the four candidates *with replacement* before taking a maximum. `acc/mean@4` is the mean sampled accuracy (pass@1 estimate). Use raw per-prompt outcomes to calculate true pass@4; do not cite `best@4/mean` as pass@4.
7. **The 3/4-GPU batch settings are logically matched, not bitwise reproducible.** In the legacy FSDP worker, global 96 prompts × 4 responses becomes 384 trajectories; per-rank minibatches are 128 on three GPUs or 96 on four. With microbatch 16, these are eight or six accumulation microbatches per rank, followed by one optimizer step. Both actor and critic use the same logical global batch. The default `token-mean` loss is first computed within each microbatch and then equally averaged across microbatches; it is not the exact token-weighted mean of all 384 sequences. Rank-dependent balancing, stochastic generation and floating-point reduction may still change trajectories and updates when GPU count changes. Thus these are comparable settings, not identical random-number streams.

## Metrics and interpretation

For the baseline, first/last 50 training steps: mean train reward 0.7444/0.7428; critic value loss 2.176/0.0559; critic explained variance -18.31/+0.416; actor entropy 0.164/0.112; mean response length 536/528 tokens; max-length fraction 7.28%/7.08%. For the warm-up run, critic value loss 1.469/0.0576 and explained variance -10.67/+0.410; after actor updates, entropy also fell to about 0.119. The critic learned a nontrivial value predictor, but neither run showed sustained accuracy or reward improvement. The initial critic head differed markedly across runs despite the nominal seed, so this is not a perfectly paired deterministic comparison.

The most promising *hypothesis*, not yet proven, is weak credit assignment from terminal reward under actor lambda 0.95 for approximately 530-token responses. The direct terminal component is attenuated by 0.95^100 ≈ 0.006 after just 100 tokens; earlier tokens depend heavily on critic TD estimates, and the final explained variance around 0.4 leaves substantial residual error. The critic-warm-up result argues against **initial** critic noise as the sole failure mechanism. A strict scorer and response truncation may further dilute useful reward. A too-high/too-low learning rate or train/test difficulty shift remains possible, but should not be grid-searched before the targeted checks below.

## Before another training run

1. Run an **evaluation-only** checkpoint comparison on fixed prompts/seeds and save raw generations, boxed extractions, outcomes and log-probs. This should verify vLLM adapter refresh, calculate true pass@4, and measure scorer false negatives (manually or with an independent math-equivalence judge on a stratified sample). It is not a new RL training run.
2. Add focused CPU tests for the one-step PPO-loss semantics, global 3/4-GPU batch scaling, and terminal-reward/GAE masks. Consider a small direct actor-versus-vLLM log-prob check after a weight sync.
3. If these checks pass, choose **one** next training hypothesis: actor lambda 0.99 (still within the requested range) to propagate terminal reward farther, while keeping all other substantive settings and evaluation protocol fixed. Compare equal numbers of actor updates and report sampled pass@1 with uncertainty, true pass@4, train reward and critic dynamics. Do not start this run automatically.

## Artifacts

- Logs: `/data1/jty/logs/ppo_lora32_math/`
- Parsed per-step CSVs, complete numeric summary, validation CSV and four PNG trend plots: `/data1/jty/analysis/ppo_lora32_math/`
- Reproducible parser/plot script: `experiments/qwen25_math_lora32/analysis/analyze_qwen25_math_ppo.py`
- Experiment launcher: `experiments/qwen25_math_lora32/run_qwen25_math_15b_math_ppo_lora32.sh`
