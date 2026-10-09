# Qwen2.5-Math-1.5B LoRA-32 PPO experiment

This top-level experiment directory contains the experiment-specific launcher, offline analysis
utilities, and reports for the MATH LoRA-32 PPO investigation. It is kept
separate from the generic PPO examples because the scripts encode this run's
dataset paths, evaluation protocol, and metric assumptions.

- `run_qwen25_math_15b_math_ppo_lora32.sh`: reproducible 3/4-GPU launcher.
  Its logical batch stays at 96 prompts × 4 responses. `ACTOR_LAMBDA`,
  `CRITIC_LAMBDA`, `ACTOR_LR`, `CRITIC_LR`, and `CRITIC_ADAPTER_PATH` make
  targeted follow-up runs explicit without duplicating launch scripts.
- `reward_math_verify.py`: experiment-scoped Math-Verify binary terminal reward;
  set it through `REWARD_FUNCTION_PATH`. It adds neither format reward nor KL.
- `requirements.txt`: pinned scorer dependency; the current environment uses
  the existing ANTLR 4.9.3 runtime to remain compatible with Hydra/OmegaConf.
- `analysis/`: log parsing, fixed-policy evaluation, reward auditing, and
  vLLM/advantage diagnostics. `analyze_critic_only.py` produces per-step and
  per-epoch statistics plus critic-learning and positional-credit plots;
  `compare_critic_holdout.py` compares fixed-rollout metrics and calibration
  across saved critic steps.
- `reports/`: conclusions from the code audit and completed diagnostics.
  `MATH_CRITIC_ONLY_FULL_WARMUP.md` records the completed 770-step fixed-actor
  critic study and its fixed-holdout checkpoint comparison.
  `MATH_PPO_LAM1_WARMCRITIC_PROTOCOL.md` predeclares the hypothesis and stopping
  criteria for the targeted lambda-1 PPO follow-up.

Reusable runtime helpers live under `verl.experimental.ppo_diagnostics`;
their tests live under `tests/experimental/ppo_diagnostics`. No diagnostic
implementation is kept in the production PPO trainer package.
