# A-sharp with a fixed-temperature log-moment critic

This implements the re-anchored plug-in and entropic lambda-return from proposal
§8.2. It supports graded, negative, dense, and binary deterministic rewards with
finite exponential moments. It does not require a distributional or binary head.

The critic emits `u(s) = log E[exp(beta * G) | s]` at each pre-action prefix.
`values` stores these log moments; `returns` stores **log lambda targets**, not
ordinary returns. Existing value/return dashboard metrics consequently refer to
log-moment quantities in this mode (including explained variance in log space).

For a fixed positive beta and undiscounted returns:

```
z_t = beta*r_t + logaddexp(log(1-lambda)+u_old(s_next), log(lambda)+z_next)
A_t = r_t + (u_fitted(s_next)-u_fitted(s_t))/beta
```

Terminal continuation has `u = z = 0` (exponential moment 1). Lambda endpoints
are handled explicitly. Targets and advantages are detached and advantages are
not whitened. Training order is rollout -> critic inference -> frozen targets ->
critic update -> critic inference -> actor update. The second inference lets this
round's actor consume the newly fitted critic; fitting and evaluation on the same
rollouts can introduce overfitting and is not an unbiasedness guarantee.

## Losses

`critic.log_moment_loss=mse` is the default: `0.5*(exp(u)-exp(z))**2`.
It is MSE in moment space, **not** MSE between log values.

The optional `positive` loss is `u + exp(z-u)`. Its population minimizer also
satisfies `exp(u)=E[Y|s]`. Its weighting differs from MSE for finite models.
Both compute in FP32, ignore padding before exponentiation, and fail explicitly
on non-finite loss rather than silently clipping or changing the objective.
Large beta or reward scale can still cause overflow and high target variance.

## Running

Run from this worktree using an environment with verl's training dependencies.
For example, set `MODEL`, `TRAIN_FILE`, and `VAL_FILE` to your existing model and
parquet datasets (with a deterministic reward scorer supported by verl):

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=asharp \
  algorithm.asharp_beta=1.0 \
  algorithm.asharp_lam=0.95 \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  critic.value_type=log_moment \
  critic.log_moment_loss=mse \
  critic.model.path="$MODEL" \
  critic.model.lora_adapter_path=null \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  data.train_batch_size=32 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.default_local_dir=outputs/asharp \
  trainer.logger='["console"]'
```

Beta is fixed for a run and is recorded in `critic/asharp_meta.json`, also copied
into exported critic LoRA adapter directories. Resume and critic adapter loading
require matching metadata/beta. Ordinary PPO critic checkpoints/adapters are not
compatible. Start with a fresh critic head or a matching A-sharp checkpoint.
Changing beta requires a new critic training run; no adaptive beta is implemented.
`algorithm.asharp_lam` exclusively controls the entropic targets; GAE's `lam`,
`lam_policy`, and `lam_critic` do not control A-sharp.

The example uses one actor optimizer step per rollout refresh. If using n>1,
ensure the effective PPO minibatch spans the rollout batch when that is desired;
multiple minibatches/epochs reuse a stale direction and approximate re-anchoring.
Actor optimization still uses the existing PPO clipping/optimizer, not an exact
reverse-KL solve.

## Scope

Both legacy FSDP/FSDP2 critic and the FSDP engine loss path are wired. Other critic
backends fail configuration validation. Single-turn contiguous response masks
are required; tool interactions and rejection masks are not supported.
Every response's final valid token is treated as task termination, including
length-capped answers scored by the task. This is **not** a time-limit bootstrap
implementation for unfinished continuing episodes. Gamma must be 1 and reward
KL shaping is rejected. Sampling should match the current reference policy;
off-policy continuation correction is not implemented for moment targets.

No extra Q head is needed: deterministic dynamics and rewards give
`Psi(s,a)=exp(beta*r)*Psi(s_next)`. This identity does not directly cover stochastic
environment transitions. Binary reward is the special case
`Psi=1+(exp(beta)-1)*p_success`.

## Validation

```bash
python -m pytest tests/trainer/ppo/test_asharp_on_cpu.py \
  tests/trainer/ppo/test_decoupled_gae_on_cpu.py -q
```

These CPU tests do not substitute for multi-GPU Ray/vLLM end-to-end validation.
