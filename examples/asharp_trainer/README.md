# Continuous A-sharp PPO

The same scalar critic head and initialization now output
`H(s) = log E[exp(beta * G) | s] / beta`. At beta=0 this is an ordinary value
critic. Beta is fixed throughout a run; binary, graded, negative and dense
deterministic rewards share this implementation.

`values` contains H predictions; `returns` contains entropic lambda targets in
**H units** (`z / beta`, with an explicit beta=0 limit). They no longer contain
unscaled log moments. No new worker, model head or actor objective is introduced.

## Critic learning

For a bootstrapped next-state H and lambda-continuation R, the target is

```
R_t = r_t + log((1-lambda_c)*exp(beta*H_next)
               + lambda_c*exp(beta*R_next)) / beta
```

The implementation uses a centered reverse recursion, log1p/expm1, logaddexp and
a small-argument cumulant expansion. At beta=0 it uses ordinary lambda returns;
terminal H and continuation are zero. Labels are detached.

Default `critic.log_moment_loss=mse`:

```
(exp(beta*H) - exp(beta*R))**2 / (2*beta**2)
```

Optional `positive`:

```
(exp(beta*(R-H)) - 1 - beta*(R-H)) / beta**2
```

Both have loss and parameter-gradient limit `0.5*(H-R)**2`. Neither regresses
log targets by MSE when beta>0. The centered positive loss differs from the old
positive objective only by a positive scale and a target-dependent constant, so
its population minimizer is unchanged. MSE is the old moment MSE scaled by
1/beta**2 under u=beta*H.

Low-precision predictions are promoted to FP32 for estimator/loss arithmetic;
FP64 inputs retain precision for numerical checks. Stable exponent differences
avoid unnecessary overflow. Truly unrepresentable losses/estimates raise an
explicit error; exponential target variance remains an intrinsic limitation.

## Actor estimators

All methods use the unchanged PPO clipped actor objective. With
`a_t = r_t + H_next - H_t`, `q_t = exp(beta*a_t)` and
`D_t = expm1(beta*a_t)/beta`:

| Estimator | Actor signal | Exact-critic property |
| --- | --- | --- |
| `log_plugin` | Actor entropic lambda-return minus H_t | Generally Jensen-biased |
| `residual_gae` | a_t plus future D residuals weighted by lambda_a^k | Conditionally unbiased |
| `residual_exp` (default) | Same, additionally weighted by intervening products of q | Conditionally unbiased |

All three estimators reduce to the one-step signal a_t when lambda_a=0 and to raw
ordinary GAE(lambda_a) when beta=0. Approximate critics do not guarantee
unbiasedness. Exponential propagation may increase variance; no estimator is
assumed uniformly superior. Actor and critic lambdas are independent.

## Configuration

Add these overrides to an existing single-turn FSDP/FSDP2 PPO run:

```yaml
algorithm:
  adv_estimator: asharp
  gamma: 1.0
  use_kl_in_reward: false
  asharp_beta: 1.0
  asharp_actor_estimator: residual_exp
  lam_policy: 0.95
  lam_critic: 1.0
  asharp_normalize_advantages: false
  asharp_refresh_after_critic_update: false
critic:
  value_type: entropic_value
  log_moment_loss: mse
  log_moment_value_clip: false
```

`critic.asharp_beta` is automatically interpolated from the algorithm beta; do
not override it independently. A-sharp and ordinary PPO share `lam_policy` for
actor traces and `lam_critic` for critic targets. Each falls back independently
to `algorithm.lam` when null; the existing default `lam` is 1.0. There is no
A-sharp-specific lambda default. Set `lam_policy=0` for one-step A-sharp with any
of the three estimators.

Migration from earlier versions of this branch:

- Replace `asharp_lam_actor` with `lam_policy`.
- Replace `asharp_lam_critic` or `asharp_lam` with `lam_critic`.
- Replace `asharp_actor_estimator=one_step` with any supported estimator (for
  example `residual_exp`) and `lam_policy=0`.

The removed keys are no longer in the configuration schema. Manually inserted
obsolete keys are rejected with migration instructions rather than silently
ignored. The deprecated `value_type=log_moment` config name is still accepted,
but always denotes the H parameterization.

Default training order remains rollout -> one critic inference -> freeze actor
advantages and critic targets -> update critic -> update actor.
`asharp_refresh_after_critic_update=true` explicitly enables a second inference
to refresh advantages after fitting, without changing the frozen critic targets.
It is off by default because of its extra cost and same-batch fitting effects.

## Reproducing ordinary PPO at beta=0

The mathematical estimators return raw advantages. This repository's GAE also
whitens advantages, and its value loss uses PPO value clipping. For a matched
comparison use:

```yaml
algorithm:
  asharp_beta: 0.0
  asharp_actor_estimator: residual_gae  # log_plugin/residual_exp are identical here
  lam_policy: 0.95
  lam_critic: 0.95
  asharp_normalize_advantages: true
critic:
  log_moment_value_clip: true
  cliprange_value: 0.5               # match the baseline
```

Also match initialization, gamma=1, reward shaping, loss aggregation, optimizer,
rollouts and actor loss settings. Either normalized critic loss calls the same
ordinary PPO value loss at beta=0. Optional clipping applies in H units at all
beta, maintaining continuity. It is disabled by default to preserve plain
moment regression. With whitening off and clipping off the limit is unwhitened
GAE and plain value MSE, not the repository's default full PPO configuration.
Masked output advantages/targets are zero; compare GAE on valid tokens (the
legacy whitener may leave nonzero masked positions, which do not affect loss).

## Checkpoints and scope

New `asharp_meta.json` files use version 2 and record H prediction/target units
and fixed beta, also in exported LoRA adapter directories. Old u-head checkpoints
are rejected rather than silently loaded. Migrating one would require rescaling
the output head and deliberately handling optimizer state; automatic conversion
is not implemented. Ordinary value checkpoints without metadata are likewise
not auto-converted. Changing beta requires a new run with a suitable critic.

Only contiguous single-turn responses and deterministic transitions/rewards are
supported. The final valid token is task termination, including scored length
caps. Unfinished continuing-episode time-limit bootstrapping, stochastic tools,
rejection masks and policy-dependent KL reward shaping are not implemented.
Continuation data should come from the current reference policy. Actor clipping
and practical optimizer steps do not imply exact proximal solutions.

## CPU validation and controlled experiments

```bash
python -m pytest tests/trainer/ppo/test_asharp_on_cpu.py \
  tests/trainer/ppo/test_decoupled_gae_on_cpu.py -q
python -m examples.asharp_trainer.compare_estimators --output /tmp/asharp-tree.json
```

The comparison enumerates all trajectories of a graded depth-three tree and
reports conditional bias, variance, MSE and seven-parameter actor gradient error.
It sweeps beta, actor lambda and state-consistent critic errors, without Monte
Carlo uncertainty. See [the experiment report](reports/continuous_asharp_tree.md).
CPU validation does not substitute for multi-GPU Ray/vLLM or full LLM RL runs.
