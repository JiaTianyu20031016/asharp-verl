# Continuous A-sharp PPO

The default scalar critic outputs
`H(s) = log E[exp(beta * G) | s] / beta`. At beta=0 this is an ordinary value
critic. Binary RLVR can instead use the beta-independent success-probability
critic described below. The entropic-value mode supports binary, graded,
negative and dense deterministic rewards.

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

### Critic-free A-sharp GRPO from group returns

Set `algorithm.adv_estimator=asharp_grpo` with at least two responses per
prompt. Let `R_i` be the sum of valid token rewards for response `i`. For
each group of size `n`, the estimator computes
`H_beta = log((sum_i exp(beta * R_i)) / n) / beta` and assigns the
response-action signal `A_i = R_i - H_beta` on all valid tokens. At beta=0,
`H_0` is the group mean return. For binary returns this exactly reduces to
`H_beta = log(1 + p_g * (exp(beta) - 1)) / beta`, where `p_g` is the empirical
success rate. Graded, negative, and summed token rewards are also supported.
The implementation uses the existing tokenwise PPO/GRPO actor loss and does
not construct or train a critic.

```text
algorithm.adv_estimator=asharp_grpo
algorithm.asharp_beta=2.0
algorithm.norm_adv_by_std_in_grpo=True
algorithm.gamma=1.0
algorithm.use_kl_in_reward=False
actor_rollout_ref.rollout.n=4
critic.enable=False
```

`algorithm.norm_adv_by_std_in_grpo` controls the optional group standard
deviation scaling after the A-sharp signal is computed. With `True`, each
signal is divided by the group's return standard deviation plus `1e-6`; with
`False`, it is left unscaled. No additional group mean subtraction is applied:
that would remove the shared `H_beta` baseline and erase beta dependence.
At beta=0, this reproduces ordinary GRPO for groups of at least two responses,
with the same scaling switch.

The A-sharp **signal** is exact at the response-action level when the true
return log moment is known; the group log moment is its sample estimate. Its
plug-in gradient is nevertheless the
ordinary policy gradient in the population limit: the shared baseline
`H_beta(p)` cancels under the sequence score expectation. Reusing the same
finite group to estimate the baseline introduces sample coupling, so beta
effects in this implementation can reflect finite-group bias and PPO clipping.
The proposal's beta-dependent token-level direction requires prefix-conditional
success probabilities. A response-level fixed-target or exponential-weighted
actor objective would be a different training rule from the current GRPO loss.

### Binary RLVR with a PPO value critic

For a terminal reward in `{0, 1}`, the critic can predict the ordinary
success probability `p(s) = E[G | s]`. A-sharp then uses the exact Bernoulli
transform `H_beta(p) = log(1 + p * (exp(beta) - 1)) / beta`, with `H_0(p) = p`.
This keeps critic training labels, PPO value loss and checkpoint semantics
independent of beta. Only actor advantages change with beta. Predictions are
clamped to `[0, 1]` for this transform; critic regression uses the raw head.

Use these overrides with the same PPO run settings at each beta:

```text
algorithm.adv_estimator=asharp
algorithm.asharp_binary_value_critic=True
algorithm.asharp_beta=2.0
algorithm.gamma=1.0
algorithm.use_kl_in_reward=False
critic.value_type=value
```

`algorithm.lam_critic` controls ordinary GAE targets; `algorithm.lam_policy`
and `algorithm.asharp_actor_estimator` control the actor trace as below. The
mode checks that nonzero rewards occur only at the final response token and
are exactly 1. Binary-value critic checkpoints can be reused across beta
values; entropic-value checkpoints cannot be loaded as binary-value heads.

### Entropic-value critic

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
