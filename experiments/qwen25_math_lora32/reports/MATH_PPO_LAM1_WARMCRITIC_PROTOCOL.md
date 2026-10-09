# MATH PPO lambda-1 + warmed-critic protocol (2026-10-09)

## Hypothesis

The previous PPO runs failed mainly because `lambda_policy=0.95` attenuated a
terminal reward over roughly 500 response tokens. A useful critic did not fix
that attenuation near the beginning of a response. On 2,772 fixed trajectories
with the step-693 critic, changing only GAE lambda from 0.95 to 1.0 increased
the whitened first-100-token correct-minus-incorrect advantage gap from 0.033
to 1.698. This run tests whether that stronger full-trajectory credit produces
an actual policy improvement.

## Run definition

- Actor: base Qwen2.5-Math-1.5B-Instruct rank-32 LoRA.
- Critic: rank-32 LoRA initialized from critic-only step 693.
- Logical batch: 96 prompts × 4 sampled responses; unchanged on 3 or 4 GPUs.
- `lambda_policy=1.0`, `lambda_critic=1.0`, `gamma=1.0`.
- Actor/critic LR: `5e-6` / `1e-5`, with 10 optimizer warm-up steps.
- One actor and one critic PPO epoch per rollout batch; symmetric clip 0.2.
- No KL reward/loss, no format reward, terminal correctness reward only.
- Terminal correctness is computed by extracting the last complete `\\boxed{}`
  answer, preserving strict-normalization positives, and using Math-Verify 0.9.0
  as an equivalence fallback. There is no positive partial format score.
- Ten dataset epochs (770 updates), with validation and checkpoint every 77.

The run changes the actor lambda and critic initialization relative to the old
baseline. The fixed-rollout diagnostic already isolates lambda on the same
critic; this training run tests the combined remedy rather than estimating a
separate causal effect for each change.

## Predeclared interpretation

The old no-warm-up run's sampled pass@1 after training had mean 0.6634,
standard deviation 0.0039, and maximum 0.6696 across six validations; it did
not improve relative to its 0.6627 step-0 value. The partial 232-step warm-up
run was likewise flat.

Those historical numbers used the old strict scorer and are therefore not an
apples-to-apples numerical baseline for the corrected run. The corrected run's
own step-0 sampled pass@1 is 0.6580. Because validation uses fresh temperature-1
samples, a single checkpoint is not treated as proof. Evidence of effective
training requires either:

1. at least two consecutive scheduled validations at least 1 percentage point
   above this run's step-0 pass@1; or
2. a fixed-policy checkpoint evaluation with a paired improvement over the
   base actor whose uncertainty excludes zero.

Early termination for lack of progress is considered only after at least four
actor epochs (step 308), and only if the best validation gain remains below
0.5 percentage points while the last three validations and training reward
show no upward trend. A degradation or instability can justify stopping sooner.
If improvement occurs, training continues until validation has failed to set a
new best for three consecutive scheduled evaluations.

## Live artifacts

- Log: `/data1/jty/logs/ppo_lora32_math/qwen25-math-15b-math-ppo-lora32-lam1-warmcritic693-n4-seed42.log`
- Checkpoints: `/data1/jty/outputs/ppo_lora32_math/qwen25-math-15b-math-ppo-lora32-lam1-warmcritic693-n4-seed42/`

An initial launch was stopped at step 8 because it still used VERL's strict
string scorer. Its archived log ends in
`.aborted-strict-scorer-step8.log`; no checkpoint was retained. On the 2,772
saved base-policy validation responses, the corrected scorer preserved every
strict positive and recovered 69 additional boxed answers (2.49 percentage
points), taking 1.29 ms per response in a sequential benchmark.
