# MATH critic-only full warm-up (2026-10-09)

## Question and setup

This run froze the actor for the entire experiment and updated only a rank-32
LoRA critic. It used the same MATH policy, prompt format, terminal reward,
96 prompts × 4 responses per logical batch, three GPUs, one critic update per
batch, critic lambda 1.0, critic learning rate `1e-5`, 10 linear warm-up steps,
and 10 dataset epochs (770 optimizer steps). The actor received zero updates.
The run completed normally in 6 h 29 min.

The purpose was to distinguish two hypotheses:

1. the randomly initialized value head cannot learn a useful critic under the
   current settings; or
2. the critic can learn, but terminal-reward prediction—especially near the
   start of a long reasoning trace—remains intrinsically difficult and cannot
   repair actor credit assignment with `lambda_policy=0.95` by itself.

## Online training result

The first hypothesis is rejected. Mean online explained variance (EV) improved
from -9.26 in epoch 0 to 0.32 in epoch 3 and 0.46 in epoch 9. Mean value loss
fell from 1.19 to 0.052. The 20-step moving average first crossed EV 0.0, 0.2,
0.3, and 0.4 at steps 208, 228, 260, and 365 respectively. It never crossed
0.5. The final 50-step mean EV was 0.472, with its best value 0.473 at step 759.

This also shows that 10 learning-rate warm-up steps do not provide a meaningful
"critic warm-up" from random initialization: useful value estimates took about
200 optimizer steps to emerge. The optimizer warm-up itself was not unstable,
but it is far shorter than the learning phase needed by the value head.

## Fixed-policy holdout result

Each retained critic was evaluated on exactly the same 693 prompts × 4 frozen
base-policy responses (1,501,823 response tokens). This removes rollout sampling
noise from checkpoint comparisons.

| Step | Approx. time | All-token EV | MSE | Bias (pred-target) | AUC |
|---:|---:|---:|---:|---:|---:|
| 462 | 3.9 h | 0.3650 | 0.1656 | +0.1006 | 0.8569 |
| 539 | 4.5 h | 0.3729 | 0.1658 | +0.1105 | 0.8592 |
| 616 | 5.2 h | 0.3759 | 0.1627 | +0.0993 | 0.8605 |
| 693 | 5.8 h | 0.3856 | **0.1570** | **+0.0810** | 0.8645 |
| 770 | 6.5 h | **0.3895** | 0.1803 | +0.1755 | **0.8664** |

Discrimination continued to improve slowly through step 770, but numerical
calibration deteriorated sharply during the final epoch. Step 693 is therefore
the practical stopping point under the present constant learning rate: step 770
adds only 0.004 EV and 0.002 AUC while worsening MSE by 0.023 and more than
doubling the positive bias.

The value function is not constant across states. At step 693, trajectory-end
EV/AUC were 0.419/0.884, while trajectory-start EV/AUC were only 0.285/0.811.
At step 770 those were 0.427/0.884 versus 0.272/0.800. The critic distinguishes
successful from unsuccessful trajectories well near the answer, but has much
less information near the beginning. This directly explains why early-token
advantages remain much less separated than late-token advantages even after a
long critic warm-up.

An exact follow-up applied VERL's batched GAE and masked whitening to all 2,772
fixed trajectories using the same step-693 values. The correct-minus-incorrect
advantage gap in the first 100 response tokens was 0.033 at lambda 0.95, 0.200
at lambda 0.99, and 1.698 at lambda 1.0. At lambda 1.0, 94.5% of early tokens
on correct trajectories had positive advantage, compared with 52.1% at lambda
0.95. Thus the weak early credit is not primarily caused by a constant critic:
even this warmed critic cannot prevent lambda 0.95 from attenuating the terminal
reward over long responses. The exact results are stored in
`holdout/step693/advantage_by_lambda.json`.

## Interpretation

- Random initialization is a real cold-start cost, but not a fatal optimizer or
  implementation failure. The current rank-32 critic can reach moderate,
  generalizing EV.
- A critic-only warm-up of roughly 600–700 steps is useful; training indefinitely
  at constant `1e-5` is not. Holdout calibration should govern early stopping.
- Better critic training does not solve the main actor issue by itself. With
  `lambda_policy=0.95`, terminal reward influence still decays strongly toward
  the front of 500-token responses. Early states are also genuinely harder for
  the critic, so bootstrapping from them cannot replace direct reward propagation.
- The next PPO experiment should be a targeted hypothesis test, not a grid:
  initialize from a warmed critic near step 693, use `lambda_policy=1.0`, keep
  the other requested PPO settings fixed, and compare fixed-policy sampled
  pass@1 plus greedy accuracy. A late-stage critic learning-rate decay or lower
  post-warm-up LR is justified, but should not be changed in the same first test.

## Artifacts

- Training log: `/data1/jty/logs/ppo_lora32_math/qwen25-math-15b-math-critic-only-fullwarmup-lora32-n4-seed42.log`
- Online summaries and plots: `/data1/jty/analysis/ppo_lora32_math/critic_only_fullwarmup/`
- Fixed-holdout checkpoint metrics and plots: `/data1/jty/analysis/ppo_lora32_math/critic_only_fullwarmup/holdout_comparison/`
- Lightweight critic adapters: `/data1/jty/analysis/ppo_lora32_math/critic_only_fullwarmup/adapters/`
- Final recoverable full checkpoint: `/data1/jty/outputs/ppo_lora32_math/qwen25-math-15b-math-critic-only-fullwarmup-lora32-n4-seed42/global_step_770/`
