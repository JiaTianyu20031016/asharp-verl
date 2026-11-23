# Vine training scaffold

This branch extends the standard Ray PPO actor update with Monte Carlo
continuations from selected states in each generated response. The first
selector in `verl/trainer/ppo/vine_algos.py` uses a blank line (`\n\n`) in the
generated response as a candidate continuation point.

## MATH prompts

The existing MATH parquet asks for step-by-step reasoning and a final answer
in `\\boxed{}`, without specifying a step separator. Create a Vine-specific
copy that adds an instruction to separate steps with exactly one blank line:

```bash
python examples/vine/prepare_math_prompts.py \
  /data1/jty/data/math_lighteval_boxed \
  /data1/jty/data/math_lighteval_boxed_vine
```

Set `data.train_files` and `data.val_files` to the new parquet files. The
selector recognizes `\n\n` even when the two newline characters are in
different tokens. This is still a formatting heuristic: a response that
ignores the instruction has fewer or no internal sampling points.

## Configuration

Use a single-turn, text-only PPO run with a terminal task reward that scores
the full response. Add these overrides:

```text
algorithm.adv_estimator=vine_mc
algorithm.vine_mc_samples=4
algorithm.vine_max_points=8
algorithm.gamma=1.0
algorithm.use_kl_in_reward=False
critic.enable=False
```

Set `algorithm.adv_estimator=vine_asharp` and `algorithm.asharp_beta=2.0`
to use the A-sharp log-moment value and its local segment residual instead.

`vine_mc_samples` is the number of fresh completions per selected prefix.
`vine_max_points` limits states per original response, including the prompt
state at offset zero. The selector returns token offsets after which a new
completion can begin; offset zero is inserted automatically. Terminal states
are excluded. When there are more blank lines than the budget allows, the
selector chooses evenly spaced boundaries deterministically; it does not
sample them randomly. With no usable blank line, only the prompt state is
sampled and the observed step count is one. The selector does not infer hidden
reasoning steps from an unformatted paragraph.

The training logger records `vine/steps/mean`, `p50`, and `p90` from the
original responses before the point budget is applied. It also records
`vine/points/selected_mean` (including the prompt state),
`vine/points/over_budget_rate`, and three formatting rates:

- `vine/format/no_separator_rate`: no usable `\n\n` separated two nonempty parts.
- `vine/format/single_newline_rate`: at least one isolated `\n` appeared.
- `vine/format/deviation_proxy_rate`: either of the above.

These formatting rates are observable proxies, not a semantic judgment about
whether the model followed the instruction. A valid one-step response can
have no separator, while a multi-step paragraph can hide steps from this
delimiter-based counter. The rates are logged once per training step for
trend plots.

For each selected state, the trainer samples continuations from the same
rollout policy snapshot as the original response. The continuation agent
appends the original token prefix to the model prompt, then presents the full
prefix plus sampled suffix as the response to the reward function. Auxiliary
responses are used only for Monte Carlo estimates; actor updates use the
original responses.

In the original VinePPO mode, the state estimate at each selected point is
the sample mean of continuation returns. The advantage of a reasoning segment
is its observed reward plus the next point's MC value minus the current
point's MC value, matching Equation 6 and the released implementation.
The terminal state has value zero. All tokens in a segment share its advantage.

In A-sharp mode, the same neighboring-point residual uses
`H_beta(s) = log(mean(exp(beta * R_k))) / beta` instead of the MC mean.
This finite-sample log-moment estimate is biased. Larger
`vine_mc_samples` reduces sampling error at additional rollout cost.
