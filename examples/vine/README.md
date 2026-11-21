# Vine training scaffold

This branch extends the standard Ray PPO actor update with Monte Carlo
continuations from selected states in each generated response. The selector is
deliberately an interface for now: implement
`select_mc_points` in `verl/trainer/ppo/vine_algos.py` before launching a run.
Until then, training raises `NotImplementedError` at the first batch.

## Configuration

Use a single-turn, text-only PPO run with a terminal task reward that scores
the full response. Add these overrides:

```text
algorithm.adv_estimator=vine
algorithm.vine_advantage_estimator=mc
algorithm.vine_mc_samples=4
algorithm.vine_max_points=8
algorithm.gamma=1.0
algorithm.use_kl_in_reward=False
critic.enable=False
```

Set `algorithm.vine_advantage_estimator=asharp` and `algorithm.asharp_beta=2.0`
to use the A-sharp log-moment value and its local segment residual instead.

`vine_mc_samples` is the number of fresh completions per selected prefix.
`vine_max_points` limits states per original response, including the prompt
state at offset zero. The selector returns token offsets after which a new
completion can begin; offset zero is inserted automatically. Terminal states
are excluded.

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
