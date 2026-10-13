# Continuous A-sharp tree audit

This experiment exactly enumerates eight trajectories through a depth-three,
seven-state binary decision tree with graded terminal rewards from -1 to 4.
Seven independent Bernoulli logits parameterize the actor. The critic either
uses exact entropic values or those values plus a fixed, state-consistent error
pattern scaled by `noise_scale`. No model training or Monte Carlo sampling is
involved in these measurements.

The script sweeps beta in {0, 0.5, 1}, actor lambda in {0, 0.8, 1}, and critic
error scale in {0, 0.1, 0.3}. The table below selects beta=1, actor lambda=0.8.
Bias, variance and MSE are conditional on root action 0. Gradient error is the
L2 distance between expected estimated and exact A-sharp gradients for **all
seven logits**, using raw score-function signals before PPO clipping/whitening.

| Critic error scale | Estimator | Bias | Variance | MSE | Actor gradient error |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.0 | one_step | 0 | 1.9721523e-31 | 0 | 0 |
| 0.0 | log_plugin | -0.19139088 | 0.35377298 | 0.39040345 | 0.0096873089 |
| 0.0 | residual_gae | -1.9428903e-16 | 0.47499241 | 0.47499241 | 1.1775693e-16 |
| 0.0 | residual_exp | 5.2041704e-17 | 0.0039761331 | 0.0039761331 | 1.1775693e-16 |
| 0.1 | one_step | -0.09 | 1.9721523e-31 | 0.0081 | 0.037147416 |
| 0.1 | log_plugin | -0.22218462 | 0.35205153 | 0.40141753 | 0.012709199 |
| 0.1 | residual_gae | -0.0570068 | 0.49625387 | 0.49950365 | 0.0062810096 |
| 0.1 | residual_exp | -0.084776109 | 0.0037136894 | 0.010900678 | 0.0064558448 |
| 0.3 | one_step | -0.27 | 0 | 0.0729 | 0.11144225 |
| 0.3 | log_plugin | -0.27651962 | 0.34403901 | 0.42050211 | 0.023482803 |
| 0.3 | residual_gae | -0.12612868 | 0.51993698 | 0.53584543 | 0.02083859 |
| 0.3 | residual_exp | -0.25568997 | 0.0032313623 | 0.068608722 | 0.021410064 |

With exact critics, log_plugin exhibits negative conditional Jensen bias while
both residual constructions preserve the conditional mean to floating-point
precision. All three become identical to GAE at beta=0 and to one_step at actor
lambda=0, as separately checked in unit tests.

With critic errors, residual constructions are no longer strictly unbiased.
This fixture shows a bias/variance tradeoff: the estimator with the smaller
conditional bias need not have the smaller conditional MSE. Its particular
variance ordering is not a general guarantee; residual_exp propagation can
amplify variance on other trees.

Reproduce the complete machine-readable sweep:

```bash
python -m examples.asharp_trainer.compare_estimators --output /tmp/asharp-tree.json
```

Full GPU/LLM training has not been run for this change.
