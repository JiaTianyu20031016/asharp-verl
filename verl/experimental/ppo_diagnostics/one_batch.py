"""Read-only diagnostic using one already-generated PPO batch."""

from __future__ import annotations

from verl.trainer.ppo import core_algos
from verl.experimental.ppo_diagnostics.advantage import summarize_position_advantages


def collect_one_batch_diagnostic(batch, gamma: float, metrics: dict) -> dict[str, float]:
    rewards = batch.batch["token_level_rewards"]
    scores = batch.batch["token_level_scores"]
    mask = batch.batch["response_mask"]
    values = batch.batch["values"]
    diagnostic = {
        "score_mean": scores.sum(-1).float().mean().item(),
        "response_length_mean": mask.sum(-1).float().mean().item(),
    }
    for policy_lam in (0.95, 0.99, 1.0):
        advantages, _ = core_algos.compute_gae_advantage_return(
            token_level_rewards=rewards, values=values,
            response_mask=mask, gamma=gamma, lam=policy_lam,
        )
        for name, result in summarize_position_advantages(advantages, mask, scores).items():
            diagnostic[f"lambda_{policy_lam}/{name}"] = result
    for name in (
        "training/rollout_probs_diff_valid",
        "training/rollout_probs_diff_mean",
        "training/rollout_probs_diff_max",
        "training/rollout_actor_probs_pearson_corr",
    ):
        if name in metrics:
            diagnostic[name] = float(metrics[name])
    return diagnostic
