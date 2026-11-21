"""Checks for critic-free response-action A-sharp with GRPO groups."""

import math

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from verl import DataProto
from verl.trainer.ppo.asharp import compute_asharp_grpo, validate_asharp_grpo_config
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.utils import need_critic


def example_batch():
    rewards = torch.tensor([[0.0, 1.0, float("nan")], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    groups = np.array(["a", "a", "b", "b"])
    return rewards, mask, groups


@pytest.mark.parametrize("normalize", [False, True])
def test_beta_zero_matches_grpo(normalize):
    rewards, mask, groups = example_batch()
    actual, returns = compute_asharp_grpo(rewards, mask, groups, 0.0, normalize_by_std=normalize)
    expected, _ = compute_grpo_outcome_advantage(rewards.nan_to_num(), mask, groups, norm_adv_by_std_in_grpo=normalize)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(returns, actual)


@pytest.mark.parametrize("beta", [0.7, 4.0, 100.0])
def test_group_baseline_is_bernoulli_log_moment(beta):
    rewards, mask, groups = example_batch()
    actual, _ = compute_asharp_grpo(rewards, mask, groups, beta, normalize_by_std=False)
    baseline = math.log1p(math.expm1(beta) / 2) / beta
    expected = torch.tensor([[1 - baseline], [-baseline], [1 - baseline], [-baseline]]) * mask
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2] == 0


@pytest.mark.parametrize("beta", [0.0, 1e-8, 0.7, 4.0, 100.0])
def test_graded_and_dense_returns_use_group_log_mean_exp(beta):
    rewards = torch.tensor([[-0.5, -0.5, float("nan")], [0.2, 0.3, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]])
    returns = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    advantages, _ = compute_asharp_grpo(rewards, mask, np.array([0, 0, 0]), beta, normalize_by_std=False)
    if beta == 0:
        baseline = returns.mean()
    elif beta < 1e-6:
        centered = returns - returns.mean()
        baseline = returns.mean() + 0.5 * beta * centered.square().mean()
    else:
        baseline = torch.logsumexp(beta * returns, dim=0).sub(math.log(3)) / beta
    torch.testing.assert_close(advantages, (returns - baseline)[:, None] * mask, rtol=1e-10, atol=1e-10)


def test_identical_graded_returns_have_zero_signal_at_large_beta():
    rewards = torch.tensor([[0.25], [0.25]])
    actual, _ = compute_asharp_grpo(rewards, torch.ones_like(rewards), np.array([0, 0]), 1000.0)
    torch.testing.assert_close(actual, torch.zeros_like(rewards))


def test_group_normalization_preserves_beta_effect():
    rewards, mask, groups = example_batch()
    ordinary, _ = compute_asharp_grpo(rewards, mask, groups, 0.0)
    transformed, _ = compute_asharp_grpo(rewards, mask, groups, 2.0)
    assert not torch.allclose(ordinary, transformed)
    assert transformed[0, 0] > 0 > transformed[1, 0]


@pytest.mark.parametrize(
    "rewards,mask,groups,match",
    [
        (torch.tensor([[float("nan")], [1.0]]), torch.ones(2, 1), np.array([0, 0]), "Non-finite"),
        (torch.zeros(2, 3), torch.tensor([[1, 0, 1], [1, 1, 1]]), np.array([0, 0]), "contiguous"),
        (torch.zeros(2, 1), torch.ones(2, 1), np.array([0, 1]), "at least two"),
    ],
)
def test_invalid_group_data(rewards, mask, groups, match):
    with pytest.raises(ValueError, match=match):
        compute_asharp_grpo(rewards, mask, groups, 1.0)


@pytest.mark.parametrize("normalize", [False, True])
def test_trainer_dispatch_and_no_critic(normalize):
    from pathlib import Path

    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=asharp_grpo",
                "algorithm.asharp_beta=2.0",
                f"algorithm.norm_adv_by_std_in_grpo={normalize}",
                "actor_rollout_ref.rollout.n=2",
                "critic.enable=False",
            ],
        )
    validate_asharp_grpo_config(cfg)
    assert need_critic(cfg) is False
    rewards, mask, groups = example_batch()
    batch = DataProto.from_single_dict({"token_level_rewards": rewards, "response_mask": mask, "uid": groups})
    result = compute_advantage(batch, "asharp_grpo", config=cfg.algorithm)
    expected, _ = compute_asharp_grpo(rewards, mask, groups, 2.0, normalize_by_std=normalize)
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], expected)
    assert "values" not in result.batch
