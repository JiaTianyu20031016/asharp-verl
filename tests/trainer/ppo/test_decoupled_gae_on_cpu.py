import pytest
import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage


def _batch():
    return DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            "values": torch.tensor([[0.2, 0.4, 0.6], [-0.1, -0.3, -0.5]]),
            "response_mask": torch.ones(2, 3),
        }
    )


def test_decoupled_gae_uses_policy_lambda_for_advantages_and_critic_lambda_for_returns():
    source = _batch()
    expected_advantages, _ = core_algos.compute_gae_advantage_return(
        source.batch["token_level_rewards"],
        source.batch["values"],
        source.batch["response_mask"],
        gamma=1.0,
        lam=0.5,
    )
    _, expected_returns = core_algos.compute_gae_advantage_return(
        source.batch["token_level_rewards"],
        source.batch["values"],
        source.batch["response_mask"],
        gamma=1.0,
        lam=1.0,
    )

    result = compute_advantage(
        source,
        adv_estimator=core_algos.AdvantageEstimator.GAE,
        gamma=1.0,
        lam=0.95,
        lam_policy=0.5,
        lam_critic=1.0,
        config={},
    )

    torch.testing.assert_close(result.batch["advantages"], expected_advantages)
    torch.testing.assert_close(result.batch["returns"], expected_returns)


def test_decoupled_gae_defaults_preserve_legacy_lambda():
    source = _batch()
    expected_advantages, expected_returns = core_algos.compute_gae_advantage_return(
        source.batch["token_level_rewards"],
        source.batch["values"],
        source.batch["response_mask"],
        gamma=1.0,
        lam=0.7,
    )

    result = compute_advantage(
        source,
        adv_estimator=core_algos.AdvantageEstimator.GAE,
        gamma=1.0,
        lam=0.7,
        config={},
    )

    torch.testing.assert_close(result.batch["advantages"], expected_advantages)
    torch.testing.assert_close(result.batch["returns"], expected_returns)


def test_decoupled_gae_rejects_invalid_lambda():
    with pytest.raises(ValueError, match="lam_policy"):
        compute_advantage(
            _batch(),
            adv_estimator=core_algos.AdvantageEstimator.GAE,
            lam_policy=1.1,
            config={},
        )
