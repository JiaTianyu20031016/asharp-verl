import torch

from verl import DataProto
from verl.experimental.ppo_diagnostics.one_batch import collect_one_batch_diagnostic


def test_one_batch_diagnostic_compares_lambdas_without_mutating_batch():
    data = DataProto.from_single_dict({
        "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "token_level_scores": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "values": torch.zeros(2, 3),
        "response_mask": torch.ones(2, 3),
    })
    original_rewards = data.batch["token_level_rewards"].clone()
    result = collect_one_batch_diagnostic(
        data, gamma=1.0, metrics={"training/rollout_probs_diff_mean": 0.01}
    )

    assert result["score_mean"] == 0.5
    assert result["response_length_mean"] == 3.0
    assert result["training/rollout_probs_diff_mean"] == 0.01
    assert "lambda_0.95/advantage_position/correct_early_mean" in result
    assert "lambda_0.99/advantage_position/correct_early_mean" in result
    assert "lambda_1.0/advantage_position/correct_early_mean" in result
    torch.testing.assert_close(data.batch["token_level_rewards"], original_rewards)
