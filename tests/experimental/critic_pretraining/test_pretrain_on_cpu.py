import numpy as np
import pytest
import torch

from verl.experimental.critic_pretraining.evaluate import binary_auc, compare_gae_lambdas, regression_metrics
from verl.experimental.critic_pretraining.pretrain import (
    CriticCollator,
    TrainingProgress,
    explained_variance_from_statistics,
    explained_variance_statistics,
    monte_carlo_targets,
)


def test_monte_carlo_targets_propagate_terminal_reward():
    assert monte_carlo_targets(2.0, 3, gamma=1.0) == [2.0, 2.0, 2.0]
    assert monte_carlo_targets(2.0, 3, gamma=0.5) == [0.5, 1.0, 2.0]


def test_collator_aligns_values_to_preceding_response_positions():
    batch = CriticCollator(pad_token_id=0, max_seq_length=16, gamma=1.0)(
        [
            {"prompt_ids": [10, 11], "response_ids": [20, 21, 22], "reward": 1.0},
            {"prompt_ids": [12, 13, 14], "response_ids": [23], "reward": -1.0},
        ]
    )

    assert batch["input_ids"].tolist() == [[10, 11, 20, 21, 22], [12, 13, 14, 23, 0]]
    assert batch["value_mask"].tolist() == [
        [False, True, True, True, False],
        [False, False, True, False, False],
    ]
    assert batch["value_targets"][0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert batch["value_targets"][1].tolist() == [0.0, 0.0, -1.0, 0.0, 0.0]


def test_explained_variance_is_one_for_perfect_predictions():
    targets = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    statistics = explained_variance_statistics(targets, targets, mask)
    assert explained_variance_from_statistics(statistics) == 1.0


def test_training_progress_round_trip():
    source = TrainingProgress()
    source.completed_steps = 7
    restored = TrainingProgress()
    restored.load_state_dict(source.state_dict())
    assert restored.completed_steps == 7


def test_holdout_regression_metrics_report_bias_and_auc():
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    targets = np.array([0.0, 0.0, 1.0, 1.0])

    metrics = regression_metrics(predictions, targets)

    assert metrics["count"] == 4
    assert metrics["auc"] == 1.0
    assert abs(metrics["bias_prediction_minus_target"]) < 1e-12
    assert metrics["mse"] == pytest.approx(0.025)


def test_fixed_rollout_lambda_comparison_uses_terminal_reward_and_reports_credit_gap():
    values = [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    result = compare_gae_lambdas(values, [1.0, 0.0], [0.5, 1.0], window=2)

    assert result["1.0"]["credit_gap/early"] > result["0.5"]["credit_gap/early"]
    assert result["1.0"]["credit_gap/early"] == pytest.approx(result["1.0"]["credit_gap/late"])
