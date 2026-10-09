import pytest
import torch

from verl.experimental.ppo_diagnostics.advantage import summarize_position_advantages


def test_position_advantages_respect_response_lengths_and_outcomes():
    advantages = torch.tensor([[1.0, -1.0, 2.0, 4.0], [-2.0, -4.0, 99.0, 99.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    scores = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    result = summarize_position_advantages(advantages, mask, scores, window=2)

    assert result["advantage_position/correct_early_mean"] == 0.0
    assert result["advantage_position/correct_late_mean"] == 3.0
    assert result["advantage_position/correct_early_positive_fraction"] == 0.5
    assert result["advantage_position/incorrect_early_mean"] == -3.0
    assert result["advantage_position/incorrect_late_mean"] == -3.0
    assert result["advantage_position/incorrect_late_tokens"] == 2


def test_position_advantages_reject_invalid_shapes():
    with pytest.raises(ValueError, match="same shape"):
        summarize_position_advantages(torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(2, 2))
