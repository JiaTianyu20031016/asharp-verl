"""Checks for Vine's Monte Carlo plumbing without rollout workers."""

import asyncio
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.vine_continuation_agent_loop import VineContinuationAgentLoop
from verl.trainer.ppo.utils import need_critic
from verl.trainer.ppo.vine_algos import (
    compute_vine_advantages,
    estimate_vine_value,
    mc_entropic_value,
    require_terminal_rewards,
    validate_mc_points,
)
from verl.trainer.ppo.vine_trainer import RayVinePPOTrainer, validate_vine_config
from verl.workers.rollout.replica import TokenOutput


def test_mc_entropic_value_and_segment_residuals():
    scores = torch.tensor([0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(mc_entropic_value(scores, 0), scores.mean())
    expected = (torch.logsumexp(2 * scores, 0) - math.log(2)) / 2
    torch.testing.assert_close(mc_entropic_value(scores, 2), expected)
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.ones_like(rewards)
    values = [[torch.tensor(0.2), torch.tensor(0.8)]]
    advantages = compute_vine_advantages(rewards, mask, [[0, 2]], values)
    torch.testing.assert_close(advantages, torch.tensor([[0.6, 0.6, 0.2]]))
    torch.testing.assert_close(estimate_vine_value(scores, "mc", beta=float("nan")), scores.mean())
    torch.testing.assert_close(estimate_vine_value(scores, "asharp", beta=2), expected)


def test_point_contract_and_validation():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert validate_mc_points([[1], [2, 0]], mask, 2) == [[0, 1], [0, 2]]
    with pytest.raises(ValueError, match="terminal"):
        validate_mc_points([[2], []], mask, 2)
    with pytest.raises(ValueError, match="exceeding"):
        validate_mc_points([[1], [1, 2]], mask, 2)
    with pytest.raises(ValueError, match="terminal-only"):
        require_terminal_rewards(torch.tensor([[0.25, 1.0]]), torch.ones(1, 2))


def test_vine_config_requires_critic_free_unshaped_rewards():
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "vine",
                "vine_advantage_estimator": "mc",
                "asharp_beta": 2.0,
                "vine_mc_samples": 2,
                "vine_max_points": 3,
                "gamma": 1.0,
                "use_kl_in_reward": False,
            },
            "critic": {"enable": False},
            "actor_rollout_ref": {"rollout": {"n": 1, "multi_turn": {"enable": False}}},
        }
    )
    validate_vine_config(cfg)
    cfg.critic.enable = True
    with pytest.raises(ValueError, match="does not use a critic"):
        validate_vine_config(cfg)


def test_full_config_selects_critic_free_vine():
    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=["algorithm.adv_estimator=vine", "critic.enable=False"],
        )
    assert need_critic(config) is False


def test_auxiliary_sampling_and_reordered_advantages(monkeypatch):
    monkeypatch.setattr("verl.trainer.ppo.vine_trainer.select_mc_points", lambda *_: [[1], [2]])

    class FakeManager:
        def generate_sequences(self, prompts):
            assert len(prompts) == 8
            assert list(prompts.non_tensor_batch["vine_prefix_ids"]) == [[], [], [1], [1], [], [], [3, 4], [3, 4]]
            assert set(prompts.non_tensor_batch["agent_name"]) == {"vine_continuation_agent"}
            scores = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            return DataProto.from_single_dict({"rm_scores": scores[:, None], "response_mask": torch.ones(8, 1)})

    trainer = object.__new__(RayVinePPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "vine_advantage_estimator": "asharp",
                "vine_mc_samples": 2,
                "vine_max_points": 2,
                "asharp_beta": 2.0,
            },
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )
    trainer.tokenizer = None
    trainer.async_rollout_manager = FakeManager()
    trainer.use_rm = False
    raw = np.empty(2, dtype=object)
    raw[:] = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    gen_batch = DataProto.from_single_dict({"prompts": torch.zeros(2, 2, dtype=torch.long), "raw_prompt": raw})
    generated = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
    )
    auxiliary = trainer._generate_auxiliary_rollouts(gen_batch, generated)
    assert generated.non_tensor_batch["vine_origin_id"].tolist() == [0, 1]
    batch = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "vine_origin_id": np.array([1, 0]),
        }
    )
    actual = trainer._compute_auxiliary_advantages(batch, auxiliary)
    h = (math.log1p(math.exp(2.0)) - math.log(2.0)) / 2.0
    torch.testing.assert_close(actual, torch.tensor([[h, h, 1 - h], [1 - h, 0.0, 0.0]]))

    trainer.config.algorithm.vine_advantage_estimator = "mc"
    batch.non_tensor_batch["vine_origin_id"] = np.array([1, 0])
    mc_actual = trainer._compute_auxiliary_advantages(batch, auxiliary)
    torch.testing.assert_close(mc_actual, torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]))


def test_continuation_agent_samples_from_exact_prefix():
    class FakeServer:
        async def generate(self, *, prompt_ids, sampling_params, **kwargs):
            assert prompt_ids == [10, 11, 21, 22]
            assert sampling_params["max_tokens"] == 3
            return TokenOutput(token_ids=[30, 31], extra_fields={})

    agent = object.__new__(VineContinuationAgentLoop)
    agent.response_length = 5
    agent.server_manager = FakeServer()

    async def apply_template(messages):
        return [10, 11]

    async def process_vision(messages):
        return {}

    agent.apply_chat_template = apply_template
    agent.process_vision_info = process_vision
    result = asyncio.run(
        agent.run({}, raw_prompt=[{"role": "user", "content": "question"}], vine_prefix_ids=[21, 22])
    )
    assert result.prompt_ids == [10, 11]
    assert result.response_ids == [21, 22, 30, 31]
    assert result.response_mask == [1, 1, 1, 1]
