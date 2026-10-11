# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analytic checks for entropic targets, critic objectives, and trainer wiring."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.asharp import (
    check_asharp_checkpoint,
    compute_asharp,
    compute_log_moment_loss,
    save_asharp_checkpoint,
    validate_asharp_config,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.utils import need_critic


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_lambda_return_matches_explicit_n_step_mixture(lam):
    beta = 0.7
    rewards = torch.tensor([[0.2, -0.6, 1.4], [-1.0, 2.0, float("nan")]])
    values = torch.tensor([[0.4, -0.3, 0.7], [0.1, 0.9, float("nan")]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    adv, targets = compute_asharp(rewards, values, mask, beta, lam)
    for row, length in enumerate([3, 2]):
        for t in range(length):
            expected = 0.0
            for n in range(1, length - t + 1):
                terminal = t + n == length
                bootstrap = 1.0 if terminal else values[row, t + n].exp().item()
                weight = lam ** (n - 1) * (1 if terminal else 1 - lam)
                expected += weight * math.exp(beta * rewards[row, t : t + n].sum().item()) * bootstrap
            assert targets[row, t].exp().item() == pytest.approx(expected, rel=1e-6)
            next_value = 0 if t + 1 == length else values[row, t + 1].item()
            assert adv[row, t].item() == pytest.approx(
                rewards[row, t].item() + (next_value - values[row, t].item()) / beta, abs=1e-6
            )
    assert adv[1, 2] == targets[1, 2] == 0


def test_exact_graded_tree_target_ratio():
    # Root chooses either branch; branch A has returns -1 or 2, B returns 0.5.
    beta = 0.8
    branch_a = 0.25 * math.exp(-beta) + 0.75 * math.exp(2 * beta)
    branch_b = math.exp(0.5 * beta)
    root = 0.4 * branch_a + 0.6 * branch_b
    rewards = torch.tensor([[0.0, -1.0], [0.0, 2.0], [0.0, 0.5]])
    values = torch.tensor(
        [
            [math.log(root), math.log(branch_a)],
            [math.log(root), math.log(branch_a)],
            [math.log(root), math.log(branch_b)],
        ]
    )
    adv, _ = compute_asharp(rewards, values, torch.ones_like(values), beta, 0.7)
    ratios = (beta * adv[:, 0]).exp()
    torch.testing.assert_close(ratios, torch.tensor([branch_a / root, branch_a / root, branch_b / root]))
    assert (0.4 * ratios[0] + 0.6 * ratios[2]).item() == pytest.approx(1.0)


def test_binary_specialization_and_small_beta_limit():
    p = torch.tensor([[0.3, 0.6]], dtype=torch.float64)
    beta = 1e-4
    values = torch.log1p(math.expm1(beta) * p)
    adv, _ = compute_asharp(torch.tensor([[0.0, 1.0]]), values, torch.ones_like(p), beta, 0.9)
    torch.testing.assert_close(adv, torch.tensor([[0.3, 0.4]]), atol=2e-5, rtol=0)


@pytest.mark.parametrize("mode", ["mse", "positive"])
def test_loss_learns_arithmetic_mean_not_mean_log(mode):
    # Two outcomes at an identical state: E[Y]=2.5, exp(E[log Y])=2.
    mean_log = torch.tensor(math.log(2.5), requires_grad=True)
    log_targets = torch.tensor([[0.0, math.log(4.0)]], requires_grad=True)
    mask = torch.ones_like(log_targets)
    loss, _ = compute_log_moment_loss(mean_log.expand_as(log_targets), log_targets, mask, mode)
    loss.backward()
    assert abs(mean_log.grad.item()) < 1e-6
    assert log_targets.grad is None
    wrong = torch.tensor(math.log(2.0), requires_grad=True)
    loss, _ = compute_log_moment_loss(wrong.expand_as(log_targets), log_targets, mask, mode)
    loss.backward()
    assert wrong.grad.item() < 0


def test_default_loss_is_moment_mse_and_padding_has_no_gradient():
    pred = torch.tensor([[math.log(2.0), float("nan")]], requires_grad=True)
    loss, _ = compute_log_moment_loss(pred, torch.tensor([[0.0, float("nan")]]), torch.tensor([[1.0, 0.0]]))
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[2.0, 0.0]]))


def test_overflow_is_explicit():
    with pytest.raises(FloatingPointError):
        compute_log_moment_loss(torch.tensor([[1000.0]]), torch.zeros(1, 1), torch.ones(1, 1))


def config():
    return OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "asharp",
                "asharp_beta": 0.7,
                "asharp_lam": 0.4,
                "gamma": 1.0,
                "use_kl_in_reward": False,
            },
            "critic": {"enable": None, "strategy": "fsdp2", "value_type": "log_moment", "log_moment_loss": "mse"},
            "actor_rollout_ref": {"rollout": {"multi_turn": {"enable": False}}},
        }
    )


def test_registry_dispatch_and_post_update_refresh_preserve_targets():
    cfg = config()
    batch = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.tensor([[0.2, 1.0]]),
            "values": torch.tensor([[0.1, 0.2]]),
            "response_mask": torch.ones(1, 2),
        }
    )
    batch = compute_advantage(batch, "asharp", config=cfg.algorithm)
    old_targets = batch.batch["returns"].clone()
    old_adv = batch.batch["advantages"].clone()
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = cfg
    trainer._compute_values = lambda _: DataProto.from_single_dict({"values": torch.tensor([[0.3, 0.9]])})
    trainer._refresh_asharp_advantages(batch)
    torch.testing.assert_close(batch.batch["returns"], old_targets)
    assert not torch.allclose(old_adv, batch.batch["advantages"])
    assert not batch.batch["advantages"].requires_grad
    assert need_critic(cfg)


@pytest.mark.parametrize(
    "path,value",
    [
        ("algorithm.asharp_beta", 0),
        ("algorithm.asharp_beta", float("nan")),
        ("algorithm.asharp_lam", 1.1),
        ("algorithm.gamma", 0.99),
        ("algorithm.use_kl_in_reward", True),
        ("critic.enable", False),
        ("critic.value_type", "value"),
        ("critic.strategy", "megatron"),
        ("critic.log_moment_loss", "log_mse"),
        ("actor_rollout_ref.rollout.multi_turn.enable", True),
    ],
)
def test_invalid_configuration(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        validate_asharp_config(cfg)


def test_checkpoint_temperature_and_semantics(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        check_asharp_checkpoint(tmp_path, 1.0)
    save_asharp_checkpoint(tmp_path, 1.0)
    check_asharp_checkpoint(tmp_path, 1.0)
    with pytest.raises(ValueError, match="beta"):
        check_asharp_checkpoint(tmp_path, 2.0)


def test_internal_mask_holes_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        compute_asharp(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([[1, 0, 1]]), 1.0, 1.0)


@pytest.mark.parametrize("mode", ["mse", "positive"])
def test_engine_critic_uses_moment_loss(monkeypatch, mode):
    from verl.workers.utils import losses

    monkeypatch.setattr(losses, "_slice_response_from_unpad_output", lambda output, data: output)
    cfg = OmegaConf.create({"value_type": "log_moment", "log_moment_loss": mode, "loss_agg_mode": "token-mean"})
    pred = torch.tensor([[0.1, 0.2]], requires_grad=True)
    data = {
        "values": torch.zeros_like(pred),
        "returns": torch.tensor([[0.4, 0.6]]),
        "response_mask": torch.ones_like(pred),
    }
    expected, _ = compute_log_moment_loss(pred, data["returns"], data["response_mask"], mode)
    actual, metrics = losses.value_loss(cfg, {"values": pred}, data)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
    assert metrics["critic/vf_clipfrac"] == 0


@pytest.mark.parametrize("refresh", [False, True])
def test_composed_hydra_config_supports_asharp(refresh):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl.trainer.config.algorithm import AlgoConfig
    from verl.utils.config import omega_conf_to_dataclass

    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=asharp",
                "critic.value_type=log_moment",
                "algorithm.asharp_beta=0.4",
                "algorithm.asharp_lam=0.9",
            ]
            + (["algorithm.asharp_refresh_after_critic_update=True"] if refresh else []),
        )
    validate_asharp_config(cfg)
    algo = omega_conf_to_dataclass(cfg.algorithm, AlgoConfig)
    assert algo.asharp_beta == 0.4
    assert algo.asharp_lam == 0.9
    assert algo.asharp_refresh_after_critic_update is refresh
    assert cfg.critic.log_moment_loss == "mse"


@pytest.mark.parametrize("mode", ["mse", "positive"])
def test_legacy_critic_optimizer_consumes_log_targets(monkeypatch, mode):
    from verl.workers.critic import dp_critic

    monkeypatch.setattr(dp_critic, "get_device_id", lambda: "cpu")
    critic = dp_critic.DataParallelPPOCritic.__new__(dp_critic.DataParallelPPOCritic)
    critic.config = OmegaConf.create(
        {
            "value_type": "log_moment",
            "log_moment_loss": mode,
            "loss_agg_mode": "token-mean",
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "use_dynamic_bsz": False,
        }
    )
    critic.critic_module = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(critic.critic_module.weight)
    critic.critic_optimizer = torch.optim.SGD(critic.critic_module.parameters(), lr=0.1)
    critic._forward_micro_batch = lambda data: critic.critic_module(torch.ones(1, 1)).expand_as(data["returns"])

    def step():
        norm = critic.critic_module.weight.grad.norm().detach()
        critic.critic_optimizer.step()
        return norm

    critic._optimizer_step = step
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3),
            "position_ids": torch.zeros(2, 3, dtype=torch.long),
            "responses": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2),
            "values": torch.zeros(2, 2),
            "returns": torch.full((2, 2), math.log(2.0)),
        }
    )
    critic.update_critic(batch)
    # Both losses at u=0 have dL/du=-1; accumulation must yield one SGD step.
    assert critic.critic_module.weight.item() == pytest.approx(0.1)
