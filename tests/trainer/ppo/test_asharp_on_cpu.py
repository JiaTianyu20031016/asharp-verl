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
                bootstrap = 1.0 if terminal else (beta * values[row, t + n]).exp().item()
                weight = lam ** (n - 1) * (1 if terminal else 1 - lam)
                expected += weight * math.exp(beta * rewards[row, t : t + n].sum().item()) * bootstrap
            assert (beta * targets[row, t]).exp().item() == pytest.approx(expected, rel=1e-6)
            next_value = 0 if t + 1 == length else values[row, t + 1].item()
            assert adv[row, t].item() == pytest.approx(
                rewards[row, t].item() + next_value - values[row, t].item(), abs=1e-6
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
    adv, _ = compute_asharp(rewards, values / beta, torch.ones_like(values), beta, 0.7)
    ratios = (beta * adv[:, 0]).exp()
    torch.testing.assert_close(ratios, torch.tensor([branch_a / root, branch_a / root, branch_b / root]))
    assert (0.4 * ratios[0] + 0.6 * ratios[2]).item() == pytest.approx(1.0)


def test_binary_specialization_and_small_beta_limit():
    p = torch.tensor([[0.3, 0.6]], dtype=torch.float64)
    beta = 1e-4
    values = torch.log1p(math.expm1(beta) * p) / beta
    adv, _ = compute_asharp(torch.tensor([[0.0, 1.0]]), values, torch.ones_like(p), beta, 0.9)
    torch.testing.assert_close(adv, torch.tensor([[0.3, 0.4]], dtype=torch.float64), atol=2e-5, rtol=0)


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
                "asharp_lam_critic": 0.4,
                "gamma": 1.0,
                "use_kl_in_reward": False,
            },
            "critic": {"enable": None, "strategy": "fsdp2", "value_type": "entropic_value", "log_moment_loss": "mse"},
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
        ("algorithm.asharp_beta", -0.1),
        ("algorithm.asharp_beta", float("nan")),
        ("algorithm.asharp_lam_critic", 1.1),
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
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("clip", [False, True])
def test_engine_critic_uses_moment_loss(monkeypatch, mode, beta, clip):
    from verl.workers.utils import losses

    monkeypatch.setattr(losses, "_slice_response_from_unpad_output", lambda output, data: output)
    cfg = OmegaConf.create(
        {
            "value_type": "entropic_value",
            "log_moment_loss": mode,
            "loss_agg_mode": "token-mean",
            "asharp_beta": beta,
            "log_moment_value_clip": clip,
            "cliprange_value": 0.05,
        }
    )
    pred = torch.tensor([[0.1, 0.2]], requires_grad=True)
    data = {
        "values": torch.zeros_like(pred),
        "returns": torch.tensor([[0.4, 0.6]]),
        "response_mask": torch.ones_like(pred),
    }
    expected, frac = compute_log_moment_loss(
        pred,
        data["returns"],
        data["response_mask"],
        mode,
        beta=beta,
        values=data["values"],
        cliprange_value=0.05 if clip else None,
    )
    actual, metrics = losses.value_loss(cfg, {"values": pred}, data)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
    assert metrics["critic/vf_clipfrac"] == frac.item()


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
                "critic.value_type=entropic_value",
                "algorithm.asharp_beta=0.4",
                "algorithm.asharp_lam_critic=0.9",
            ]
            + (["algorithm.asharp_refresh_after_critic_update=True"] if refresh else []),
        )
    validate_asharp_config(cfg)
    algo = omega_conf_to_dataclass(cfg.algorithm, AlgoConfig)
    assert algo.asharp_beta == 0.4
    assert algo.asharp_lam_critic == 0.9
    assert cfg.critic.asharp_beta == 0.4
    assert algo.asharp_refresh_after_critic_update is refresh
    assert cfg.critic.log_moment_loss == "mse"


@pytest.mark.parametrize("mode", ["mse", "positive"])
@pytest.mark.parametrize("beta", [0.0, 1e-12, 0.7, 1.0])
def test_legacy_critic_optimizer_consumes_H_targets(monkeypatch, mode, beta):
    from verl.workers.critic import dp_critic

    monkeypatch.setattr(dp_critic, "get_device_id", lambda: "cpu")
    critic = dp_critic.DataParallelPPOCritic.__new__(dp_critic.DataParallelPPOCritic)
    critic.config = OmegaConf.create(
        {
            "value_type": "entropic_value",
            "log_moment_loss": mode,
            "loss_agg_mode": "token-mean",
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "asharp_beta": beta,
            "log_moment_value_clip": False,
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
    # Both normalized losses at H=0 have the same gradient; check accumulation.
    expected = 0.1 * (math.log(2.0) if beta == 0 else math.expm1(beta * math.log(2.0)) / beta)
    assert critic.critic_module.weight.item() == pytest.approx(expected)


ESTIMATORS = ["log_plugin", "residual_gae", "residual_exp"]


def ordinary_gae(rewards, values, mask, lam):
    """Use verl's baseline to compare both raw and normalized valid-token signals."""
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return

    normalized, returns = compute_gae_advantage_return(rewards, values, mask, 1.0, lam)
    return (returns - values) * mask, normalized * mask, returns * mask


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("lam_actor,lam_critic", [(0.0, 0.95), (0.4, 1.0), (1.0, 0.0), (0.7, 0.3)])
def test_beta_zero_matches_ppo_gae_and_targets(estimator, lam_actor, lam_critic):
    rewards = torch.tensor([[0.2, -0.7, 1.3], [-0.5, 0.8, 0.0]], dtype=torch.float64)
    values = torch.tensor([[0.4, -0.3, 0.9], [0.7, 0.1, 0.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    raw, normalized, _ = ordinary_gae(rewards, values, mask, lam_actor)
    _, _, expected_targets = ordinary_gae(rewards, values, mask, lam_critic)
    for normalize in (False, True):
        adv, targets = compute_asharp(rewards, values, mask, 0.0, lam_critic, lam_actor, estimator, normalize)
        torch.testing.assert_close(adv, normalized if normalize else raw, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(targets, expected_targets, atol=0, rtol=0)


@pytest.mark.parametrize("beta", [0.0, 1e-12, 0.7, 3.0])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_actor_lambda_zero_is_exact_one_step(beta, estimator):
    rewards = torch.tensor([[0.2, -0.5, 1.0]])
    values = torch.tensor([[0.3, -0.2, 0.8]])
    mask = torch.ones_like(values)
    expected, targets = compute_asharp(rewards, values, mask, beta, 0.4)
    actual, other_targets = compute_asharp(rewards, values, mask, beta, 0.4, 0.0, estimator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_targets, targets, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["mse", "positive"])
@pytest.mark.parametrize("aggregation", ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum"])
@pytest.mark.parametrize("clip", [None, 0.2])
def test_beta_zero_loss_and_parameter_gradient_match_ppo(mode, aggregation, clip):
    from verl.trainer.ppo.core_algos import compute_value_loss

    head = torch.nn.Linear(2, 1, bias=True).double()
    reference = torch.nn.Linear(2, 1, bias=True).double()
    reference.load_state_dict(head.state_dict())
    features = torch.tensor([[[1.0, -2.0], [0.5, 3.0]], [[-1.0, 0.2], [0.0, 0.0]]], dtype=torch.float64)
    old = torch.zeros(2, 2, dtype=torch.float64)
    target = torch.tensor([[1.0, -1.0], [0.5, 0.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    loss, frac = compute_log_moment_loss(
        head(features).squeeze(-1), target, mask, mode, aggregation, beta=0.0, values=old, cliprange_value=clip
    )
    expected, expected_frac = compute_value_loss(
        reference(features).squeeze(-1), target, old, mask, float("inf") if clip is None else clip, aggregation
    )
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    torch.testing.assert_close(frac, expected_frac, atol=0, rtol=0)
    loss.backward()
    expected.backward()
    for p, q in zip(head.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["mse", "positive"])
@pytest.mark.parametrize("beta", [0.2, 0.7, 2.0])
def test_fixed_beta_loss_equals_reparameterized_old_objective(mode, beta):
    h = torch.tensor([[-0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    loss, _ = compute_log_moment_loss(h, target, torch.ones_like(h), mode, beta=beta)
    x = beta * (target - h)
    if mode == "mse":
        exact = ((torch.exp(beta * h) - torch.exp(beta * target)).square() / (2 * beta**2)).mean()
    else:
        exact = ((torch.exp(x) - 1 - x) / beta**2).mean()
    torch.testing.assert_close(loss, exact, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(loss, h)[0], torch.autograd.grad(exact, h)[0], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("beta", [1e-4, 1e-8, 1e-16, 1e-30])
@pytest.mark.parametrize("mode", ["mse", "positive"])
def test_tiny_beta_loss_has_nonvanishing_value_gradient(beta, mode):
    h = torch.tensor([[0.4, -0.2]], requires_grad=True)
    target = torch.tensor([[1.1, 0.7]])
    loss, _ = compute_log_moment_loss(h, target, torch.ones_like(h), mode, beta=beta)
    expected = 0.5 * (h - target).square().mean()
    torch.testing.assert_close(loss, expected, rtol=3e-4, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(loss, h)[0], torch.autograd.grad(expected, h)[0], rtol=3e-4, atol=1e-7
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("beta", [1e-8, 1e-20])
def test_small_beta_actor_and_target_converge_with_reward_scale(estimator, scale, beta):
    rewards = scale * torch.tensor([[0.2, -0.7, 1.1]])
    values = scale * torch.tensor([[0.3, -0.2, 0.8]])
    mask = torch.ones_like(values)
    adv, target = compute_asharp(rewards, values, mask, beta, 0.9, 0.8, estimator)
    expected_adv, expected_target = compute_asharp(rewards, values, mask, 0.0, 0.9, 0.8, estimator)
    torch.testing.assert_close(adv, expected_adv, rtol=1e-4, atol=1e-6 * scale)
    torch.testing.assert_close(target, expected_target, rtol=1e-4, atol=1e-6 * scale)


def test_critic_lambda_independent_of_actor_estimator_and_lambda():
    rewards, values, mask = torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.1, 0.5, -0.3]]), torch.ones(1, 3)
    _, target = compute_asharp(rewards, values, mask, 0.8, 0.6)
    for estimator in ESTIMATORS:
        for lam in (0.0, 0.5, 1.0):
            _, actual = compute_asharp(rewards, values, mask, 0.8, 0.6, lam, estimator)
            torch.testing.assert_close(target, actual, rtol=0, atol=0)


def test_lambda_alias_migration_and_conflict():
    from verl.trainer.ppo.asharp import resolve_critic_lambda

    assert resolve_critic_lambda({}) == 0.95
    with pytest.warns(FutureWarning, match="deprecated"):
        assert resolve_critic_lambda({"asharp_lam": 0.4}) == 0.4
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_critic_lambda({"asharp_lam": 0.4, "asharp_lam_critic": 0.6})


def test_old_log_head_checkpoint_cannot_silently_resume(tmp_path):
    (tmp_path / "asharp_meta.json").write_text('{"value_type":"log_moment","beta":1.0}')
    with pytest.raises(ValueError, match="parameterization"):
        check_asharp_checkpoint(tmp_path, 1.0)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_padding_and_terminal_for_all_estimators(estimator):
    rewards = torch.tensor([[0.3, -1.0, float("nan")], [0.0, 0.0, 2.0]])
    values = torch.tensor([[0.4, -0.2, float("nan")], [0.2, 0.4, 0.7]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    adv, targets = compute_asharp(rewards, values, mask, 0.8, 0.9, 0.7, estimator)
    assert adv[0, 2] == targets[0, 2] == 0
    torch.testing.assert_close(adv[0, 1], torch.tensor(-0.8))
    torch.testing.assert_close(targets[0, 1], torch.tensor(-1.0))
    torch.testing.assert_close(targets[1, 2], torch.tensor(2.0))


def test_mse_large_negative_prediction_does_not_spuriously_overflow():
    h = torch.tensor([[-1000.0]], requires_grad=True)
    loss, _ = compute_log_moment_loss(h, torch.zeros_like(h), torch.ones_like(h), beta=1.0)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(h.grad).all()


def test_actor_exponential_overflow_is_not_silently_clipped():
    with pytest.raises(FloatingPointError):
        compute_asharp(
            torch.tensor([[1000.0, 1.0]]), torch.zeros(1, 2), torch.ones(1, 2), 1.0, 0.9, 0.9, "residual_exp"
        )


@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("lam", [0.4, 0.8, 1.0])
def test_exact_tree_jensen_bias_and_conditional_unbiasedness(beta, lam):
    from examples.asharp_trainer.compare_estimators import compare

    result = compare(beta, lam, 0.0)["estimators"]
    for row in result["log_plugin"]["root_conditional"]:
        assert row["bias"] < -1e-6  # strict Jensen inequality on both stochastic continuations
    for estimator in ("one_step", "residual_gae", "residual_exp"):
        for row in result[estimator]["root_conditional"]:
            assert abs(row["bias"]) < 1e-12
        assert result[estimator]["actor_gradient_l2_error"] < 1e-12


@pytest.mark.parametrize("noise", [0.1, 0.3])
def test_controlled_critic_error_bias_variance_mse_and_actor_gradient(noise):
    from examples.asharp_trainer.compare_estimators import compare

    result = compare(1.0, 0.8, noise)["estimators"]
    for estimator in ("one_step", *ESTIMATORS):
        metrics = result[estimator]
        assert math.isfinite(metrics["actor_gradient_l2_error"])
        assert metrics["actor_gradient_l2_error"] > 1e-6
        for row in metrics["root_conditional"]:
            assert abs(row["bias"]) > 1e-6
            assert row["variance"] >= 0
            assert row["mse"] == pytest.approx(row["bias"] ** 2 + row["variance"], abs=1e-12)
    # The fixture actually distinguishes the three constructions under critic error.
    assert len({round(result[e]["actor_gradient_l2_error"], 8) for e in ESTIMATORS}) == 3


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_actor_recursions_match_explicit_formula(estimator):
    beta, lam = 0.8, 0.6
    rewards = torch.tensor([[0.3, -0.2, 0.8]], dtype=torch.float64)
    h = torch.tensor([[0.1, 0.4, 0.6]], dtype=torch.float64)
    mask = torch.ones_like(h)
    actual, _ = compute_asharp(rewards, h, mask, beta, 0.9, lam, estimator)
    a, _ = compute_asharp(rewards, h, mask, beta)
    q, d = (beta * a).exp(), torch.expm1(beta * a) / beta
    expected = a.clone()
    for t in range(3):
        if estimator == "log_plugin":
            y = 0.0
            for n in range(1, 4 - t):
                terminal = t + n == 3
                bootstrap = 1.0 if terminal else (beta * h[0, t + n]).exp()
                y += (
                    lam ** (n - 1)
                    * (1 if terminal else 1 - lam)
                    * (beta * rewards[0, t : t + n].sum()).exp()
                    * bootstrap
                )
            expected[0, t] = y.log() / beta - h[0, t]
        else:
            for k in range(1, 3 - t):
                weight = 1.0 if estimator == "residual_gae" else q[0, t : t + k].prod()
                expected[0, t] += lam**k * weight * d[0, t + k]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["mse", "positive"])
@pytest.mark.parametrize("beta", [0.0, 1e-8, 0.7])
def test_loss_autograd_gradcheck(mode, beta):
    h = torch.tensor([[0.3, -0.4, 0.5]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.3, 0.6, -0.7]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda x: compute_log_moment_loss(x, target, torch.ones_like(x), mode, beta=beta)[0], (h,)
    )


def test_positive_loss_is_centered_at_zero_residual():
    h = torch.tensor([[0.3]], requires_grad=True)
    loss, _ = compute_log_moment_loss(h, h.detach(), torch.ones_like(h), "positive", beta=0.7)
    assert loss.item() == 0.0
    loss.backward()
    assert h.grad.item() == 0.0


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_single_terminal_action_has_no_future_exponential_correction(estimator):
    adv, target = compute_asharp(
        torch.tensor([[1000.0]]), torch.zeros(1, 1), torch.ones(1, 1), 1.0, 0.9, 0.9, estimator
    )
    assert adv.item() == target.item() == 1000.0


def test_positive_loss_large_negative_exponent_stays_finite():
    h = torch.tensor([[1e20]], requires_grad=True)
    loss, _ = compute_log_moment_loss(h, torch.zeros_like(h), torch.ones_like(h), "positive", beta=1.0)
    assert loss.item() == pytest.approx(1e20)
    loss.backward()
    assert h.grad.item() == pytest.approx(1.0)
