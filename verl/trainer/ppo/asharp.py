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
"""A-sharp estimators with entropic, binary-value, or group baselines.

For the entropic critic, H = log E[exp(beta G)] / beta and targets are in H
units. Final valid tokens terminate the scored episode.
"""

import math
import warnings
from collections import defaultdict

import torch

ACTOR_ESTIMATORS = ("log_plugin", "residual_gae", "residual_exp")
VALUE_TYPES = ("entropic_value", "log_moment")  # latter is a deprecated config alias


def _binary_terminal_rewards(rewards, response_mask, beta):
    rewards, _, valid = _inputs(rewards, torch.zeros_like(rewards), response_mask, beta)
    if not valid.any(dim=1).all():
        raise ValueError("Binary A-sharp requires at least one response token per sample")
    lengths = valid.sum(dim=1) - 1
    terminal = torch.zeros_like(valid).scatter_(1, lengths[:, None], True)
    if torch.any(rewards.masked_fill(terminal, 0) != 0) or torch.any(
        (rewards[terminal] != 0) & (rewards[terminal] != 1)
    ):
        raise ValueError("Binary A-sharp requires a terminal reward in {0, 1} and zero earlier rewards")
    return rewards, valid, rewards[terminal]


def binary_value_to_entropic(values, beta):
    """Convert Bernoulli success probabilities to log-moment values."""
    _check_beta(beta)
    p = values.clamp(0, 1)
    if beta == 0:
        return p
    if beta < 1e-3:
        # Avoid cancellation when the log-moment approaches its mean limit.
        return p + beta * p * (1 - p) * (0.5 + beta * (1 - 2 * p) / 6)
    if beta < 80:
        return torch.log1p(p * math.expm1(beta)) / beta
    return torch.logaddexp(torch.log1p(-p), torch.log(p) + beta) / beta


def _empirical_entropic_value(scores, beta):
    """Stable log(mean(exp(beta * return))) / beta for one response group."""
    mean = scores.mean()
    if beta == 0:
        return mean
    centered = scores - mean
    if beta * centered.abs().max() < 1e-3:
        variance = centered.square().mean()
        third_moment = centered.pow(3).mean()
        return mean + 0.5 * beta * variance + (beta * beta / 6) * third_moment
    maximum = scores.max()
    return maximum + (torch.logsumexp(beta * (scores - maximum), dim=0) - math.log(scores.numel())) / beta


def compute_binary_value_asharp(
    rewards,
    values,
    response_mask,
    beta,
    lam_critic=1.0,
    lam_actor=0.0,
    actor_estimator="residual_exp",
    normalize_advantages=False,
):
    """A-sharp actor signal from a PPO success-probability critic.

    The critic target stays in ordinary expected-reward units and is independent
    of beta. Only the actor uses the Bernoulli log-moment transform.
    """
    rewards, values, _ = _inputs(rewards, values, response_mask, beta)
    _binary_terminal_rewards(rewards, response_mask, beta)
    entropic_values = binary_value_to_entropic(values, beta)
    advantages, _ = compute_asharp(
        rewards,
        entropic_values,
        response_mask,
        beta,
        lam_critic,
        lam_actor,
        actor_estimator,
        normalize_advantages,
    )
    _, targets = compute_asharp(rewards, values, response_mask, 0.0, lam_critic)
    return advantages, targets


@torch.no_grad()
def compute_asharp_grpo(rewards, response_mask, index, beta, *, normalize_by_std=True, epsilon=1e-6):
    """Response-action A-sharp using each prompt group's empirical returns.

    Every response receives R_i - log(mean_j exp(beta R_j)) / beta, broadcast
    to its tokens. The group log moment estimates the prompt's value.
    """
    rewards, _, valid = _inputs(rewards, torch.zeros_like(rewards), response_mask, beta)
    if not valid.any(dim=1).all():
        raise ValueError("A-sharp GRPO requires at least one response token per sample")
    scores = rewards.sum(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("A-sharp GRPO requires finite response returns")
    if index is None or len(index) != len(scores):
        raise ValueError("A-sharp GRPO requires one group id per response")
    groups = defaultdict(list)
    for i, group_id in enumerate(index):
        groups[group_id].append(i)
    advantages = torch.empty_like(scores)
    for members in groups.values():
        if len(members) < 2:
            raise ValueError("A-sharp GRPO requires at least two responses per group")
        group_scores = scores[members]
        baseline = _empirical_entropic_value(group_scores, beta)
        group_advantages = group_scores - baseline
        if normalize_by_std:
            group_advantages = group_advantages / (group_scores.std() + epsilon)
        advantages[members] = group_advantages
    advantages = advantages[:, None] * valid.to(advantages.dtype)
    return advantages, advantages


def _check_beta(beta):
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("A-sharp beta must be finite and nonnegative")


def _check_lambda(lam):
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("A-sharp lambda must be in [0, 1]")


def _dtype(*tensors):
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _inputs(rewards, values, mask, beta):
    _check_beta(beta)
    if rewards.shape != values.shape or rewards.shape != mask.shape or rewards.ndim != 2:
        raise ValueError("A-sharp expects matching [batch, response_length] tensors")
    if rewards.numel() == 0 or not mask.bool().any():
        raise ValueError("A-sharp requires at least one valid response token")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("A-sharp requires a binary response mask")
    valid = mask.bool()
    if valid.shape[1] > 1 and torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("A-sharp currently requires contiguous responses followed by padding")
    dtype = _dtype(rewards, values)
    rewards = rewards.to(dtype).masked_fill(~valid, 0)
    values = values.to(dtype).masked_fill(~valid, 0)
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("Non-finite A-sharp rewards or critic predictions")
    return rewards, values, valid


def _exprel(x):
    """expm1(x)/x with a smooth, cancellation-free limit, including its gradient."""
    small = x.abs() < 1e-3
    xs = x.clamp(-1e-3, 1e-3)
    series = 1 + xs * (0.5 + xs * (1 / 6 + xs * (1 / 24 + xs / 120)))
    # Never evaluate a singular/overflowing unused Taylor branch in autograd.
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, torch.expm1(safe) / safe)


def _exprel2(x):
    """(expm1(x)-x)/x**2, stable near zero and differentiable."""
    small = x.abs() < 1e-2
    xs = x.clamp(-1e-2, 1e-2)
    series = 0.5 + xs * (1 / 6 + xs * (1 / 24 + xs * (1 / 120 + xs / 720)))
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, (torch.expm1(safe) - safe) / safe.square())


def _entropic_mix(delta, beta, lam):
    """log(1-lam + lam*exp(beta*delta))/beta, in return units."""
    if lam == 0:
        return torch.zeros_like(delta)
    if lam == 1:
        return delta
    if beta == 0:
        return lam * delta
    x = beta * delta
    small = x.abs() < 1e-3
    xs = x.clamp(-1e-3, 1e-3)
    # Bernoulli cumulant expansion avoids even dividing by a tiny beta.
    series = delta * (
        lam + xs * lam * (1 - lam) * (0.5 + xs * ((1 - 2 * lam) / 6 + xs * (1 - 6 * lam * (1 - lam)) / 24))
    )
    safe = torch.where(small, torch.zeros_like(x), x)
    moderate = safe.abs() < 0.5
    xm = safe.clamp(-0.5, 0.5)
    log_mix = torch.where(
        moderate,
        torch.log1p(lam * torch.expm1(xm)),
        torch.logaddexp(torch.full_like(safe, math.log1p(-lam)), safe + math.log(lam)),
    )
    return torch.where(small, series, log_mix / beta)


@torch.no_grad()
def compute_asharp(
    rewards,
    values,
    response_mask,
    beta,
    lam_critic=1.0,
    lam_actor=0.0,
    actor_estimator="residual_exp",
    normalize_advantages=False,
):
    """Compute actor A-sharp estimates and entropic critic targets in H units.

    Actor and critic lambdas are independent. The single reverse scan maintains
    the critic entropic residual, and the selected actor residual/return. At
    beta=0 all three multi-step estimators are ordinary undiscounted GAE.
    All estimators give the one-step signal when lam_actor=0.
    """
    _check_lambda(lam_critic)
    _check_lambda(lam_actor)
    if actor_estimator not in ACTOR_ESTIMATORS:
        raise ValueError(f"Unknown A-sharp actor estimator: {actor_estimator}")
    rewards, values, valid = _inputs(rewards, values, response_mask, beta)
    advantages = torch.zeros_like(values)
    targets = torch.zeros_like(values)
    next_value = torch.zeros_like(values[:, 0])
    critic_tail = torch.zeros_like(next_value)
    actor_tail = torch.zeros_like(next_value)
    effective_lam = lam_actor
    for t in reversed(range(values.shape[1])):
        a = rewards[:, t] + next_value - values[:, t]
        critic_tail = (a + _entropic_mix(critic_tail, beta, lam_critic)).masked_fill(~valid[:, t], 0)
        targets[:, t] = (critic_tail + values[:, t]).masked_fill(~valid[:, t], 0)
        if effective_lam == 0:
            adv = a
        elif beta == 0 or actor_estimator == "log_plugin":
            adv = a + _entropic_mix(actor_tail, beta, effective_lam)
            actor_tail = adv.masked_fill(~valid[:, t], 0)
        else:
            x = beta * a
            d = a * _exprel(x)
            propagated = effective_lam * actor_tail
            if actor_estimator == "residual_exp":
                # A zero future correction remains zero even if q overflows.
                propagation_x = torch.where(actor_tail == 0, torch.zeros_like(x), x)
                propagated = propagation_x.exp() * propagated
            adv = a + propagated
            actor_tail = (d + propagated).masked_fill(~valid[:, t], 0)
        advantages[:, t] = adv.masked_fill(~valid[:, t], 0)
        next_value = values[:, t]
    if not torch.isfinite(advantages).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite A-sharp signal; reduce fixed beta or reward scale")
    if normalize_advantages:
        from verl.utils.torch_functional import masked_whiten

        advantages = masked_whiten(advantages, response_mask).masked_fill(~valid, 0)
    return advantages, targets


def compute_log_moment_loss(
    vpreds,
    returns,
    response_mask,
    loss_mode="mse",
    loss_agg_mode="token-mean",
    *,
    beta=1.0,
    values=None,
    cliprange_value=None,
):
    """Normalized exponential-moment regression on H predictions and H targets.

    mse: (exp(beta*H)-exp(beta*R))**2/(2*beta**2).
    positive: (exp(beta*(R-H))-1-beta*(R-H))/beta**2.
    Both converge in value AND parameter gradient to value MSE. Optional PPO
    clipping is applied in H units to both branches, maintaining that limit.
    """
    from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss
    from verl.utils.torch_functional import masked_mean

    _check_beta(beta)
    if loss_mode not in ("mse", "positive"):
        raise ValueError(f"Unknown log-moment loss: {loss_mode}")
    valid = response_mask.bool()
    dtype = _dtype(vpreds, returns)
    h = vpreds.to(dtype).masked_fill(~valid, 0)
    target = returns.detach().to(dtype).masked_fill(~valid, 0)
    old = h.detach() if values is None else values.detach().to(dtype).masked_fill(~valid, 0)
    if cliprange_value is not None and (values is None or cliprange_value < 0):
        raise ValueError("Value clipping requires old values and a nonnegative clip range")
    if not torch.isfinite(h).all() or not torch.isfinite(target).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Non-finite entropic critic predictions/targets")
    if beta == 0:
        return compute_value_loss(
            h,
            target,
            old,
            response_mask,
            float("inf") if cliprange_value is None else cliprange_value,
            loss_agg_mode,
        )

    def element_loss(pred):
        delta = target - pred
        x = beta * delta
        if loss_mode == "mse":
            # Anchor at the larger exponent so expm1 never overflows merely
            # because the OTHER moment is tiny (e.g. H=-1000, R=0).
            difference = (beta * torch.maximum(pred, target)).exp() * delta * _exprel(-x.abs())
            return 0.5 * difference.square()
        small = x.abs() < 1e-2
        small_delta = torch.where(small, delta, torch.zeros_like(delta))
        small_x = torch.where(small, x, torch.zeros_like(x))
        series = small_delta.square() * _exprel2(small_x)
        far_x = torch.where(small, torch.zeros_like(x), x)
        far_beta = torch.where(small, torch.ones_like(x), torch.full_like(x, beta))
        # Divide twice rather than forming beta**2, which can underflow.
        # The far branch also avoids delta**2 overflow for large negative x.
        direct = ((torch.expm1(far_x) - far_x) / far_beta) / far_beta
        return torch.where(small, series, direct)

    losses = element_loss(h)
    clipfrac = h.new_zeros(())
    if cliprange_value is not None:
        clipped = torch.clamp(h, min=old - cliprange_value, max=old + cliprange_value)
        clipped_losses = element_loss(clipped)
        clipfrac = masked_mean((clipped_losses > losses).to(dtype), response_mask)
        losses = torch.maximum(losses, clipped_losses)
    if not torch.isfinite(losses[valid]).all():
        raise FloatingPointError("Non-finite log-moment loss; reduce fixed beta or reward scale")
    return agg_loss(losses, response_mask, loss_agg_mode), clipfrac


def resolve_asharp_lambdas(config):
    """Use the shared PPO lambda settings; reject obsolete ambiguous aliases."""
    from verl.trainer.ppo.core_algos import resolve_lambdas

    for old, new in (
        ("asharp_lam", "lam_critic"),
        ("asharp_lam_critic", "lam_critic"),
        ("asharp_lam_actor", "lam_policy"),
    ):
        if config.get(old) is not None:
            raise ValueError(f"{old} was removed; use algorithm.{new}")
    if config.get("asharp_actor_estimator") == "one_step":
        raise ValueError("one_step was removed; use residual_exp (or another estimator) with lam_policy=0")
    return resolve_lambdas(config.get("lam", 1.0), config.get("lam_policy"), config.get("lam_critic"))


def compute_asharp_from_config(rewards, values, response_mask, config):
    """Shared driver entry for initial estimation and optional post-update refresh."""
    policy_lam, critic_lam = resolve_asharp_lambdas(config)
    estimator = compute_binary_value_asharp if config.get("asharp_binary_value_critic", False) else compute_asharp
    return estimator(
        rewards,
        values,
        response_mask,
        config.asharp_beta,
        critic_lam,
        policy_lam,
        config.get("asharp_actor_estimator", "residual_exp"),
        config.get("asharp_normalize_advantages", False),
    )


def validate_asharp_config(config):
    """Fail before workers start when critic semantics or task assumptions differ."""
    algo, critic = config.algorithm, config.critic
    binary_value = algo.get("asharp_binary_value_critic", False)
    expected_types = ("value",) if binary_value else VALUE_TYPES
    if algo.adv_estimator != "asharp" or critic.get("value_type", "value") not in expected_types:
        raise ValueError(
            "Binary-value A-sharp requires critic.value_type=value; "
            "ordinary A-sharp requires critic.value_type=entropic_value"
        )
    if critic.value_type == "log_moment":
        warnings.warn(
            "value_type=log_moment is deprecated; use entropic_value (head outputs H, not u)",
            FutureWarning,
            stacklevel=2,
        )
    if critic.enable is False:
        raise ValueError("A-sharp requires a critic")
    if critic.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("A-sharp currently supports FSDP/FSDP2 critics only")
    if not binary_value and critic.get("log_moment_loss", "mse") not in ("mse", "positive"):
        raise ValueError("critic.log_moment_loss must be mse or positive")
    _check_beta(algo.asharp_beta)
    if not binary_value and critic.get("asharp_beta", algo.asharp_beta) != algo.asharp_beta:
        raise ValueError("critic.asharp_beta must equal algorithm.asharp_beta")
    resolve_asharp_lambdas(algo)
    if algo.get("asharp_actor_estimator", "residual_exp") not in ACTOR_ESTIMATORS:
        raise ValueError("Unknown asharp_actor_estimator")
    if algo.gamma != 1:
        raise ValueError("A-sharp currently implements undiscounted returns (gamma=1)")
    if algo.use_kl_in_reward:
        raise ValueError("A-sharp requires task rewards without policy-dependent KL shaping")
    if algo.get("use_pf_ppo", False):
        raise ValueError("PF-PPO reweighting is not implemented for A-sharp")
    rollout = config.actor_rollout_ref.rollout
    if rollout.get("multi_turn", {}).get("enable", False):
        raise ValueError("A-sharp currently supports single-turn deterministic tasks only")
    correction = algo.get("rollout_correction")
    if correction and correction.get("rollout_rs"):
        raise ValueError("A-sharp does not support rejection masks in critic targets")
    adapter = critic.get("model", {}).get("lora_adapter_path")
    if adapter:
        check_asharp_checkpoint(adapter, algo.asharp_beta, binary_value=binary_value)


def validate_asharp_grpo_config(config):
    """Check assumptions for the critic-free group return estimator."""
    algo = config.algorithm
    if algo.adv_estimator != "asharp_grpo":
        raise ValueError("Use algorithm.adv_estimator=asharp_grpo")
    _check_beta(algo.asharp_beta)
    if algo.gamma != 1:
        raise ValueError("A-sharp GRPO requires gamma=1")
    if algo.use_kl_in_reward:
        raise ValueError("A-sharp GRPO requires task rewards without KL shaping")
    if algo.get("use_pf_ppo", False):
        raise ValueError("PF-PPO reweighting is not implemented for A-sharp GRPO")
    if config.critic.enable is True:
        raise ValueError("A-sharp GRPO does not use a critic; set critic.enable=False")
    rollout = config.actor_rollout_ref.rollout
    if rollout.n < 2:
        raise ValueError("A-sharp GRPO requires rollout.n >= 2")
    if rollout.get("multi_turn", {}).get("enable", False):
        raise ValueError("A-sharp GRPO currently supports single-turn tasks only")


def save_asharp_checkpoint(directory, beta, *, binary_value=False):
    """Persist fixed temperature and versioned head/target units."""
    import json
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    meta = (
        {"version": 3, "value_type": "value", "parameterization": "p", "target_units": "p", "beta_independent": True}
        if binary_value
        else {"version": 2, "value_type": "entropic_value", "parameterization": "H", "target_units": "H", "beta": beta}
    )
    metadata = json.dumps(meta) + "\n"
    (path / "asharp_meta.json").write_text(metadata)
    if (path / "lora_adapter").is_dir():
        (path / "lora_adapter" / "asharp_meta.json").write_text(metadata)


def check_asharp_checkpoint(directory, beta, *, binary_value=False):
    """Check critic units, allowing a binary-value head across beta values."""
    import json
    from pathlib import Path

    path = Path(directory) / "asharp_meta.json"
    if not path.exists():
        units = "success probability p" if binary_value else "H"
        warnings.warn(
            f"Missing A-sharp critic metadata at {path}; interpreting the loaded critic output as {units}",
            UserWarning,
            stacklevel=2,
        )
        return
    meta = json.loads(path.read_text())
    if binary_value:
        expected = {
            "version": 3,
            "value_type": "value",
            "parameterization": "p",
            "target_units": "p",
            "beta_independent": True,
        }
        if meta != expected:
            raise ValueError("Incompatible A-sharp critic checkpoint: expected a binary success-probability head")
        return
    if (
        meta.get("version") != 2
        or meta.get("parameterization") != "H"
        or meta.get("target_units") != "H"
        or meta.get("value_type") != "entropic_value"
    ):
        raise ValueError("Incompatible critic parameterization: old u/log-moment checkpoints need explicit migration")
    if "beta" not in meta:
        warnings.warn(
            f"A-sharp critic metadata at {path} has no beta; loading with beta={beta}",
            UserWarning,
            stacklevel=2,
        )
    elif meta["beta"] != beta:
        warnings.warn(
            f"A-sharp critic beta={meta['beta']} differs from current beta={beta}; loading the saved H head",
            UserWarning,
            stacklevel=2,
        )
