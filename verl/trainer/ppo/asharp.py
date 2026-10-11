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
"""Fixed-temperature A-sharp estimation for deterministic, finite episodes.

Critic outputs and targets are log exponential moments, NOT ordinary values.
The final valid response token terminates the task (including scored length caps).
"""

import math

import torch


def _inputs(rewards, log_values, mask, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("A-sharp beta must be finite and positive")
    if rewards.shape != log_values.shape or rewards.shape != mask.shape or rewards.ndim != 2:
        raise ValueError("A-sharp expects matching [batch, response_length] tensors")
    if rewards.numel() == 0 or not mask.bool().any():
        raise ValueError("A-sharp requires at least one valid response token")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("A-sharp requires a binary response mask")
    valid = mask.bool()
    if valid.shape[1] > 1 and torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("A-sharp currently requires contiguous responses followed by padding")
    rewards = rewards.float().masked_fill(~valid, 0)
    log_values = log_values.float().masked_fill(~valid, 0)
    if not torch.isfinite(rewards).all() or not torch.isfinite(log_values).all():
        raise ValueError("Non-finite A-sharp rewards or critic predictions")
    return rewards, log_values, valid


@torch.no_grad()
def compute_asharp(rewards, log_values, response_mask, beta, lam):
    """Return detached A-sharp and log entropic lambda targets (proposal 58, 60).

    log_values[:, t] is the pre-action state prediction. Terminal log moment is
    zero. Lambda changes critic targets only, never accumulates actor advantages.
    """
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("A-sharp lambda must be in [0, 1]")
    rewards, log_values, valid = _inputs(rewards, log_values, response_mask, beta)
    advantages = torch.zeros_like(log_values)
    targets = torch.zeros_like(log_values)
    next_value = torch.zeros_like(log_values[:, 0])
    next_target = torch.zeros_like(next_value)
    for t in reversed(range(log_values.shape[1])):
        if lam == 0:
            continuation = next_value
        elif lam == 1:
            continuation = next_target
        else:
            continuation = torch.logaddexp(next_value + math.log1p(-lam), next_target + math.log(lam))
        target = beta * rewards[:, t] + continuation
        advantages[:, t] = (rewards[:, t] + (next_value - log_values[:, t]) / beta).masked_fill(~valid[:, t], 0)
        targets[:, t] = target.masked_fill(~valid[:, t], 0)
        next_target = targets[:, t]
        next_value = log_values[:, t]
    if not torch.isfinite(advantages).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite A-sharp signal; reduce fixed beta or reward scale")
    return advantages, targets


def compute_log_moment_loss(vpreds, returns, response_mask, loss_mode="mse", loss_agg_mode="token-mean"):
    """Regress E[Y|s], even though predictions and detached targets are in log space.

    mse: 1/2 (exp(u) - Y)^2. positive: u + Y exp(-u).
    Neither is MSE(u, log(Y)), which would learn the wrong conditional statistic.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    valid = response_mask.bool()
    u = vpreds.float().masked_fill(~valid, 0)
    z = returns.detach().float().masked_fill(~valid, 0)
    if loss_mode == "mse":
        losses = 0.5 * (u.exp() - z.exp()).square()
    elif loss_mode == "positive":
        losses = u + (z - u).exp()
    else:
        raise ValueError(f"Unknown log-moment loss: {loss_mode}")
    if not torch.isfinite(losses[valid]).all():
        raise FloatingPointError("Non-finite log-moment loss; reduce fixed beta or reward scale")
    return agg_loss(losses, response_mask, loss_agg_mode), u.new_zeros(())


def validate_asharp_config(config):
    """Fail before workers start when critic semantics or task assumptions differ."""
    algo, critic = config.algorithm, config.critic
    if algo.adv_estimator != "asharp" or critic.get("value_type", "value") != "log_moment":
        raise ValueError("Use algorithm.adv_estimator=asharp together with critic.value_type=log_moment")
    if critic.enable is False:
        raise ValueError("A-sharp requires a critic")
    if critic.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("A-sharp currently supports FSDP/FSDP2 critics only")
    if critic.get("log_moment_loss", "mse") not in ("mse", "positive"):
        raise ValueError("critic.log_moment_loss must be mse or positive")
    if not math.isfinite(algo.asharp_beta) or algo.asharp_beta <= 0:
        raise ValueError("A-sharp beta must be finite and positive")
    if not math.isfinite(algo.asharp_lam) or not 0 <= algo.asharp_lam <= 1:
        raise ValueError("A-sharp lambda must be in [0, 1]")
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
        check_asharp_checkpoint(adapter, algo.asharp_beta)


def save_asharp_checkpoint(directory, beta):
    """Persist the meaning of a critic head and its fixed temperature."""
    import json
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"value_type": "log_moment", "beta": beta}) + "\n"
    (path / "asharp_meta.json").write_text(metadata)
    if (path / "lora_adapter").is_dir():
        (path / "lora_adapter" / "asharp_meta.json").write_text(metadata)


def check_asharp_checkpoint(directory, beta):
    import json
    from pathlib import Path

    path = Path(directory) / "asharp_meta.json"
    if not path.exists():
        raise ValueError(f"Missing log-moment critic metadata: {path}; ordinary value checkpoints are incompatible")
    meta = json.loads(path.read_text())
    if meta.get("value_type") != "log_moment" or meta.get("beta") != beta:
        raise ValueError("A-sharp checkpoint critic semantics or fixed beta differ from this run")
