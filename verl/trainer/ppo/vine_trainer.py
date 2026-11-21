"""Vine training: auxiliary continuations before the PPO actor update."""

import math

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.vine_algos import (
    compute_vine_advantages,
    estimate_vine_value,
    require_terminal_rewards,
    select_mc_points,
    validate_mc_points,
)


def validate_vine_config(config) -> None:
    """Reject modes whose rewards or trajectories cannot be interpreted here."""
    algo = config.algorithm
    if algo.adv_estimator != "vine":
        raise ValueError("Vine requires algorithm.adv_estimator=vine")
    if algo.vine_advantage_estimator not in ("mc", "asharp"):
        raise ValueError("Vine advantage estimator must be mc or asharp")
    if algo.vine_advantage_estimator == "asharp" and (not math.isfinite(algo.asharp_beta) or algo.asharp_beta < 0):
        raise ValueError("Vine requires a finite, nonnegative asharp_beta")
    if not isinstance(algo.vine_mc_samples, int) or algo.vine_mc_samples < 1:
        raise ValueError("Vine requires vine_mc_samples >= 1")
    if not isinstance(algo.vine_max_points, int) or algo.vine_max_points < 1:
        raise ValueError("Vine requires vine_max_points >= 1")
    if algo.gamma != 1:
        raise ValueError("Vine currently requires gamma=1")
    if algo.use_kl_in_reward:
        raise ValueError("Vine requires unshaped task rewards; set use_kl_in_reward=False")
    if algo.get("use_pf_ppo", False):
        raise ValueError("Vine does not support PF-PPO reweighting")
    if config.critic.enable is not False:
        raise ValueError("Vine does not use a critic; set critic.enable=False")
    rollout = config.actor_rollout_ref.rollout
    if rollout.n < 1 or rollout.get("multi_turn", {}).get("enable", False):
        raise ValueError("Vine currently requires single-turn rollout with n >= 1")


class RayVinePPOTrainer(RayPPOTrainer):
    """Reuse PPO actor optimization, extending its data collection stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.processor is not None:
            raise ValueError("Vine continuation currently supports text-only models")

    def _generate_auxiliary_rollouts(self, gen_batch, gen_batch_output):
        config = self.config.algorithm
        mask = gen_batch_output.batch["response_mask"]
        selected = select_mc_points(gen_batch_output.batch["responses"], mask, self.tokenizer, config)
        points = validate_mc_points(selected, mask, config.vine_max_points)

        # The primary batch may later be balanced and reordered. Carry a stable
        # row id through the normal union and use it to restore point estimates.
        original_count = len(gen_batch_output)
        gen_batch_output.non_tensor_batch["vine_origin_id"] = np.arange(original_count)
        origin_prompts = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        if len(origin_prompts) != original_count:
            raise ValueError("Vine prompt and primary response counts differ")

        origin_ids = []
        point_ids = []
        prefixes = []
        for row, offsets in enumerate(points):
            for point in offsets:
                prefix = gen_batch_output.batch["responses"][row, :point].tolist()
                for _ in range(config.vine_mc_samples):
                    origin_ids.append(row)
                    point_ids.append(point)
                    prefixes.append(prefix)

        auxiliary_prompts = origin_prompts.select_idxs(origin_ids)
        prefix_array = np.empty(len(prefixes), dtype=object)
        prefix_array[:] = prefixes
        auxiliary_prompts.non_tensor_batch["vine_prefix_ids"] = prefix_array
        auxiliary_prompts.non_tensor_batch["agent_name"] = np.array(
            ["vine_continuation_agent"] * len(prefixes), dtype=object
        )
        auxiliary = self.async_rollout_manager.generate_sequences(auxiliary_prompts)
        auxiliary.meta_info.pop("timing", None)
        return auxiliary, points, origin_ids, point_ids

    def _compute_auxiliary_advantages(self, batch, auxiliary_rollouts):
        if auxiliary_rollouts is None:
            raise ValueError("Vine auxiliary rollouts are missing")
        auxiliary, original_points, origin_ids, point_ids = auxiliary_rollouts
        if "rm_scores" not in auxiliary.batch:
            if not self.use_rm:
                raise ValueError("Vine auxiliary responses have no reward scores")
            auxiliary = auxiliary.union(self._compute_reward_colocate(auxiliary))
        auxiliary_rewards, _ = extract_reward(auxiliary)
        require_terminal_rewards(auxiliary_rewards, auxiliary.batch["response_mask"])
        returns = auxiliary_rewards.sum(dim=-1)
        if not torch.isfinite(returns).all():
            raise ValueError("Vine Monte Carlo rewards must be finite")

        k = self.config.algorithm.vine_mc_samples
        estimator = self.config.algorithm.vine_advantage_estimator
        beta = self.config.algorithm.asharp_beta
        estimates = {row: [] for row in range(len(original_points))}
        cursor = 0
        for row, points in enumerate(original_points):
            for point in points:
                if origin_ids[cursor : cursor + k] != [row] * k or point_ids[cursor : cursor + k] != [point] * k:
                    raise ValueError("Vine auxiliary rollout order changed")
                estimates[row].append(estimate_vine_value(returns[cursor : cursor + k], estimator, beta))
                cursor += k
        if cursor != len(returns):
            raise ValueError("Vine auxiliary rollout count changed")

        original_ids = batch.non_tensor_batch["vine_origin_id"]
        ordered_points = [original_points[int(row)] for row in original_ids]
        ordered_estimates = [estimates[int(row)] for row in original_ids]
        rewards = batch.batch["token_level_rewards"]
        require_terminal_rewards(rewards, batch.batch["response_mask"])
        ordered_estimates = [
            [value.to(device=rewards.device, dtype=rewards.dtype) for value in row] for row in ordered_estimates
        ]
        advantages = compute_vine_advantages(rewards, batch.batch["response_mask"], ordered_points, ordered_estimates)
        batch.non_tensor_batch.pop("vine_origin_id")
        return advantages
