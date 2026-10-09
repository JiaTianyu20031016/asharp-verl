"""Experimental checks for the single-minibatch PPO regime used in MATH runs."""

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla


def test_on_policy_single_update_has_nonzero_gradient_but_zero_clipping_metrics():
    config = OmegaConf.create(
        {"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2,
         "clip_ratio_c": 1.0e10, "global_batch_info": {}}
    )
    log_prob = torch.tensor([[-0.2, -0.4]], requires_grad=True)
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(advantages)

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=log_prob.detach(), log_prob=log_prob,
        advantages=advantages, response_mask=mask, config=config,
    )
    loss.backward()

    assert metrics["actor/ppo_kl"] == 0.0
    assert metrics["actor/pg_clipfrac"] == 0.0
    assert metrics["actor/pg_clipfrac_lower"] == 0.0
    torch.testing.assert_close(log_prob.grad, torch.tensor([[-0.5, 0.5]]))


def test_3_and_4_gpu_microbatch_averaging_match_for_same_order():
    # 96 prompts x 4 responses = 384 trajectories; 16 trajectories/microbatch.
    values = torch.arange(384 * 5, dtype=torch.float32).reshape(384, 5) / 1000
    mask = torch.ones_like(values)
    mask[:, 1:] = (torch.arange(384).unsqueeze(1) % 5 < torch.arange(1, 5)).float()

    def simulated_fsdp_loss(gpus: int) -> torch.Tensor:
        local_size = 384 // gpus
        local_losses = []
        for rank in range(gpus):
            local = values[rank * local_size : (rank + 1) * local_size]
            local_mask = mask[rank * local_size : (rank + 1) * local_size]
            micro_losses = [
                agg_loss(micro, micro_mask, "token-mean")
                for micro, micro_mask in zip(local.split(16), local_mask.split(16), strict=True)
            ]
            local_losses.append(torch.stack(micro_losses).mean())
        return torch.stack(local_losses).mean()

    torch.testing.assert_close(simulated_fsdp_loss(3), simulated_fsdp_loss(4))
    # This legacy microbatch average is not an exact global token-weighted mean.
    assert not torch.isclose(simulated_fsdp_loss(3), agg_loss(values, mask, "token-mean"))
