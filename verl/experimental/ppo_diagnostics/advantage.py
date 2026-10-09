"""Optional, read-only diagnostics for long-response terminal-reward PPO."""

from __future__ import annotations

import torch


def summarize_position_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor,
    window: int = 100,
) -> dict[str, float]:
    """Measure advantage sign near the beginning and end of correct/incorrect responses."""
    if window < 1:
        raise ValueError("window must be positive")
    if advantages.shape != response_mask.shape or advantages.shape != token_level_scores.shape:
        raise ValueError("advantages, response_mask and token_level_scores must have the same shape")
    if advantages.ndim != 2:
        raise ValueError("expected [batch, response_tokens] tensors")

    with torch.no_grad():
        valid = response_mask.bool()
        lengths = valid.sum(dim=-1)
        positions = torch.arange(advantages.shape[-1], device=advantages.device).unsqueeze(0)
        early = valid & (positions < window)
        late = valid & (positions >= (lengths.unsqueeze(1) - window))
        correct = token_level_scores.sum(dim=-1) > 0.5
        result: dict[str, float] = {}
        for outcome, group in (("correct", correct), ("incorrect", ~correct)):
            for part, region in (("early", early), ("late", late)):
                selected = region & group.unsqueeze(1)
                if not selected.any():
                    continue
                values = advantages[selected].float()
                prefix = f"advantage_position/{outcome}_{part}"
                result[f"{prefix}_mean"] = values.mean().item()
                result[f"{prefix}_positive_fraction"] = (values > 0).float().mean().item()
                result[f"{prefix}_tokens"] = int(values.numel())
        return result
