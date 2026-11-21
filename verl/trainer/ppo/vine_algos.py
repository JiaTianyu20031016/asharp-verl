"""Monte Carlo value estimates for selected response-prefix states.

The selector is intentionally left open: different projects can define what a
reasoning step or other useful continuation point means for their data.
"""

import math

import torch


def select_mc_points(response_ids: torch.Tensor, response_mask: torch.Tensor, tokenizer, config) -> list[list[int]]:
    """Return token offsets for states to sample, one list per original response.

    An offset ``p`` denotes the state after the first ``p`` valid response
    tokens. Offset zero is added by the trainer, so implementations only need
    to return additional points in ``[1, response_length)``. A point at the
    terminal state has no continuation and must not be returned.
    """
    raise NotImplementedError("Implement select_mc_points for the task's reasoning traces")


def validate_mc_points(points: list[list[int]], response_mask: torch.Tensor, max_points: int) -> list[list[int]]:
    """Normalize point lists and enforce the rollout budget before sampling."""
    if len(points) != response_mask.shape[0]:
        raise ValueError("Vine point selector must return one list per response")
    normalized = []
    for row, selected in enumerate(points):
        valid = response_mask[row].bool()
        length = int(valid.sum().item())
        if length == 0 or not valid[:length].all() or valid[length:].any():
            raise ValueError("Vine requires a nonempty, contiguous single-turn response mask")
        if any(not isinstance(point, int) or isinstance(point, bool) for point in selected):
            raise ValueError("Vine point offsets must be integers")
        states = sorted(set([0, *selected]))
        if len(states) > max_points:
            raise ValueError(f"Vine selected {len(states)} points, exceeding vine_max_points={max_points}")
        if states[-1] >= length or states[0] < 0:
            raise ValueError("Vine points must be before the terminal response state")
        normalized.append(states)
    return normalized


def mc_entropic_value(returns: torch.Tensor, beta: float) -> torch.Tensor:
    """Plug-in estimate of log E[exp(beta R)] / beta from continuation returns."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Vine beta must be finite and nonnegative")
    if returns.ndim != 1 or returns.numel() == 0 or not torch.isfinite(returns).all():
        raise ValueError("Vine requires a nonempty vector of finite Monte Carlo returns")
    if beta == 0:
        return returns.mean()
    mean = returns.mean()
    centered = returns - mean
    if beta * centered.abs().max() < 1e-3:
        return mean + 0.5 * beta * centered.square().mean() + beta**2 * centered.pow(3).mean() / 6
    maximum = returns.max()
    return maximum + (torch.logsumexp(beta * (returns - maximum), dim=0) - math.log(returns.numel())) / beta


def estimate_vine_value(returns: torch.Tensor, estimator: str, beta: float) -> torch.Tensor:
    """Estimate a prefix value with ordinary MC or the A-sharp log moment."""
    if returns.ndim != 1 or returns.numel() == 0 or not torch.isfinite(returns).all():
        raise ValueError("Vine requires a nonempty vector of finite Monte Carlo returns")
    if estimator == "mc":
        return returns.mean()
    if estimator == "asharp":
        return mc_entropic_value(returns, beta)
    raise ValueError(f"Unsupported Vine value estimator: {estimator}")


def require_terminal_rewards(rewards: torch.Tensor, response_mask: torch.Tensor) -> None:
    """Ensure full-response Monte Carlo scores have no earlier reward to double-count."""
    if rewards.shape != response_mask.shape:
        raise ValueError("Vine reward and response-mask shapes must agree")
    valid = response_mask.bool()
    lengths = valid.sum(dim=-1)
    if (lengths == 0).any():
        raise ValueError("Vine requires a nonempty scored response")
    terminal = torch.zeros_like(valid).scatter_(1, (lengths - 1)[:, None], True)
    if torch.any(rewards.masked_fill(~valid | terminal, 0) != 0):
        raise ValueError("Vine currently requires terminal-only task rewards")


def compute_vine_advantages(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    points: list[list[int]],
    state_values: list[list[torch.Tensor]],
    estimator: str = "asharp",
) -> torch.Tensor:
    """Assign one advantage to each reasoning segment.

    Original VinePPO uses the observed return from the segment's start minus
    its MC value. A-sharp keeps the local segment reward plus next entropic
    value minus current entropic value. Values are broadcast over the segment.
    """
    if estimator not in ("mc", "asharp"):
        raise ValueError(f"Unsupported Vine advantage estimator: {estimator}")
    if rewards.shape != response_mask.shape or len(points) != rewards.shape[0] or len(state_values) != len(points):
        raise ValueError("Vine reward, mask, point, and value shapes must agree")
    result = torch.zeros_like(rewards)
    for row, (selected, estimates) in enumerate(zip(points, state_values, strict=True)):
        if len(selected) != len(estimates):
            raise ValueError("Vine requires one Monte Carlo estimate per selected point")
        length = int(response_mask[row].sum().item())
        if not selected or selected[0] != 0 or selected != sorted(set(selected)) or selected[-1] >= length:
            raise ValueError("Vine points must start at zero and be unique, sorted, and nonterminal")
        for idx, start in enumerate(selected):
            end = selected[idx + 1] if idx + 1 < len(selected) else length
            if estimator == "mc":
                result[row, start:end] = rewards[row, start:length].sum() - estimates[idx]
            else:
                next_value = estimates[idx + 1] if idx + 1 < len(selected) else 0
                segment_reward = rewards[row, start:end].sum()
                result[row, start:end] = segment_reward + next_value - estimates[idx]
    if not torch.isfinite(result).all():
        raise ValueError("Vine produced non-finite advantages")
    return result
