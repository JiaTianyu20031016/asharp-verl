"""Binary terminal reward for a complete final boxed math answer."""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score.math_reward import compute_score as strict_math_score
from verl.utils.reward_score.math_reward import last_boxed_only_string
from verl.utils.reward_score.math_verify import compute_score as verify_math_answer


def compute_score(
    data_source: str | None,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Score the final complete boxed answer by strict or symbolic equivalence."""
    del data_source, extra_info, kwargs
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is None:
        return 0.0
    if strict_math_score(solution_str, ground_truth):
        return 1.0
    return float(verify_math_answer(boxed_answer, ground_truth))
