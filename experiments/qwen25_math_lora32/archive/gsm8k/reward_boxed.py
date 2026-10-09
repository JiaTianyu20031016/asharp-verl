"""Archived binary reward from the superseded GSM8K experiment."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_BOXED_NUMBER = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")


def _as_decimal(text: str) -> Decimal | None:
    normalized = text.strip().replace(",", "").replace("$", "")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def compute_score(
    data_source: str | None,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return 1 only when the last boxed numeric answer equals the target."""
    del data_source, extra_info, kwargs
    matches = _BOXED_NUMBER.findall(solution_str)
    if not matches:
        return 0.0
    prediction = _as_decimal(matches[-1])
    target = _as_decimal(str(ground_truth))
    return float(prediction is not None and target is not None and prediction == target)
