import pytest

pytest.importorskip("math_verify")

from experiments.qwen25_math_lora32.reward_math_verify import compute_score


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (r"The answer is \boxed{87.5}.", r"\frac{175}{2}"),
        (r"Therefore \boxed{58,500}.", "58500"),
        (r"Thus \boxed{12\sqrt{2}+16}.", r"16+12\sqrt{2}"),
    ],
)
def test_terminal_scorer_accepts_equivalent_math(prediction, target):
    assert compute_score(None, prediction, target) == 1.0


def test_terminal_scorer_rejects_wrong_answer_and_has_no_partial_format_reward():
    assert compute_score(None, r"The answer is \boxed{2}.", "3") == 0.0
    assert compute_score(None, "No final answer was given.", "3") == 0.0
    assert compute_score(None, r"The derivation reaches \frac{6}{7}, but the output is truncated", r"\frac{6}{7}") == 0.0


def test_terminal_scorer_preserves_strict_percentage_equivalence():
    assert compute_score(None, r"The answer is \boxed{25}.", r"25\%") == 1.0
