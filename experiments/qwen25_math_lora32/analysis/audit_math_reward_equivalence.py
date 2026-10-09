#!/usr/bin/env python3
"""Audit this experiment's boxed MATH rewards for symbolic equivalence."""

from __future__ import annotations

import argparse
import re
import signal
from pathlib import Path

import pandas as pd
import sympy
from sympy.parsing.latex import parse_latex

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def normalize(text: str) -> str:
    text = text.strip().replace(r"\$", "").replace("$", "")
    text = text.replace(r"\!", "").replace(r"\,", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return text.strip()


class TimedOut(Exception):
    pass


def _alarm(_signum, _frame):
    raise TimedOut


def is_candidate_equivalent(pred: str, gold: str) -> tuple[bool, str]:
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        predicted = parse_latex(normalize(pred), backend="lark")
        target = parse_latex(normalize(gold), backend="lark")
        equivalent = sympy.simplify(predicted - target) == 0
        return bool(equivalent), "parsed"
    except TimedOut:
        return False, "timeout"
    except Exception:
        return False, "unparsed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.input)
    wrong = data[data.reward.eq(0)].copy()
    results = []
    for row in wrong.itertuples(index=False):
        boxed = last_boxed_only_string(row.response)
        if boxed is None:
            continue
        try:
            predicted_answer = remove_boxed(boxed)
        except Exception:
            continue
        gold_answer = row.reward_model["ground_truth"]
        equivalent, status = is_candidate_equivalent(predicted_answer, gold_answer)
        results.append(
            {"prompt_index": row.prompt_index, "sample_index": row.sample_index,
             "predicted_answer": predicted_answer, "gold_answer": gold_answer,
             "equivalent_candidate": equivalent, "parse_status": status}
        )
    output = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"responses={len(data)}, scored_zero={len(wrong)}, boxed_scored_zero={len(output)}")
    print(output.parse_status.value_counts().to_dict())
    print(f"symbolic_equivalence_candidates={int(output.equivalent_candidate.sum())}")
    print(output[output.equivalent_candidate].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
