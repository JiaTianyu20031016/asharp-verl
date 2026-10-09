#!/usr/bin/env python3
"""Compare strict boxed equality with Math-Verify on saved MATH rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from verl.utils.reward_score.math_reward import compute_score as strict_score
from experiments.qwen25_math_lora32.reward_math_verify import compute_score as robust_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    started = time.perf_counter()
    rows = []
    for row in frame.itertuples(index=False):
        ground_truth = row.reward_model["ground_truth"]
        strict = float(strict_score(row.response, ground_truth))
        robust = float(robust_score(row.data_source, row.response, ground_truth))
        if strict != robust:
            rows.append(
                {
                    "prompt_index": row.prompt_index,
                    "sample_index": row.sample_index,
                    "ground_truth": ground_truth,
                    "strict_score": strict,
                    "math_verify_score": robust,
                    "response": row.response,
                }
            )
    elapsed = time.perf_counter() - started
    disagreements = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    disagreements.to_parquet(args.output / "scorer_disagreements.parquet", index=False)
    disagreements.drop(columns=["response"], errors="ignore").to_csv(
        args.output / "scorer_disagreements.csv", index=False
    )
    summary = {
        "responses": len(frame),
        "elapsed_seconds": elapsed,
        "milliseconds_per_response": 1000 * elapsed / len(frame),
        "disagreements": len(disagreements),
        "strict_zero_math_verify_one": int(
            ((disagreements.strict_score == 0) & (disagreements.math_verify_score == 1)).sum()
        )
        if not disagreements.empty
        else 0,
        "strict_one_math_verify_zero": int(
            ((disagreements.strict_score == 1) & (disagreements.math_verify_score == 0)).sum()
        )
        if not disagreements.empty
        else 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
