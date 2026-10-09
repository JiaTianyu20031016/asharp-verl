#!/usr/bin/env python3
"""Experiment-specific summary of checkpoint PPO credit diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LAMBDAS = ("0.95", "0.99", "1.0")


def parse(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.open(errors="replace"):
        if "PPO_DIAGNOSTIC " not in line:
            continue
        raw = ANSI.sub("", line.split("PPO_DIAGNOSTIC ", 1)[1]).strip()
        # Ray may interleave another log after the JSON object on the same line.
        rows.append(json.JSONDecoder().raw_decode(raw)[0])
    if not rows:
        raise ValueError(f"No PPO_DIAGNOSTIC records in {path}")
    frame = pd.DataFrame(rows).sort_values("batch_index").reset_index(drop=True)
    if frame.batch_index.duplicated().any():
        raise ValueError("Duplicate diagnostic batch indices")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = parse(args.log)
    frame.to_csv(args.output_dir / "diagnostic_batches.csv", index=False)

    records = []
    for lam in LAMBDAS:
        for part in ("early", "late"):
            correct = frame[f"lambda_{lam}/advantage_position/correct_{part}_mean"]
            incorrect = frame[f"lambda_{lam}/advantage_position/incorrect_{part}_mean"]
            gap = correct - incorrect
            for batch, c, w, difference in zip(frame.batch_index, correct, incorrect, gap, strict=True):
                records.append({"lambda": float(lam), "part": part, "batch_index": int(batch),
                                "correct_mean": c, "incorrect_mean": w, "correct_minus_incorrect": difference})
    position = pd.DataFrame(records)
    position.to_csv(args.output_dir / "diagnostic_position_by_batch.csv", index=False)
    summary = position.groupby(["lambda", "part"], as_index=False).agg(
        correct_mean=("correct_mean", "mean"),
        incorrect_mean=("incorrect_mean", "mean"),
        gap_mean=("correct_minus_incorrect", "mean"),
        gap_std=("correct_minus_incorrect", "std"),
        gap_min=("correct_minus_incorrect", "min"),
        gap_max=("correct_minus_incorrect", "max"),
    )
    summary.to_csv(args.output_dir / "diagnostic_position_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"early": "#1f77b4", "late": "#d62728"}
    for part in ("early", "late"):
        subset = summary[summary.part == part]
        ax.errorbar(subset["lambda"], subset.gap_mean, yerr=subset.gap_std,
                    color=colors[part], marker="o", capsize=4, label=f"First/last 100 tokens: {part}")
        individual = position[position.part == part]
        ax.scatter(individual["lambda"], individual.correct_minus_incorrect,
                   color=colors[part], alpha=0.22, s=18)
    ax.set_xlabel("Actor GAE lambda (same fixed-policy batch and critic)")
    ax.set_ylabel("Mean advantage: correct minus incorrect")
    ax.set_xticks([float(x) for x in LAMBDAS])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "diagnostic_advantage_by_position.png", dpi=180)
    plt.close(fig)

    print(f"batches={len(frame)}, score_mean={frame.score_mean.mean():.4f}, "
          f"response_length_mean={frame.response_length_mean.mean():.1f}")
    print(summary.to_string(index=False))
    print("rollout probability gap mean=", frame["training/rollout_probs_diff_mean"].mean())
    print("rollout/actor probability correlation mean=", frame["training/rollout_actor_probs_pearson_corr"].mean())


if __name__ == "__main__":
    main()
