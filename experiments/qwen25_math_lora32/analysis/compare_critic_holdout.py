#!/usr/bin/env python3
"""Compare this experiment's fixed-rollout critic checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCOPES = (
    "all_tokens",
    "first_100_tokens",
    "last_100_tokens",
    "trajectory_start",
    "trajectory_middle",
    "trajectory_end",
    "prompt_start_vs_empirical_pass1",
)
METRICS = ("explained_variance", "mse", "bias_prediction_minus_target", "auc")


def parse_evaluation(value: str) -> tuple[int, Path]:
    step, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("evaluations must use STEP=PATH")
    return int(step), Path(path)


def load_summary(evaluations: list[tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for step, path in sorted(evaluations):
        metrics = json.loads((path / "metrics.json").read_text())
        row = {"step": step}
        for scope in SCOPES:
            for metric in METRICS:
                row[f"{scope}/{metric}"] = metrics[scope][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_main_metrics(summary: pd.DataFrame, output: Path) -> None:
    labels = {
        "all_tokens": "All tokens",
        "first_100_tokens": "First 100",
        "last_100_tokens": "Last 100",
        "trajectory_start": "Trajectory start",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS, strict=True):
        for scope, label in labels.items():
            ax.plot(summary.step, summary[f"{scope}/{metric}"], marker="o", label=label)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Critic optimizer step")
    fig.tight_layout()
    fig.savefig(output / "holdout_metrics_by_step.png", dpi=170)
    plt.close(fig)


def plot_position(evaluations: list[tuple[int, Path]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for step, path in sorted(evaluations):
        frame = pd.read_csv(path / "metrics_by_relative_position.csv")
        x = (frame.relative_bin + 0.5) / 10
        axes[0].plot(x, frame.explained_variance, marker="o", label=f"step {step}")
        axes[1].plot(x, frame.bias_prediction_minus_target, marker="o", label=f"step {step}")
    axes[0].set_ylabel("Explained variance")
    axes[1].set_ylabel("Prediction - target bias")
    for ax in axes:
        ax.set_xlabel("Relative response position")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "holdout_metrics_by_position.png", dpi=170)
    plt.close(fig)


def plot_calibration(evaluations: list[tuple[int, Path]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="ideal")
    for step, path in sorted(evaluations):
        frame = pd.read_csv(path / "token_calibration.csv")
        ax.plot(frame.prediction_mean_clipped, frame.target_mean, marker="o", label=f"step {step}")
    ax.set_xlabel("Mean clipped prediction")
    ax.set_ylabel("Empirical success rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "holdout_calibration.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", type=parse_evaluation, required=True, dest="evaluations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.evaluations)
    summary.to_csv(args.output / "holdout_checkpoint_summary.csv", index=False)
    plot_main_metrics(summary, args.output)
    plot_position(args.evaluations, args.output)
    plot_calibration(args.evaluations, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
