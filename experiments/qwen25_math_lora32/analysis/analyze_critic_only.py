#!/usr/bin/env python3
"""Analyze this experiment's fixed-actor, critic-only VERL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analyze_qwen25_math_ppo import parse_log
except ImportError:
    from analyze_qwen25_math_ppo import parse_log


METRICS = [
    "critic/vf_explained_var",
    "critic/vf_loss",
    "critic/grad_norm",
    "critic/vpred_mean",
    "critic/returns/mean",
    "critic/values/min",
    "critic/values/max",
    "timing_s/step",
]


def rolling_summary(frame, window: int) -> dict[str, float]:
    tail = frame.tail(window)
    return {key: float(tail[key].mean()) for key in METRICS if key in tail}


def first_rolling_threshold(frame, threshold: float, window: int = 20) -> int | None:
    rolling = frame["critic/vf_explained_var"].rolling(window, min_periods=window).mean()
    hit = frame.loc[rolling.ge(threshold), "step"]
    return int(hit.iloc[0]) if len(hit) else None


def add_credit_columns(frame):
    for part in ("early", "late"):
        correct = f"advantage_position/correct_{part}_mean"
        incorrect = f"advantage_position/incorrect_{part}_mean"
        if correct in frame and incorrect in frame:
            frame[f"credit_gap/{part}"] = frame[correct] - frame[incorrect]
    return frame


def make_plots(frame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("critic/vf_explained_var", "Explained variance", False),
        ("critic/vf_loss", "Value loss", True),
        ("critic/grad_norm", "Gradient norm", True),
        ("critic/vpred_mean", "Mean value / target", False),
    ]
    for ax, (key, title, log_y) in zip(axes.flat, panels, strict=True):
        ax.plot(frame.step, frame[key], alpha=0.18, linewidth=0.7, label="per batch")
        ax.plot(frame.step, frame[key].rolling(20, min_periods=1).mean(), label="20-step mean")
        ax.plot(frame.step, frame[key].rolling(50, min_periods=1).mean(), label="50-step mean")
        if key == "critic/vpred_mean":
            ax.plot(
                frame.step,
                frame["critic/returns/mean"].rolling(20, min_periods=1).mean(),
                label="target mean (20-step)",
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Critic optimizer step")
    fig.tight_layout()
    fig.savefig(output / "critic_learning.png", dpi=170)
    plt.close(fig)

    if {"credit_gap/early", "credit_gap/late"}.issubset(frame):
        fig, ax = plt.subplots(figsize=(10, 5))
        for part, color in (("early", "#1f77b4"), ("late", "#d62728")):
            key = f"credit_gap/{part}"
            ax.plot(
                frame.step,
                frame[key].rolling(20, min_periods=1).mean(),
                color=color,
                label=f"Correct - incorrect, {part} 100 tokens",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Critic optimizer step")
        ax.set_ylabel("Whitened advantage gap")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "credit_gap_by_position.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.step, frame["timing_s/step"].rolling(20, min_periods=1).mean())
    ax.set_xlabel("Critic optimizer step")
    ax.set_ylabel("Seconds per step (20-step mean)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "step_time.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = add_credit_columns(parse_log(args.log, warmup_steps=1_000_000))
    frame.to_csv(args.output / "critic_only_metrics.csv", index=False)

    available = [key for key in METRICS if key in frame]
    epoch_summary = frame.groupby("training/epoch")[available].agg(["count", "mean", "std", "min", "max"])
    epoch_summary.columns = [f"{metric}/{stat}" for metric, stat in epoch_summary.columns]
    epoch_summary.to_csv(args.output / "critic_only_epoch_summary.csv")

    ev = frame["critic/vf_explained_var"]
    rolling50 = ev.rolling(50, min_periods=50).mean()
    best_index = rolling50.idxmax() if rolling50.notna().any() else ev.idxmax()
    slope_tail = frame.tail(min(100, len(frame)))
    summary = {
        "logged_steps": int(len(frame)),
        "last_step": int(frame.step.max()),
        "rolling_20": rolling_summary(frame, 20),
        "rolling_50": rolling_summary(frame, 50),
        "best_rolling_50_ev": float(rolling50.loc[best_index]) if rolling50.notna().any() else None,
        "best_rolling_50_ev_step": int(frame.loc[best_index, "step"]),
        "ev_slope_last_100_steps_per_100": float(
            np.polyfit(slope_tail.step, slope_tail["critic/vf_explained_var"], 1)[0] * 100
        ),
        "first_rolling_20_ev_threshold_step": {
            str(threshold): first_rolling_threshold(frame, threshold) for threshold in (0.0, 0.2, 0.3, 0.4, 0.5)
        },
    }
    (args.output / "critic_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(frame, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
