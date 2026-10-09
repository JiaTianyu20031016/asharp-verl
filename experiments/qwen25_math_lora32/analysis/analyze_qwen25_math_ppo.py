#!/usr/bin/env python3
"""Summarize this experiment's MATH PPO logs and draw comparable trends."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\bstep:(\d+) - ")
DEFAULT_LOG_DIR = Path("/data1/jty/logs/ppo_lora32_math")
RUNS = {
    "baseline": ("qwen25-math-15b-math-ppo-lora32-n4-seed42.log", 0),
    "critic_warmup_231_steps": (
        "qwen25-math-15b-math-ppo-lora32-n4-criticwarmup232-seed42.log",
        231,
    ),
}


def parse_log(path: Path, warmup_steps: int) -> pd.DataFrame:
    rows: dict[int, dict[str, float]] = {}
    for line in path.open(errors="replace"):
        line = ANSI.sub("", line)
        match = STEP.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        row = rows.setdefault(step, {"step": step})
        for part in line[match.start() :].split(" - "):
            key, separator, raw = part.partition(":")
            if not separator:
                continue
            raw = raw.strip()
            if raw.startswith("np.") and "(" in raw:
                raw = raw.split("(", 1)[1].split(")", 1)[0]
            try:
                row[key] = float(raw)
            except ValueError:
                continue
    frame = pd.DataFrame(rows.values()).sort_values("step").reset_index(drop=True)
    frame["actor_updates"] = (frame["step"] - warmup_steps).clip(lower=0)
    return frame


def plot_line(ax, frame, key, label, *, x="step", smooth=1, **kwargs):
    if key not in frame:
        return
    data = frame[[x, key]].dropna()
    if data.empty:
        return
    y = data[key].rolling(smooth, min_periods=1).mean() if smooth > 1 else data[key]
    ax.plot(data[x], y, label=label, **kwargs)


def plot_validation(ax, frame, key, label, warmup_steps, **kwargs):
    if key not in frame:
        return
    data = frame[["step", "actor_updates", key]].dropna()
    if warmup_steps:
        data = data[(data["step"] == 0) | (data["step"] >= warmup_steps)]
        # The last frozen-actor validation is the control point at x=0.
        data = data.drop_duplicates("actor_updates", keep="last")
    ax.plot(data["actor_updates"], data[key], marker="o", label=label, **kwargs)


def make_plots(frames: dict[str, pd.DataFrame], output: Path):
    colors = {"baseline": "#1f77b4", "critic_warmup_231_steps": "#d62728"}
    labels = {"baseline": "No warm-up", "critic_warmup_231_steps": "Critic warm-up"}
    accuracy = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@4"
    bootstrap_best4 = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/best@4/mean"

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, frame in frames.items():
        active = frame[frame["actor_updates"] > 0]
        plot_line(axes[0], active, "critic/score/mean", labels[name] + " train (20-step mean)",
                  x="actor_updates", smooth=20, color=colors[name], linestyle="--", alpha=0.65)
        plot_validation(axes[0], frame, accuracy, labels[name] + " validation pass@1",
                        RUNS[name][1], color=colors[name])
        plot_validation(axes[1], frame, bootstrap_best4, labels[name] + " bootstrap best-of-4",
                        RUNS[name][1], color=colors[name])
    axes[0].set_ylabel("Accuracy / reward")
    axes[1].set_ylabel("Bootstrap best-of-4 score")
    axes[1].set_xlabel("Actor optimizer updates")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_actor_updates.png", dpi=170)
    plt.close(fig)

    panels = [
        ("critic/vf_explained_var", "Critic explained variance"),
        ("critic/vf_loss", "Critic value loss"),
        ("critic/vpred_mean", "Predicted value mean"),
        ("critic/returns/mean", "Critic target mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        for name, frame in frames.items():
            plot_line(ax, frame, key, labels[name], smooth=15, color=colors[name])
        ax.axvline(231, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Global training step")
    fig.tight_layout()
    fig.savefig(output / "critic_dynamics.png", dpi=170)
    plt.close(fig)

    panels = [
        ("actor/grad_norm", "Actor gradient norm"),
        ("actor/pg_loss", "Actor policy loss"),
        ("actor/entropy", "Actor entropy"),
        ("response_length/mean", "Response length"),
        ("response_length/clip_ratio", "Response at max length"),
        ("actor/lr", "Actor learning rate"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        for name, frame in frames.items():
            active = frame[frame["actor_updates"] > 0]
            plot_line(ax, active, key, labels[name], x="actor_updates", smooth=15,
                      color=colors[name])
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[2]:
        ax.set_xlabel("Actor optimizer updates")
    fig.tight_layout()
    fig.savefig(output / "actor_and_rollout.png", dpi=170)
    plt.close(fig)

    panels = [
        ("timing_s/step", "Step time (seconds)"),
        ("timing_s/gen", "Generation time (seconds)"),
        ("timing_s/update_critic", "Critic update time (seconds)"),
        ("timing_s/update_actor", "Actor update time (seconds)"),
        ("perf/max_memory_allocated_gb", "Peak allocated GPU memory (GB)"),
        ("perf/throughput", "Throughput (tokens/s)")
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        for name, frame in frames.items():
            plot_line(ax, frame, key, labels[name], smooth=15, color=colors[name])
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[2]:
        ax.set_xlabel("Global training step")
    fig.tight_layout()
    fig.savefig(output / "timing_and_memory.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=Path("/data1/jty/analysis/ppo_lora32_math"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {}
    all_summaries = []
    validations = []
    for name, (filename, warmup_steps) in RUNS.items():
        frame = parse_log(args.log_dir / filename, warmup_steps)
        frame.to_csv(args.output / f"{name}_metrics.csv", index=False)
        frames[name] = frame
        for key in frame.select_dtypes(include="number").columns:
            values = frame[key].dropna()
            if values.empty:
                continue
            all_summaries.append({"run": name, "metric": key, "count": len(values),
                                  "mean": values.mean(), "std": values.std(),
                                  "min": values.min(), "max": values.max(),
                                  "first": values.iloc[0], "last": values.iloc[-1]})
        val_columns = [key for key in frame if key.startswith("val-") and "/acc/" in key]
        for _, row in frame.loc[frame[val_columns].notna().any(axis=1)].iterrows():
            validations.append({"run": name, "step": int(row["step"]),
                                "actor_updates": int(row["actor_updates"]),
                                "sampled_pass1": row.get("val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@4"),
                                "bootstrap_best_of_4": row.get("val-core/DigitalLearningGmbH/MATH-lighteval/acc/best@4/mean")})
    pd.DataFrame(all_summaries).to_csv(args.output / "all_metric_summary.csv", index=False)
    pd.DataFrame(validations).to_csv(args.output / "validation_metrics.csv", index=False)
    make_plots(frames, args.output)
    for name, frame in frames.items():
        print(f"{name}: {len(frame)} logged steps, {int(frame.step.max())} maximum global step")
    print(f"Analysis written to {args.output}")


if __name__ == "__main__":
    main()
