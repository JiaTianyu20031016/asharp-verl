#!/usr/bin/env python3
"""Compare this experiment's fixed-seed MATH rollouts and empirical pass@4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from verl.utils.reward_score.math_reward import last_boxed_only_string


def summarize(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_parquet(path)
    required = {"prompt_index", "sample_index", "response", "reward", "finish_reason"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if data.duplicated(["prompt_index", "sample_index"]).any():
        raise ValueError(f"{path} has duplicate prompt/sample pairs")
    if not data.groupby("prompt_index").size().eq(4).all():
        raise ValueError(f"{path} does not contain exactly four samples per prompt")
    data["reward"] = data["reward"].astype(float)
    data["has_boxed"] = data["response"].map(lambda text: last_boxed_only_string(text) is not None)
    data["length_stop"] = data["finish_reason"].eq("length")
    by_prompt = data.groupby("prompt_index", sort=True).agg(
        pass1=("reward", "mean"),
        pass4=("reward", "max"),
        boxed_fraction=("has_boxed", "mean"),
        length_fraction=("length_stop", "mean"),
    )
    summary = {
        "prompts": float(len(by_prompt)),
        "responses": float(len(data)),
        "sampled_pass1": float(by_prompt.pass1.mean()),
        "empirical_pass4": float(by_prompt.pass4.mean()),
        "boxed_fraction": float(data.has_boxed.mean()),
        "length_stop_fraction": float(data.length_stop.mean()),
        "boxed_but_zero_fraction": float(((data.has_boxed) & (data.reward == 0)).mean()),
    }
    return by_prompt, summary


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, resamples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[samples].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--step385", type=Path, required=True)
    parser.add_argument("--step462", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"base": args.base, "step385": args.step385, "step462": args.step462}
    per_prompt = {}
    summary_rows = []
    for name, path in paths.items():
        per_prompt[name], summary = summarize(path)
        summary_rows.append({"run": name, **summary})
    common = set.intersection(*(set(frame.index) for frame in per_prompt.values()))
    if not all(set(frame.index) == common for frame in per_prompt.values()):
        raise ValueError("Evaluations do not cover the same prompt indices")
    paired = pd.concat({name: frame for name, frame in per_prompt.items()}, axis=1)
    paired.to_csv(args.output_dir / "fixed_policy_per_prompt.csv")
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "fixed_policy_summary.csv", index=False)
    for row in summary_rows:
        print(row)
    for name in ("step385", "step462"):
        for metric in ("pass1", "pass4"):
            delta = (paired[(name, metric)] - paired[("base", metric)]).to_numpy()
            low, high = bootstrap_mean_ci(delta)
            print(f"{name} - base {metric}: mean={delta.mean():+.4f}, 95% prompt-bootstrap CI=[{low:+.4f}, {high:+.4f}]")
    for metric in ("pass1", "pass4"):
        delta = (paired[("step462", metric)] - paired[("step385", metric)]).to_numpy()
        low, high = bootstrap_mean_ci(delta)
        print(f"step462 - step385 {metric}: mean={delta.mean():+.4f}, 95% prompt-bootstrap CI=[{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
