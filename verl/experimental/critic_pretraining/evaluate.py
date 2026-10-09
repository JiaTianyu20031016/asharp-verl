"""Evaluate a token-value critic on fixed rollout trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from verl.experimental.ppo_diagnostics.advantage import summarize_position_advantages
from verl.trainer.ppo import core_algos

from .pretrain import CriticCollator


class EvaluationRolloutDataset(Dataset):
    def __init__(self, files: list[str], max_samples: int = -1):
        columns = ["prompt_ids", "response_ids", "reward", "prompt_index", "sample_index"]
        frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
        if max_samples >= 0:
            frame = frame.iloc[:max_samples]
        self.rows = frame.to_dict(orient="records")
        if not self.rows:
            raise ValueError("The evaluation rollout dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "prompt_ids": [int(token) for token in row["prompt_ids"]],
            "response_ids": [int(token) for token in row["response_ids"]],
            "reward": float(row["reward"]),
            "prompt_index": int(row["prompt_index"]),
            "sample_index": int(row["sample_index"]),
        }


class EvaluationCollator:
    def __init__(self, pad_token_id: int, max_seq_length: int):
        self.value_collator = CriticCollator(pad_token_id, max_seq_length, gamma=1.0)

    def __call__(self, features: list[dict]) -> dict:
        batch = self.value_collator(features)
        batch["prompt_index"] = torch.tensor([item["prompt_index"] for item in features], dtype=torch.long)
        batch["sample_index"] = torch.tensor([item["sample_index"] for item in features], dtype=torch.long)
        return batch


def binary_auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.astype(bool)
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(predictions).rank(method="average").to_numpy()
    return float((ranks[targets].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    residual = targets - predictions
    target_variance = float(targets.var())
    explained_variance = 0.0 if target_variance <= 1e-12 else 1.0 - float(residual.var()) / target_variance
    correlation = float(np.corrcoef(predictions, targets)[0, 1]) if predictions.std() and targets.std() else float("nan")
    return {
        "count": int(len(targets)),
        "explained_variance": explained_variance,
        "mse": float(np.mean((predictions - targets) ** 2)),
        "mae": float(np.mean(np.abs(predictions - targets))),
        "bias_prediction_minus_target": float(np.mean(predictions - targets)),
        "pearson_correlation": correlation,
        "auc": binary_auc(predictions, targets),
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_outside_0_1_fraction": float(np.mean((predictions < 0) | (predictions > 1))),
        "clipped_brier_score": float(np.mean((np.clip(predictions, 0, 1) - targets) ** 2)),
    }


def _load_model(base_model: str, adapter: str | None, dtype: torch.dtype, device: torch.device):
    config = AutoConfig.from_pretrained(base_model)
    config.num_labels = 1
    config.classifier_dropout = 0.0
    config.hidden_dropout = 0.0
    config.summary_dropout_prob = 0.0
    model = AutoModelForTokenClassification.from_pretrained(base_model, config=config, dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model


def _calibration_frame(predictions: np.ndarray, targets: np.ndarray, bins: int = 10) -> pd.DataFrame:
    clipped = np.clip(predictions, 0, 1)
    indices = np.minimum((clipped * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": int(selected.sum()),
                "prediction_mean_raw": float(predictions[selected].mean()),
                "prediction_mean_clipped": float(clipped[selected].mean()),
                "target_mean": float(targets[selected].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_gae_lambdas(
    value_sequences: list[np.ndarray],
    rewards: list[float],
    lambdas: list[float],
    window: int = 100,
) -> dict[str, dict[str, float]]:
    """Apply VERL's exact batched GAE/whitening to fixed critic trajectories."""
    if len(value_sequences) != len(rewards):
        raise ValueError("value_sequences and rewards must have the same length")
    if not value_sequences:
        raise ValueError("at least one trajectory is required")
    if any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("GAE lambdas must be in [0, 1]")

    lengths = torch.tensor([len(values) for values in value_sequences], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("all trajectories must contain at least one response value")
    width = int(lengths.max())
    values = torch.zeros((len(value_sequences), width), dtype=torch.float32)
    response_mask = torch.zeros_like(values)
    token_level_rewards = torch.zeros_like(values)
    token_level_scores = torch.zeros_like(values)
    for row, (sequence, reward) in enumerate(zip(value_sequences, rewards, strict=True)):
        length = len(sequence)
        values[row, :length] = torch.from_numpy(np.asarray(sequence, dtype=np.float32))
        response_mask[row, :length] = 1
        token_level_rewards[row, length - 1] = reward
        token_level_scores[row, length - 1] = reward

    result = {}
    for lam in lambdas:
        advantages, _ = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=1.0,
            lam=lam,
        )
        summary = summarize_position_advantages(advantages, response_mask, token_level_scores, window=window)
        for position in ("early", "late"):
            correct = summary[f"advantage_position/correct_{position}_mean"]
            incorrect = summary[f"advantage_position/incorrect_{position}_mean"]
            summary[f"credit_gap/{position}"] = correct - incorrect
        result[str(lam)] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--rollout-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--advantage-lambdas",
        nargs="*",
        type=float,
        default=[],
        help="Also compare VERL GAE advantages for these lambdas on the fixed trajectories.",
    )
    parser.add_argument("--advantage-window", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = EvaluationRolloutDataset(args.rollout_files, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=EvaluationCollator(tokenizer.pad_token_id, args.max_seq_length),
    )
    model = _load_model(args.base_model, args.adapter, dtype, device)

    token_rows = []
    trajectory_rows = []
    value_sequences = []
    trajectory_rewards = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.squeeze(-1)
            for row in range(len(input_ids)):
                mask = batch["value_mask"][row].to(device)
                values = logits[row][mask].float().cpu().numpy()
                targets = batch["value_targets"][row][batch["value_mask"][row]].numpy()
                length = len(values)
                prompt_index = int(batch["prompt_index"][row])
                sample_index = int(batch["sample_index"][row])
                reward = float(targets[-1])
                value_sequences.append(values)
                trajectory_rewards.append(reward)
                trajectory_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "reward": reward,
                        "response_length": length,
                        "value_start": float(values[0]),
                        "value_middle": float(values[length // 2]),
                        "value_end": float(values[-1]),
                        "value_mean": float(values.mean()),
                    }
                )
                token_rows.append(
                    pd.DataFrame(
                        {
                            "prediction": values,
                            "target": targets,
                            "absolute_position": np.arange(length),
                            "from_end": np.arange(length - 1, -1, -1),
                            "relative_bin": np.minimum(np.arange(length) * 10 // length, 9),
                        }
                    )
                )

    tokens = pd.concat(token_rows, ignore_index=True)
    trajectories = pd.DataFrame(trajectory_rows).sort_values(["prompt_index", "sample_index"])
    predictions = tokens.prediction.to_numpy()
    targets = tokens.target.to_numpy()
    result = {"all_tokens": regression_metrics(predictions, targets)}
    for name, selected in {
        "first_100_tokens": tokens.absolute_position.lt(100),
        "last_100_tokens": tokens.from_end.lt(100),
    }.items():
        result[name] = regression_metrics(tokens.loc[selected, "prediction"], tokens.loc[selected, "target"])
    for point in ("start", "middle", "end", "mean"):
        result[f"trajectory_{point}"] = regression_metrics(
            trajectories[f"value_{point}"].to_numpy(), trajectories.reward.to_numpy()
        )

    prompt_start = trajectories.groupby("prompt_index").agg(
        predicted_success=("value_start", "mean"), empirical_success=("reward", "mean")
    )
    result["prompt_start_vs_empirical_pass1"] = regression_metrics(
        prompt_start.predicted_success.to_numpy(), prompt_start.empirical_success.to_numpy()
    )

    position_rows = []
    for relative_bin, group in tokens.groupby("relative_bin"):
        position_rows.append({"relative_bin": int(relative_bin), **regression_metrics(group.prediction, group.target)})
    pd.DataFrame(position_rows).to_csv(output / "metrics_by_relative_position.csv", index=False)
    _calibration_frame(predictions, targets).to_csv(output / "token_calibration.csv", index=False)
    trajectories.to_parquet(output / "trajectory_values.parquet", index=False)
    prompt_start.reset_index().to_csv(output / "prompt_start_values.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    if args.advantage_lambdas:
        advantage_result = compare_gae_lambdas(
            value_sequences,
            trajectory_rewards,
            args.advantage_lambdas,
            window=args.advantage_window,
        )
        (output / "advantage_by_lambda.json").write_text(
            json.dumps(advantage_result, indent=2, allow_nan=True) + "\n"
        )
        print(json.dumps({"advantage_by_lambda": advantage_result}, indent=2, allow_nan=True))
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
