# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Offline Monte-Carlo critic pretraining for fixed-policy rollouts.

Launch on one GPU with ``python -m verl.experimental.critic_pretraining.pretrain``
or on multiple GPUs with ``torchrun --nproc_per_node=N
-m verl.experimental.critic_pretraining.pretrain``.
The final export is directly consumable by verl PPO as either a full Hugging
Face value model or a PEFT adapter plus trainable score head.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, get_scheduler


def monte_carlo_targets(reward: float, response_length: int, gamma: float = 1.0) -> list[float]:
    """Return terminal-reward Monte-Carlo targets for every response action."""
    if response_length < 1:
        raise ValueError("response_length must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return [float(reward) * gamma ** (response_length - 1 - index) for index in range(response_length)]


class CriticRolloutDataset(Dataset):
    def __init__(self, files: list[str], max_samples: int = -1):
        frames = [pd.read_parquet(path, columns=["prompt_ids", "response_ids", "reward"]) for path in files]
        frame = pd.concat(frames, ignore_index=True)
        if max_samples >= 0:
            frame = frame.iloc[:max_samples]
        self.rows = frame.to_dict(orient="records")
        if not self.rows:
            raise ValueError("The critic pretraining dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "prompt_ids": [int(token) for token in row["prompt_ids"]],
            "response_ids": [int(token) for token in row["response_ids"]],
            "reward": float(row["reward"]),
        }


class CriticCollator:
    """Right-pad sequences and align V(s_t) targets with response actions."""

    def __init__(self, pad_token_id: int, max_seq_length: int, gamma: float):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.gamma = gamma

    def _truncate(self, prompt_ids: list[int], response_ids: list[int]) -> tuple[list[int], list[int]]:
        if not prompt_ids:
            raise ValueError("A rollout contains an empty prompt")
        if not response_ids:
            raise ValueError("A rollout contains an empty response")
        overflow = len(prompt_ids) + len(response_ids) - self.max_seq_length
        if overflow > 0:
            removable_prompt = max(len(prompt_ids) - 1, 0)
            trim_prompt = min(overflow, removable_prompt)
            prompt_ids = prompt_ids[trim_prompt:]
            overflow -= trim_prompt
        if overflow > 0:
            response_ids = response_ids[: len(response_ids) - overflow]
        if not response_ids:
            raise ValueError("max_seq_length leaves no response tokens")
        return prompt_ids, response_ids

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prepared = []
        max_length = 0
        for feature in features:
            prompt_ids, response_ids = self._truncate(
                list(feature["prompt_ids"]),
                list(feature["response_ids"]),
            )
            sequence = prompt_ids + response_ids
            targets = monte_carlo_targets(feature["reward"], len(response_ids), self.gamma)
            prepared.append((prompt_ids, response_ids, sequence, targets))
            max_length = max(max_length, len(sequence))

        batch_size = len(prepared)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        value_targets = torch.zeros((batch_size, max_length), dtype=torch.float32)
        value_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

        for row, (prompt_ids, response_ids, sequence, targets) in enumerate(prepared):
            sequence_length = len(sequence)
            input_ids[row, :sequence_length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :sequence_length] = 1
            # The value for response action j is emitted at the preceding token.
            start = len(prompt_ids) - 1
            stop = start + len(response_ids)
            value_targets[row, start:stop] = torch.tensor(targets, dtype=torch.float32)
            value_mask[row, start:stop] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "value_targets": value_targets,
            "value_mask": value_mask,
        }


def explained_variance_statistics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected_predictions = predictions.float()[mask]
    selected_targets = targets.float()[mask]
    residual = selected_targets - selected_predictions
    return torch.stack(
        (
            residual.sum(),
            residual.square().sum(),
            selected_targets.sum(),
            selected_targets.square().sum(),
            torch.tensor(float(selected_targets.numel()), device=predictions.device),
        )
    )


def explained_variance_from_statistics(statistics: torch.Tensor) -> float:
    residual_sum, residual_squared_sum, target_sum, target_squared_sum, count = (float(item) for item in statistics)
    if count == 0:
        return float("nan")
    residual_variance = residual_squared_sum / count - (residual_sum / count) ** 2
    target_variance = target_squared_sum / count - (target_sum / count) ** 2
    if target_variance <= 1e-12:
        return 0.0
    return 1.0 - residual_variance / target_variance


class TrainingProgress:
    """Small checkpointable object for exact optimizer-step resumption."""

    def __init__(self) -> None:
        self.completed_steps = 0

    def state_dict(self) -> dict[str, int]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.completed_steps = int(state_dict["completed_steps"])


def _parse_modules(value: str) -> list[str] | str:
    if value == "all-linear":
        return value
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target-modules must be all-linear or a comma-separated list")
    return modules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base or reward-model initialization.")
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", default="constant_with_warmup")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--model-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="auto uses FP32 for full training and the mixed-precision dtype for LoRA.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0, help="Zero selects full-parameter training.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--log-every-steps", type=int, default=1)
    return parser


def _resolved_model_dtype(args) -> str:
    if args.model_dtype != "auto":
        return args.model_dtype
    if args.lora_rank > 0 and args.mixed_precision != "no":
        return {"fp16": "float16", "bf16": "bfloat16"}[args.mixed_precision]
    return "float32"


def _build_model(args):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    config.num_labels = 1
    config.classifier_dropout = 0.0
    config.hidden_dropout = 0.0
    config.summary_dropout_prob = 0.0
    dtype_name = _resolved_model_dtype(args)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        config=config,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora_rank > 0:
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=_parse_modules(args.target_modules),
                bias="none",
            ),
        )
    return model


def _save_export(accelerator: Accelerator, model, tokenizer, args, completed_steps: int) -> None:
    accelerator.wait_for_everyone()
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        ppo_config = (
            {
                "critic.model.path": args.model_path,
                "critic.model.lora_adapter_path": str(output_dir),
                "critic.model.lora_rank": args.lora_rank,
                "critic.model.lora_alpha": args.lora_alpha,
                "critic.model.target_modules": _parse_modules(args.target_modules),
            }
            if args.lora_rank > 0
            else {
                "critic.model.path": str(output_dir),
                "critic.model.lora_rank": 0,
            }
        )
        metadata = {
            "format_version": 1,
            "model_type": "verl_hf_token_value_model",
            "base_model_path": args.model_path,
            "training_mode": "lora" if args.lora_rank > 0 else "full",
            "completed_steps": completed_steps,
            "gamma": args.gamma,
            "model_dtype": _resolved_model_dtype(args),
            "ppo_config": ppo_config,
        }
        with open(output_dir / "verl_critic_pretraining.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
    accelerator.wait_for_everyone()


def main() -> None:
    args = build_parser().parse_args()
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be non-negative")
    if args.max_steps == 0:
        raise ValueError("--max-steps must be negative (unlimited) or positive")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = CriticRolloutDataset(args.train_files, max_samples=args.max_samples)
    collator = CriticCollator(tokenizer.pad_token_id, args.max_seq_length, args.gamma)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = _build_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * updates_per_epoch
    scheduler = get_scheduler(
        args.scheduler,
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    progress = TrainingProgress()
    accelerator.register_for_checkpointing(progress)
    num_epochs = max(args.epochs, math.ceil(total_steps / updates_per_epoch))

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)

    model.train()
    optimizer.zero_grad()
    completed_steps = progress.completed_steps
    for epoch in range(num_epochs):
        if completed_steps >= total_steps:
            break
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                predictions = outputs.logits.squeeze(-1).float()
                mask = batch["value_mask"]
                targets = batch["value_targets"]
                loss = (predictions[mask] - targets[mask]).square().mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                progress.completed_steps = completed_steps
                statistics = explained_variance_statistics(predictions.detach(), targets, mask)
                statistics = accelerator.reduce(statistics, reduction="sum")
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                if accelerator.is_main_process and completed_steps % args.log_every_steps == 0:
                    print(
                        json.dumps(
                            {
                                "step": completed_steps,
                                "epoch": epoch,
                                "value_loss": float(reduced_loss),
                                "explained_variance": explained_variance_from_statistics(statistics),
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )
                    )
                if args.save_every_steps > 0 and completed_steps % args.save_every_steps == 0:
                    checkpoint_dir = Path(args.output_dir) / f"training_state_step_{completed_steps}"
                    accelerator.save_state(checkpoint_dir)
                if completed_steps >= total_steps:
                    break
        if completed_steps >= total_steps:
            break

    _save_export(accelerator, model, tokenizer, args, completed_steps)
    if accelerator.is_main_process:
        print(json.dumps({"completed_steps": completed_steps, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
