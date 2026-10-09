# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Collect fixed-policy rollouts for offline critic pretraining.

The output is one parquet row per trajectory. Exact prompt/response token ids
are persisted so value pretraining uses the same token boundaries as rollout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from verl.utils.reward_score import default_compute_score


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _load_reward_fn(path: str | None, name: str) -> Callable[..., Any]:
    if path is None:
        return default_compute_score
    spec = importlib.util.spec_from_file_location("verl_custom_rollout_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reward module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, name, None)
    if not callable(fn):
        raise AttributeError(f"{path} does not define callable {name!r}")
    return fn


def _render_prompt(tokenizer, prompt: Any, add_generation_prompt: bool) -> str:
    prompt = _plain(prompt)
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        raise TypeError(f"Expected prompt to be a string or chat message list, got {type(prompt).__name__}")
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _score(reward_fn: Callable[..., Any], row: dict[str, Any], response: str) -> tuple[float, dict[str, Any]]:
    reward_model = _plain(row.get("reward_model") or {})
    result = reward_fn(
        data_source=row.get("data_source"),
        solution_str=response,
        ground_truth=reward_model.get("ground_truth"),
        extra_info=_plain(row.get("extra_info") or {}),
    )
    if isinstance(result, dict):
        details = _plain(result)
        for key in ("score", "reward", "accuracy"):
            if key in result:
                return float(result[key]), details
        raise ValueError(f"Reward dictionary has no scalar score/reward/accuracy field: {result}")
    return float(result), {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--n", type=int, default=1, help="Responses sampled per prompt.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of unique prompts.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-overlong-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-adapter-path")
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--reward-function-path")
    parser.add_argument("--reward-function-name", default="compute_score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n < 1:
        raise ValueError("--n must be at least 1")

    frames = [pd.read_parquet(path) for path in args.input_files]
    frame = pd.concat(frames, ignore_index=True)
    if args.max_samples >= 0:
        frame = frame.iloc[: args.max_samples]
    rows = [_plain(row) for row in frame.to_dict(orient="records")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prepared: list[tuple[int, dict[str, Any], str, list[int]]] = []
    skipped = 0
    for index, row in enumerate(rows):
        text = _render_prompt(tokenizer, row[args.prompt_key], args.add_generation_prompt)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_length:
            if args.skip_overlong_prompts:
                skipped += 1
                continue
            raise ValueError(f"Prompt at row {index} has {len(prompt_ids)} tokens, exceeding {args.max_prompt_length}")
        prepared.append((index, row, text, prompt_ids))

    if not prepared:
        raise ValueError("No prompts remain after loading and length filtering")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_lora=args.lora_adapter_path is not None,
        max_lora_rank=args.max_lora_rank if args.lora_adapter_path is not None else 16,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )
    lora_request = None
    if args.lora_adapter_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("policy", 1, args.lora_adapter_path)

    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start : start + args.batch_size]
        request_outputs = llm.generate(
            [item[2] for item in batch],
            sampling,
            lora_request=lora_request,
            use_tqdm=True,
        )
        if len(request_outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(request_outputs)} outputs for {len(batch)} prompts")
        for (source_index, row, prompt_text, fallback_prompt_ids), request_output in zip(
            batch, request_outputs, strict=True
        ):
            prompt_ids = list(request_output.prompt_token_ids or fallback_prompt_ids)
            for sample_index, completion in enumerate(request_output.outputs):
                response = completion.text
                reward, reward_details = _score(reward_fn, row, response)
                output_rows.append(
                    {
                        "trajectory_id": str(uuid.uuid4()),
                        "prompt_index": source_index,
                        "sample_index": sample_index,
                        "data_source": row.get("data_source"),
                        "prompt": row.get(args.prompt_key),
                        "prompt_text": prompt_text,
                        "response": response,
                        "prompt_ids": prompt_ids,
                        "response_ids": list(completion.token_ids),
                        "reward": reward,
                        "reward_details_json": json.dumps(reward_details, ensure_ascii=False),
                        "reward_model": row.get("reward_model"),
                        "extra_info": row.get("extra_info"),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": _plain(completion.stop_reason),
                        "policy_model_path": args.model_path,
                        "policy_lora_adapter_path": args.lora_adapter_path,
                        "sampling": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "max_response_length": args.max_response_length,
                            "seed": args.seed,
                        },
                    }
                )

    if not output_rows:
        raise RuntimeError("The rollout engine returned no trajectories")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary_path)
    os.replace(temporary_path, output_path)

    summary = {
        "unique_prompts": len(prepared),
        "trajectories": len(output_rows),
        "skipped_overlong_prompts": skipped,
        "mean_reward": sum(row["reward"] for row in output_rows) / max(len(output_rows), 1),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
