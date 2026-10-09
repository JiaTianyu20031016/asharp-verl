#!/usr/bin/env python3
"""Experiment-specific check of LoRA refresh versus inference nondeterminism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--prompts", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    frame = pd.read_parquet(args.data)
    rendered = []
    for prompt in frame.prompt:
        text = tokenizer.apply_chat_template(prompt.tolist(), tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= 512:
            rendered.append(text)
        if len(rendered) >= args.prompts:
            break
    if len(rendered) != args.prompts:
        raise ValueError(f"Only {len(rendered)} valid prompts found")

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", enable_lora=True,
              max_lora_rank=32, gpu_memory_utilization=0.6, enforce_eager=True)
    sampling = SamplingParams(temperature=0, max_tokens=128)

    def generate(path: str):
        result = llm.generate(rendered, sampling, lora_request=LoRARequest("policy", 1, path), use_tqdm=False)
        return [tuple(item.outputs[0].token_ids) for item in result]

    a_first = generate(args.adapter_a)
    a_repeat = generate(args.adapter_a)
    removed_a = llm.llm_engine.remove_lora(1)
    a_reloaded = generate(args.adapter_a)
    removed_a_again = llm.llm_engine.remove_lora(1)
    b = generate(args.adapter_b)
    removed_b = llm.llm_engine.remove_lora(1)
    a_restored = generate(args.adapter_a)

    def changed(left, right):
        return sum(x != y for x, y in zip(left, right, strict=True))

    print(json.dumps({"prompts": len(rendered), "removed_a": removed_a,
                      "removed_a_again": removed_a_again, "removed_b": removed_b,
                      "changed_a_repeat_without_reload": changed(a_first, a_repeat),
                      "changed_a_after_same_adapter_reload": changed(a_first, a_reloaded),
                      "changed_a_to_b": changed(a_first, b),
                      "changed_a_to_a_after_b_reload": changed(a_first, a_restored)}, indent=2))
    if not removed_a or not removed_a_again or not removed_b:
        raise AssertionError("vLLM did not acknowledge a LoRA removal")
    stable_a = [x == y == z == w for x, y, z, w in zip(
        a_first, a_repeat, a_reloaded, a_restored, strict=True
    )]
    changed_b_on_stable = sum(stable and x != y for stable, x, y in zip(stable_a, a_first, b, strict=True))
    print(json.dumps({"stable_a_prompts": sum(stable_a), "changed_b_on_stable_a_prompts": changed_b_on_stable}))


if __name__ == "__main__":
    main()
