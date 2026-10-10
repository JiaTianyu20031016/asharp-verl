# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared PEFT adapter export for checkpoint conversion and live critic saves."""

import json
import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch


def save_lora_adapter(
    state_dict: dict[str, torch.Tensor],
    target_dir: str,
    lora_meta: Optional[dict[str, object]] = None,
    *,
    required_dtype: Optional[torch.dtype] = None,
    require_modules_to_save: bool = False,
) -> Optional[str]:
    """Extract PEFT tensors and normalize the remaining base state dict in place."""
    lora_params_names = [name for name in state_dict.keys() if "lora_" in name]

    if len(lora_params_names) == 0:
        return None

    import peft
    from safetensors.torch import save_file

    lora_params = OrderedDict()
    target_modules = set()
    modules_to_save = set()
    lora_key = None

    for name in lora_params_names:
        lora_key = name.replace(".default.weight", ".weight")
        target_modules.add(lora_key.split(".")[-3])
        lora_params[lora_key] = state_dict.pop(name)

    modules_to_save_names = [name for name in state_dict if ".modules_to_save.default." in name]
    for name in modules_to_save_names:
        adapter_key = name.replace(".modules_to_save.default.", ".")
        module_name = name.split(".modules_to_save.default.", 1)[0].split(".")[-1]
        modules_to_save.add(module_name)
        lora_params[adapter_key] = state_dict.pop(name)

    inferred_lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
    lora_rank = inferred_lora_rank
    lora_alpha = 0
    task_type = None

    if lora_meta is not None:
        meta_rank = lora_meta.get("r")
        if meta_rank is not None and meta_rank > 0:
            if meta_rank != inferred_lora_rank:
                warnings.warn(
                    f"LoRA rank mismatch between metadata ({meta_rank}) and adapter weights "
                    f"({inferred_lora_rank}); using metadata rank.",
                    stacklevel=2,
                )
            lora_rank = meta_rank

        meta_alpha = lora_meta.get("lora_alpha")
        if meta_alpha is not None:
            lora_alpha = meta_alpha

        meta_task_type = lora_meta.get("task_type")
        if meta_task_type is not None:
            task_type = meta_task_type

    if lora_alpha == 0:
        warnings.warn(
            "LoRA alpha metadata is missing or equals 0; falling back to lora_alpha=0. "
            "Please verify checkpoint LoRA metadata (lora_train_meta.json).",
            stacklevel=2,
        )

    peft_dict = {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
    }
    if modules_to_save:
        peft_dict["modules_to_save"] = sorted(modules_to_save)
        if task_type is None and modules_to_save.intersection({"classifier", "score"}):
            task_type = "TOKEN_CLS"
    if task_type is not None:
        peft_dict["task_type"] = task_type
    peft_config = peft.LoraConfig(**peft_dict).to_dict()
    for enum_key in ("task_type", "peft_type"):
        enum_value = peft_config.get(enum_key)
        if hasattr(enum_value, "value"):
            peft_config[enum_key] = enum_value.value
    peft_config["target_modules"] = list(peft_config["target_modules"])

    if required_dtype is not None:
        wrong_dtype = {name: str(param.dtype) for name, param in lora_params.items() if param.dtype != required_dtype}
        if wrong_dtype:
            raise ValueError(f"Adapter parameters must be {required_dtype}: {wrong_dtype}")
    if require_modules_to_save and not modules_to_save.intersection({"score", "classifier"}):
        raise ValueError("Critic adapter has no trained score/classifier head in modules_to_save")

    lora_path = os.path.join(target_dir, "lora_adapter")
    os.makedirs(lora_path, exist_ok=True)
    with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(peft_config, f, ensure_ascii=False, indent=4)
    save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

    for name in list(state_dict.keys()):
        key = (
            name.replace("base_model.model.", "")
            .replace(".base_layer.weight", ".weight")
            .replace(".base_layer.bias", ".bias")
            .replace(".original_module.weight", ".weight")
            .replace(".original_module.bias", ".bias")
        )
        state_dict[key] = state_dict.pop(name)

    return lora_path
