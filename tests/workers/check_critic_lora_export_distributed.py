"""Manual two-GPU check for live FSDP1/FSDP2 critic LoRA export.

Run with ``python tests/workers/check_critic_lora_export_distributed.py``.
"""

import json
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardedStateDictConfig,
    StateDictType,
    fully_shard,
)
from transformers import Qwen2Config, Qwen2ForTokenClassification

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.workers.fsdp_workers import _save_critic_lora_adapter


def _worker(rank: int, directory: str, strategy: str):
    dist.init_process_group("nccl", init_method=f"file://{directory}/init", rank=rank, world_size=2)
    try:
        torch.cuda.set_device(rank)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_labels=1,
        )
        model = get_peft_model(
            Qwen2ForTokenClassification(config),
            LoraConfig(task_type=TaskType.TOKEN_CLS, r=2, lora_alpha=4, target_modules=["q_proj"], bias="none"),
        )
        if strategy == "fsdp":
            wrap_policy = get_fsdp_wrap_policy(
                model, {"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]}, is_lora=True
            )
            mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
            model = FSDP(model, device_id=rank, auto_wrap_policy=wrap_policy, device_mesh=mesh)
        else:
            model.cuda(rank)
            fully_shard(model)
        _save_critic_lora_adapter(model, directory)
        if strategy == "fsdp":
            with FSDP.state_dict_type(
                model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)
            ):
                torch.save(model.state_dict(), f"{directory}/model_world_size_2_rank_{rank}.pt")
            if rank == 0:
                with open(f"{directory}/fsdp_config.json", "w", encoding="utf-8") as file:
                    json.dump({"world_size": 2}, file)
        if rank == 0:
            weights = load_file(f"{directory}/lora_adapter/adapter_model.safetensors")
            assert weights and all(value.dtype == torch.float32 for value in weights.values())
            assert any("score.weight" in name for name in weights)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    for strategy in ("fsdp", "fsdp2"):
        with tempfile.TemporaryDirectory(prefix=f"critic-lora-{strategy}-", dir="/tmp") as directory:
            mp.spawn(_worker, args=(directory, strategy), nprocs=2)
            if strategy == "fsdp":
                merger = object.__new__(FSDPModelMerger)
                merger.config = SimpleNamespace(
                    local_dir=directory,
                    target_dir=f"{directory}/merged",
                    lora_alpha=4,
                    lora_task_type="TOKEN_CLS",
                )
                world_size = merger._get_world_size()
                rank_zero = merger._load_rank_zero_state_dict(world_size)
                mesh, mesh_names = merger._extract_device_mesh_info(rank_zero, world_size)
                shard_count, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_names)
                state = merger._load_and_merge_state_dicts(world_size, shard_count, mesh_shape, mesh_names)
                merger.save_lora_adapter(state)
                live = load_file(f"{directory}/lora_adapter/adapter_model.safetensors")
                merged = load_file(f"{directory}/merged/lora_adapter/adapter_model.safetensors")
                assert live.keys() == merged.keys()
                for name in live:
                    assert live[name].dtype == torch.float32
                    assert merged[name].dtype == torch.bfloat16
                    torch.testing.assert_close(live[name].bfloat16(), merged[name], rtol=0, atol=0)
                with open(f"{directory}/lora_adapter/adapter_config.json", encoding="utf-8") as file:
                    live_config = json.load(file)
                with open(f"{directory}/merged/lora_adapter/adapter_config.json", encoding="utf-8") as file:
                    merged_config = json.load(file)
                assert live_config == merged_config
        print(f"Two-rank {strategy} critic adapter export passed")
