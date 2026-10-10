"""The online critic adapter must match the merger format and reload through PEFT."""

import json
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForTokenClassification

from verl.model_merger.base_model_merger import BaseModelMerger
from verl.workers import fsdp_workers


class _TestMerger(BaseModelMerger):
    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        pass


def _tiny_critic():
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_labels=1,
    )
    return Qwen2ForTokenClassification(config)


def test_online_critic_adapter_matches_merger_and_reloads(tmp_path, monkeypatch):
    torch.manual_seed(3)
    base_model = _tiny_critic()
    base_state = {name: value.clone() for name, value in base_model.state_dict().items()}
    critic = get_peft_model(
        base_model,
        LoraConfig(task_type=TaskType.TOKEN_CLS, r=2, lora_alpha=4, target_modules=["q_proj"], bias="none"),
    )
    with torch.no_grad():
        for name, parameter in critic.named_parameters():
            if "lora_A" in name:
                parameter.fill_(0.125)
            elif "lora_B" in name:
                parameter.fill_(0.25)
            elif ".modules_to_save.default." in name:
                parameter.fill_(0.375)
    critic.eval()
    inputs = torch.tensor([[1, 2, 3]])
    expected_logits = critic(inputs).logits.detach()
    raw_state = {name: value.detach().clone() for name, value in critic.state_dict().items()}

    monkeypatch.setattr(fsdp_workers, "get_fsdp_full_state_dict", lambda *args, **kwargs: raw_state.copy())
    monkeypatch.setattr(fsdp_workers.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_workers.dist, "broadcast_object_list", lambda errors, src: None)
    live_dir = tmp_path / "live"
    fsdp_workers._save_critic_lora_adapter(SimpleNamespace(peft_config=critic.peft_config), str(live_dir))

    merger_dir = tmp_path / "merger"
    merger_ckpt = tmp_path / "merger_ckpt"
    merger_ckpt.mkdir()
    (merger_ckpt / "lora_train_meta.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "task_type": "TOKEN_CLS"}))
    merger = object.__new__(_TestMerger)
    merger.config = SimpleNamespace(
        local_dir=str(merger_ckpt), target_dir=str(merger_dir), lora_alpha=None, lora_task_type=None
    )
    merger.save_lora_adapter({name: value.bfloat16() for name, value in raw_state.items()})

    live_adapter = live_dir / "lora_adapter"
    merged_adapter = merger_dir / "lora_adapter"
    assert json.loads((live_adapter / "adapter_config.json").read_text()) == json.loads(
        (merged_adapter / "adapter_config.json").read_text()
    )
    live_weights = load_file(str(live_adapter / "adapter_model.safetensors"))
    merged_weights = load_file(str(merged_adapter / "adapter_model.safetensors"))
    assert live_weights.keys() == merged_weights.keys()
    assert any("score.weight" in name for name in live_weights)
    for name in live_weights:
        assert live_weights[name].dtype == torch.float32
        assert merged_weights[name].dtype == torch.bfloat16
        torch.testing.assert_close(live_weights[name].bfloat16(), merged_weights[name], rtol=0, atol=0)

    restored_base = _tiny_critic()
    restored_base.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(restored_base, str(live_adapter), is_trainable=True).eval()
    torch.testing.assert_close(restored(inputs).logits, expected_logits, rtol=0, atol=0)


def test_online_critic_adapter_rejects_lossy_source_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fsdp_workers,
        "get_fsdp_full_state_dict",
        lambda *args, **kwargs: {
            "base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 4, dtype=torch.bfloat16),
            "base_model.model.q_proj.lora_B.default.weight": torch.ones(4, 2, dtype=torch.bfloat16),
            "base_model.model.score.modules_to_save.default.weight": torch.ones(1, 4, dtype=torch.bfloat16),
        },
    )
    monkeypatch.setattr(fsdp_workers.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_workers.dist, "broadcast_object_list", lambda errors, src: None)
    config = SimpleNamespace(r=2, lora_alpha=4, task_type=TaskType.TOKEN_CLS)
    with pytest.raises(RuntimeError, match="must be torch.float32"):
        fsdp_workers._save_critic_lora_adapter(SimpleNamespace(peft_config={"default": config}), str(tmp_path))
    assert not (tmp_path / "lora_adapter").exists()
