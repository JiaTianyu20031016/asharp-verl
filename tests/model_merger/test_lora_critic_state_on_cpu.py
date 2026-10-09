import json
from types import SimpleNamespace

import torch

from transformers import AutoModelForTokenClassification

from verl.model_merger.base_model_merger import BaseModelMerger


class _TestMerger(BaseModelMerger):
    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        pass


def test_save_lora_adapter_keeps_trained_score_head(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    config = SimpleNamespace(
        local_dir=str(checkpoint),
        target_dir=str(output),
        lora_alpha=2,
        lora_task_type="TOKEN_CLS",
        is_value_model=True,
    )
    merger = object.__new__(_TestMerger)
    merger.config = config
    state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.ones(4, 2),
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros(4, 4),
        "base_model.model.score.modules_to_save.default.weight": torch.full((1, 4), 3.0),
        "base_model.model.score.modules_to_save.default.bias": torch.full((1,), 4.0),
        "base_model.model.score.original_module.weight": torch.zeros(1, 4),
        "base_model.model.score.original_module.bias": torch.zeros(1),
    }

    adapter_path = merger.save_lora_adapter(state)

    from safetensors.torch import load_file

    adapter = load_file(str(output / "lora_adapter" / "adapter_model.safetensors"))
    adapter_config = json.loads((output / "lora_adapter" / "adapter_config.json").read_text())
    assert adapter_path == str(output / "lora_adapter")
    assert "base_model.model.score.weight" in adapter
    assert "base_model.model.score.bias" in adapter
    assert torch.equal(adapter["base_model.model.score.weight"], torch.full((1, 4), 3.0))
    assert adapter_config["task_type"] == "TOKEN_CLS"
    assert adapter_config["lora_alpha"] == 2
    assert "score" in adapter_config["modules_to_save"]
    assert set(state) == {"model.layers.0.self_attn.q_proj.weight", "score.weight", "score.bias"}


def test_value_model_override_selects_token_classifier():
    merger = object.__new__(_TestMerger)
    merger.config = SimpleNamespace(is_value_model=True)
    merger.model_config = SimpleNamespace(architectures=["Qwen2ForCausalLM"])

    assert merger.get_transformers_auto_model_class() is AutoModelForTokenClassification
