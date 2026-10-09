import json
from types import SimpleNamespace

from verl.trainer.ppo.ray_trainer import _save_lora_train_meta


def test_save_lora_train_meta(tmp_path):
    config = SimpleNamespace(
        get=lambda key, default=None: {"lora_rank": 32, "lora_alpha": 32}.get(key, default)
    )

    _save_lora_train_meta(str(tmp_path), config, "TOKEN_CLS")

    assert json.loads((tmp_path / "lora_train_meta.json").read_text()) == {
        "r": 32,
        "lora_alpha": 32,
        "task_type": "TOKEN_CLS",
    }
