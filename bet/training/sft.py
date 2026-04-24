from __future__ import annotations

from typing import Any, Dict

from .model_utils import default_lora_targets


def build_lora_config(config: Dict[str, Any]):
    from peft import LoraConfig
    lora = config.get('lora', {})
    return LoraConfig(
        r=int(lora.get('r', 64)),
        lora_alpha=int(lora.get('alpha', 128)),
        lora_dropout=float(lora.get('dropout', 0.05)),
        target_modules=lora.get('target_modules') or default_lora_targets(),
        bias='none',
        task_type='CAUSAL_LM',
    )
