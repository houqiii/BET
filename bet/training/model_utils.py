from __future__ import annotations

from typing import Any, Dict, Iterable, List


def default_lora_targets(model_family: str = 'qwen') -> List[str]:
    # Works for Qwen/Llama-like decoder-only models. Override in config for custom backbones.
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def maybe_set_pad_token(tokenizer: Any) -> Any:
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
