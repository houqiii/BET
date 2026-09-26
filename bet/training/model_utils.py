from __future__ import annotations

from typing import Any, Dict, Iterable, List


def default_lora_targets(model_family: str = 'qwen') -> List[str]:
    # Works for Qwen/Llama-like decoder-only models. Override in config for custom backbones.
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def maybe_set_pad_token(tokenizer: Any) -> Any:
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(model_name: str, **kwargs):
    from transformers import AutoModelForCausalLM
    from transformers.utils.peft_utils import find_adapter_config_file
    if find_adapter_config_file(model_name) is not None:
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **kwargs)
        model = PeftModel.from_pretrained(base, model_name).merge_and_unload()
        model.requires_grad_(True)
        return model
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
