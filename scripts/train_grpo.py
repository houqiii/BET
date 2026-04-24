#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from bet.data.preprocess import normalize_grpo_record
from bet.prompts import apply_chat_template
from bet.rewards import make_trl_reward_functions
from bet.training.callbacks import BETConsoleCallback
from bet.training.config import load_config
from bet.training.grpo import reward_config_from_dict
from bet.training.model_utils import maybe_set_pad_token


def parse_args():
    p = argparse.ArgumentParser(description='Stage-2 GRPO for BET with dynamic group-profile rewards.')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--model_name_or_path', type=str, default=None)
    p.add_argument('--train_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--vllm_server_url', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model_name_or_path or cfg['model']['name_or_path']
    train_file = args.train_file or cfg['data']['train_file']
    output_dir = args.output_dir or cfg['training']['output_dir']

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = maybe_set_pad_token(tokenizer)

    raw = load_dataset('json', data_files=train_file, split='train')
    rows = []
    for r in raw:
        norm = normalize_grpo_record(r)
        # TRL expects a prompt column and arbitrary metadata columns used by reward functions.
        prompt = apply_chat_template(tokenizer, norm['prompt'].split('Problem:', 1)[-1])
        rows.append({'prompt': prompt, 'answer': norm['answer'], 'id': norm.get('id', '')})
    train_dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get('training', {}).get('bf16', True) else torch.float16,
        trust_remote_code=True,
        attn_implementation=cfg.get('model', {}).get('attn_implementation', 'sdpa'),
    )

    tr = cfg.get('training', {})
    grpo_kwargs = dict(
        output_dir=output_dir,
        learning_rate=tr.get('learning_rate', 1e-6),
        num_train_epochs=tr.get('num_train_epochs', 1),
        per_device_train_batch_size=tr.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=tr.get('gradient_accumulation_steps', 16),
        num_generations=tr.get('num_generations', 8),
        generation_batch_size=tr.get('generation_batch_size', None),
        max_completion_length=tr.get('max_completion_length', 16384),
        temperature=tr.get('temperature', 0.8),
        top_p=tr.get('top_p', 1.0),
        logging_steps=tr.get('logging_steps', 5),
        save_steps=tr.get('save_steps', 100),
        report_to=tr.get('report_to', 'none'),
        bf16=tr.get('bf16', True),
        seed=tr.get('seed', 42),
    )
    if args.vllm_server_url:
        grpo_kwargs['use_vllm'] = True
        grpo_kwargs['vllm_server_host'] = args.vllm_server_url
    grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if v is not None}
    training_args = GRPOConfig(**grpo_kwargs)

    reward_cfg = reward_config_from_dict(cfg)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_trl_reward_functions(reward_cfg),
        train_dataset=train_dataset,
        args=training_args,
        callbacks=[BETConsoleCallback()],
    )
    trainer.train()
    final_dir = Path(output_dir) / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == '__main__':
    main()
