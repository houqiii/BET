#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bet.data.preprocess import normalize_sft_record
from bet.training.config import load_config
from bet.training.model_utils import maybe_set_pad_token
from bet.training.sft import build_lora_config


def parse_args():
    p = argparse.ArgumentParser(description='Stage-1 LoRA SFT for BET cold-start behavior.')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--model_name_or_path', type=str, default=None)
    p.add_argument('--train_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--max_length', type=int, default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--num_train_epochs', type=float, default=None)
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
    records = [normalize_sft_record(r, tokenizer=tokenizer) for r in raw]
    train_dataset = Dataset.from_list(records)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get('training', {}).get('bf16', True) else torch.float16,
        trust_remote_code=True,
        attn_implementation=cfg.get('model', {}).get('attn_implementation', 'sdpa'),
    )

    tr = cfg.get('training', {})
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or tr.get('num_train_epochs', 2),
        per_device_train_batch_size=tr.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=tr.get('gradient_accumulation_steps', 16),
        learning_rate=args.learning_rate or tr.get('learning_rate', 2e-5),
        warmup_ratio=tr.get('warmup_ratio', 0.03),
        lr_scheduler_type=tr.get('lr_scheduler_type', 'cosine'),
        weight_decay=tr.get('weight_decay', 0.01),
        logging_steps=tr.get('logging_steps', 5),
        save_strategy=tr.get('save_strategy', 'steps'),
        save_steps=tr.get('save_steps', 100),
        bf16=tr.get('bf16', True),
        gradient_checkpointing=tr.get('gradient_checkpointing', True),
        max_length=args.max_length or tr.get('max_length', 16384),
        completion_only_loss=tr.get('completion_only_loss', True),
        report_to=tr.get('report_to', 'none'),
        seed=tr.get('seed', 42),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=build_lora_config(cfg),
        args=training_args,
    )
    trainer.train()
    final_dir = Path(output_dir) / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == '__main__':
    main()
