#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Merge a LoRA adapter into its base model for serving or GRPO initialization.')
    p.add_argument('--base_model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
