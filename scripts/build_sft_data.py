#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse

from bet.data.loaders import load_jsonl, write_jsonl
from bet.data.profiling import ProfileRecord, profile_to_sft_target, build_cold_start_profiles


def parse_args():
    p = argparse.ArgumentParser(description='Build cold-start SFT examples from profile summaries.')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--profiles')
    source.add_argument('--samples', help='One base-policy completion per query')
    p.add_argument('--tokenizer', default=None)
    p.add_argument('--max_completion_tokens', type=int, default=16384)
    p.add_argument('--split_fraction', type=float, default=0.6)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = []
    if args.samples:
        if not args.tokenizer:
            raise ValueError('--samples requires --tokenizer for think-token costs.')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        records = build_cold_start_profiles(
            load_jsonl(args.samples), tokenizer=tokenizer,
            split_fraction=args.split_fraction,
        )
        write_jsonl(args.output, [profile_to_sft_target(r, args.max_completion_tokens) for r in records])
        return
    for r in load_jsonl(args.profiles):
        rec = ProfileRecord(
            problem=r['problem'],
            answer=r['answer'],
            regime=r['regime'],
            solvability=float(r['solvability']),
            efficient_cost=float(r['efficient_cost']),
            selected_trace=r.get('selected_trace', ''),
            selected_answer=r.get('selected_answer', r['answer']),
            difficulty=None if r.get('difficulty') is None else float(r['difficulty']),
        )
        out.append(profile_to_sft_target(rec, args.max_completion_tokens))
    write_jsonl(args.output, out)


if __name__ == '__main__':
    main()
