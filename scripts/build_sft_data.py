#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse

from bet.data.loaders import load_jsonl, write_jsonl
from bet.data.profiling import ProfileRecord, profile_to_sft_target


def parse_args():
    p = argparse.ArgumentParser(description='Build cold-start SFT examples from profile summaries.')
    p.add_argument('--profiles', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = []
    for r in load_jsonl(args.profiles):
        rec = ProfileRecord(
            problem=r['problem'],
            answer=r['answer'],
            regime=r['regime'],
            solvability=float(r['solvability']),
            efficient_cost=float(r['efficient_cost']),
            selected_trace=r.get('selected_trace', ''),
            selected_answer=r.get('selected_answer', r.get('answer', '').replace('\\boxed{', '').rstrip('}')),
        )
        out.append(profile_to_sft_target(rec))
    write_jsonl(args.output, out)


if __name__ == '__main__':
    main()
