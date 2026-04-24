#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json

from bet.group_stats import compute_group_profiles


def parse_args():
    p = argparse.ArgumentParser(description='Summarize grouped rollouts into solvability and efficient-cost estimates.')
    p.add_argument('--rollouts', required=True, help='JSONL with prompt, answer, and rollouts list')
    p.add_argument('--max_completion_tokens', type=int, default=16384)
    return p.parse_args()


def main():
    args = parse_args()
    prompts, completions, answers = [], [], []
    with open(args.rollouts, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            for rollout in row['rollouts']:
                prompts.append(row['prompt'])
                answers.append(row['answer'])
                completions.append(rollout['completion'])
    profiles = compute_group_profiles(prompts, completions, answers, max_completion_tokens=args.max_completion_tokens)
    for p in profiles.values():
        print(json.dumps(p.__dict__, ensure_ascii=False))


if __name__ == '__main__':
    main()
