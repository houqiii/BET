#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json

from bet.data.loaders import load_jsonl
from bet.rewards import BETRewardConfig, compute_bet_rewards


def parse_args():
    p = argparse.ArgumentParser(description='Inspect BET rewards on grouped example rollouts.')
    p.add_argument('--examples', required=True)
    p.add_argument('--max_completion_tokens', type=int, default=16384)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.examples)
    prompts, completions, answers, ids = [], [], [], []
    for row in rows:
        prompt = row['prompt']
        answer = row['answer']
        for rollout in row['rollouts']:
            prompts.append(prompt)
            answers.append(answer)
            completions.append(rollout['completion'])
            ids.append(f"{row.get('id','')}::{rollout.get('id','')}")
    cfg = BETRewardConfig(max_completion_tokens=args.max_completion_tokens)
    rewards = compute_bet_rewards(prompts, completions, answers, cfg)
    for rid, reward in zip(ids, rewards):
        print(json.dumps({'id': rid, **reward.to_dict()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
