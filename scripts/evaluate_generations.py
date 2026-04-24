#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

from bet.data.loaders import load_jsonl
from bet.evaluation.metrics import compute_metrics, relative_accuracy_efficiency


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate JSONL generations for BET-style output.')
    p.add_argument('--predictions', required=True)
    p.add_argument('--baseline', default=None)
    p.add_argument('--output', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    metrics = compute_metrics(load_jsonl(args.predictions))
    if args.baseline:
        base = compute_metrics(load_jsonl(args.baseline))
        metrics['relative_accuracy_efficiency'] = relative_accuracy_efficiency(metrics, base)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
