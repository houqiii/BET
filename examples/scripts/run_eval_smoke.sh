#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_generations.py \
  --predictions examples/predictions/sample_generations.jsonl \
  --output outputs/eval_smoke.json
