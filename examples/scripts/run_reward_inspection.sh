#!/usr/bin/env bash
set -euo pipefail
python scripts/inspect_rewards.py \
  --examples examples/profiling/sample_group_rollouts.jsonl \
  --max_completion_tokens 8192
