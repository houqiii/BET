# Training pipeline

## 1. Offline profiling

For Stage 1, generate one base-policy sample per training query. Failed samples become nice-fold targets; split correct samples by think-token cost into short-solve and hero-call targets. The split fraction is configurable. K-rollout solvability and efficient cost are estimated online in Stage 2.

```bash
python scripts/build_sft_data.py --samples data/base_samples.jsonl \
  --tokenizer /path/to/base_model --split_fraction 0.6 --output data/cold_start.jsonl
```

Each sample contains `problem`, `answer`, and `completion`, with an optional difficulty normalized to `[0, 1]`. The script retains the sampled correct reasoning trace. `--profiles` remains available for preconstructed demonstrations.

## 2. Cold-start SFT

Construct demonstrations that expose the model to three canonical behaviors:

- short solve for easy queries with a short correct trace;
- hero call for hard but solvable queries that require nontrivial reasoning;
- nice fold for zero-return or underspecified queries.

The SFT stage teaches the response protocol and action vocabulary, not the final decision boundary.

## 3. GRPO

During GRPO, each batch samples grouped completions. Group profiles are recomputed from the current policy outputs, so the reward changes as the policy changes. This is why `bet/group_stats.py` is called from the reward path rather than from a static preprocessor.

## 4. Evaluation

Use temperature 0.8 and top-p 1.0, repeat evaluation five times, extract the boxed answer, and report accuracy, average think-token count, fold rate, and format rate. If a vanilla baseline is supplied, also report relative accuracy-efficiency.

Pass `--tokenizer /path/to/model` to `evaluate_generations.py` for token counts matching the paper. Without a tokenizer, the evaluator reports only a character-based proxy.
