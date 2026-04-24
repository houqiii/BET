# Training pipeline

## 1. Offline profiling

Generate K rollouts per training query from the base or intermediate policy. Use these rollouts to estimate solvability and efficient solution cost.

## 2. Cold-start SFT

Construct demonstrations that expose the model to three canonical behaviors:

- short solve for easy queries with a short correct trace;
- hero call for hard but solvable queries that require nontrivial reasoning;
- nice fold for zero-return or underspecified queries.

The SFT stage teaches the response protocol and action vocabulary, not the final decision boundary.

## 3. GRPO

During GRPO, each batch samples grouped completions. Group profiles are recomputed from the current policy outputs, so the reward changes as the policy changes. This is why `bet/group_stats.py` is called from the reward path rather than from a static preprocessor.

## 4. Evaluation

Use greedy or fixed-temperature decoding, extract the boxed answer, and report accuracy, average think-token count, fold rate, and format rate. If a vanilla baseline is supplied, also report relative accuracy-efficiency.
