# Examples

The example files are small enough to run quickly but are structured like real training artifacts.

- `sft/bet_cold_start_examples.jsonl` contains cold-start demonstrations for short solve, hero call, and nice fold behavior.
- `grpo/mini_math_train.jsonl` contains math problems with canonical boxed answers for GRPO debugging.
- `profiling/sample_group_rollouts.jsonl` contains grouped rollouts that exercise the group profile, fold gate, and efficient-cost estimator.
- `profiling/profile_summaries.jsonl` shows the profile-summary format accepted by `scripts/build_sft_data.py`.
- `predictions/sample_generations.jsonl` can be scored by `scripts/evaluate_generations.py`.

These files are not benchmark subsets and are not intended to reproduce paper-scale results. They are meant for checking parser behavior, reward signs, data loading, and evaluator output before launching full experiments.
