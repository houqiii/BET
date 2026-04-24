# Reward design

BET uses dynamic group profiles. For each prompt, K sampled completions are grouped to estimate:

- `s_hat`: empirical group success rate.
- `c_star`: robust lower-envelope cost among correct rollouts.
- `d_star`: calibration target derived from solvability and efficient cost.
- `b_star`: discretized budget target derived from efficient cost.

The reward is decomposed into four engineering components:

1. `format`: encourages exactly one `<predict>` block, exactly one `<think>` block, and exactly one boxed answer.
2. `value`: rewards correct answers, rewards abstention only when the group has zero correct rollouts, and penalizes failed attempts by cost.
3. `efficiency`: rewards concise correct solutions when the group is reliably solvable.
4. `calibration`: aligns declared difficulty and budget with the group profile.

Only the final answer inside `\boxed{}` is used for correctness.
