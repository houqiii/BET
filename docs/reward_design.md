# Reward design

BET uses dynamic group profiles (Section 3.2). For each prompt, K sampled completions are grouped to estimate:

- `s_hat(x)`: empirical group success rate (Eq. 3).
- `c_star(x)`: lower-envelope cost among correct rollouts.
- `b_star(x) = c_star(x) / L_max`: normalized budget target (Table 7).

The composite reward R(y|x) = R_VAL + R_EFF + R_CAL (Section 3.3) is decomposed into four engineering components:

1. `format`: encourages exactly one `<predict>` block, exactly one `<think>` block, and exactly one boxed answer.
2. `value` (R_VAL, Eq. 4): rewards correct answers (+1), rewards abstention (+δ) only when the group has zero correct rollouts, and penalizes failed attempts by cost-proportional penalty φ(c) = α_fail · c / L_max.
3. `efficiency` (R_EFF, Eq. 5): rewards concise correct solutions when solvability exceeds confidence threshold τ.
4. `calibration` (R_CAL, Eq. 6): aligns predicted solvability and budget with the group profile. Uses asymmetric budget loss with underestimation penalty multiplier μ=2.0.

Only the final answer inside `\boxed{}` is used for correctness.
