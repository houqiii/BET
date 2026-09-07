# Reward design

For each prompt, `K` sampled completions form a group and are summarized into:

- `s_hat(x)`: fraction of correct rollouts in the group.
- `c_star(x)`: mean cost of the shortest `p`-fraction of correct rollouts.
- `b_star(x) = c_star(x) / L_max`: normalized budget target.

The composite reward `R(y|x) = R_VAL + R_EFF + R_CAL` is implemented as four components:

1. `format`: exactly one `<predict>` block, one `<think>` block, and one boxed answer, in that order.
2. `value` (`R_VAL`): `+1` for a correct answer; `+delta` for abstention when the group has no correct rollout and `-lambda` otherwise; `-alpha_fail * c / L_max` for a failed attempt.
3. `efficiency` (`R_EFF`): `beta * max(0, 1 - c(y) / c_star(x))` for correct solutions when `s_hat(x) > tau`, and `0` otherwise.
4. `calibration` (`R_CAL`): aligns the declared solvability and budget with the group profile, with an asymmetric budget loss that scales underestimation by `mu`.

Only the final answer inside `\boxed{}` is used for correctness.

## Guaranteed attempt within a group

The abstention gate in `R_VAL` only separates rollouts when a group contains an attempt. If every
rollout of a query declares abstention, `s_hat(x)` is zero by construction, all rollouts receive
`+delta`, and the group-relative advantage vanishes.

`bet.group_constraint` keeps one attempt in every group. The abstain-or-attempt decision is read from
the `<predict>` block, so a group in which all `K` rollouts declare abstention has one entry, chosen
uniformly at random, rewritten to a positive budget before any reasoning is generated. The group size
stays at `K`. The rewritten declaration is excluded from `R_CAL`, while `R_VAL` and `R_EFF` apply to
its reasoning and answer as usual.

`bet.training.rollout.generate_group` implements the two-phase rollout this requires: generate up to
`</predict>`, apply the constraint, then complete each trace. Both phases take a caller-supplied
generation function, so the same code path works with a local model or a vLLM server.
