import random

from bet.group_constraint import (
    apply_guaranteed_attempt,
    declared_abstentions,
    force_attempt_declaration,
    guaranteed_attempt_index,
)
from bet.parsing import parse_predict
from bet.rewards import BETRewardConfig, compute_bet_rewards

ABSTAIN = "<predict>\nSolvability: 0.00\nBudget: 0.00\n</predict>\n"
ATTEMPT = "<predict>\nSolvability: 0.60\nBudget: 0.30\n</predict>\n"


def test_declared_abstentions_reads_predict_block():
    assert declared_abstentions([ABSTAIN, ATTEMPT]) == [True, False]


def test_no_override_when_group_has_an_attempt():
    assert guaranteed_attempt_index([True, False, True]) is None


def test_override_when_every_rollout_abstains():
    assert guaranteed_attempt_index([True] * 4, random.Random(0)) in range(4)


def test_forced_declaration_commits_a_positive_budget():
    forced = force_attempt_declaration(ABSTAIN, solvability=1 / 16, budget=1.0)
    solvability, budget, ok = parse_predict(forced)
    assert ok and budget == 1.0 and solvability > 0


def test_apply_guaranteed_attempt_keeps_group_size():
    texts, overridden = apply_guaranteed_attempt([ABSTAIN] * 8, k=8, rng=random.Random(1))
    assert len(texts) == 8
    assert sum(overridden) == 1
    assert declared_abstentions(texts).count(False) == 1


def test_calibration_is_masked_for_the_overridden_rollout():
    prompts = ["Problem: impossible"] * 2
    answers = [r"\boxed{999}"] * 2
    completions = [
        ABSTAIN + "<think>\nCannot solve.\n</think>\n\\boxed{Unsolvable}",
        ATTEMPT + "<think>\nTry a construction.\n</think>\n\\boxed{123}",
    ]
    cfg = BETRewardConfig(max_completion_tokens=128)
    plain = compute_bet_rewards(prompts, completions, answers, cfg)
    masked = compute_bet_rewards(prompts, completions, answers, cfg, overridden=[False, True])
    assert plain[1].calibration != 0.0
    assert masked[1].calibration == 0.0
    assert masked[1].value == plain[1].value
    assert masked[1].efficiency == plain[1].efficiency
