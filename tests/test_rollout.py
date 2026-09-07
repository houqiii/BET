import random

from bet.training.rollout import generate_group, generate_groups

ABSTAIN = "<predict>\nSolvability: 0.00\nBudget: 0.00\n</predict>\n"


def _declare_all_abstain(prompts):
    return [ABSTAIN for _ in prompts]


def _complete(prefixes):
    return ["<think>\nWork.\n</think>\n\\boxed{1}" for _ in prefixes]


def test_generate_group_forces_one_attempt():
    completions, overridden = generate_group(
        "Problem: x", declare=_declare_all_abstain, complete=_complete, k=4, rng=random.Random(0),
    )
    assert len(completions) == 4
    assert sum(overridden) == 1
    assert all(c.startswith("<predict>") for c in completions)


def test_generate_groups_flattens_prompts():
    prompts, completions, overridden = generate_groups(
        ["Problem: a", "Problem: b"],
        declare=_declare_all_abstain, complete=_complete, k=3, rng=random.Random(0),
    )
    assert len(prompts) == len(completions) == len(overridden) == 6
    assert sum(overridden) == 2
