from bet.rewards import BETRewardConfig, compute_bet_rewards


def test_rewards_correct_short_solution_positive():
    p = ["Problem: 1+1?"] * 2
    a = [r"\boxed{2}"] * 2
    c = [
        "<predict>\nSolvability: 0.9\nBudget: 0.1\n</predict>\n<think>\nAdd.\n</think>\n\\boxed{2}",
        "<predict>\nSolvability: 0.9\nBudget: 0.1\n</predict>\n<think>\nAdd one and one.\n</think>\n\\boxed{2}",
    ]
    rewards = compute_bet_rewards(p, c, a, BETRewardConfig(max_completion_tokens=128))
    assert all(r.value == 1.0 for r in rewards)
    assert all(r.format > 0 for r in rewards)


def test_fold_reward_when_group_unsolved():
    p = ["Problem: impossible"] * 2
    a = [r"\boxed{999}"] * 2
    c = [
        "<predict>\nSolvability: 0.1\nBudget: 0.05\n</predict>\n<think>\nCannot solve.\n</think>\n\\boxed{Unsolvable}",
        "<predict>\nSolvability: 0.1\nBudget: 0.05\n</predict>\n<think>\nCannot solve.\n</think>\n\\boxed{Unsolvable}",
    ]
    rewards = compute_bet_rewards(p, c, a, BETRewardConfig(max_completion_tokens=128))
    assert all(r.value > 0 for r in rewards)
