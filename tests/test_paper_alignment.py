import pytest

from bet.data.profiling import build_cold_start_profiles, profile_to_sft_target
from bet.data.preprocess import normalize_sft_record
from bet.evaluation.metrics import compute_metrics
from bet.group_stats import compute_group_profiles
from bet.math_eval import math_equal
from bet.parsing import declared_abstention, parse_predict, think_token_proxy
from bet.prompts import apply_chat_template
from bet.rewards import BETRewardConfig, compute_bet_rewards, make_trl_reward_functions


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '\nAssistant:\n<think>\n'


def response(think, answer='1', budget=0.1):
    return f'<predict>\nSolvability: 0.5\nBudget: {budget}\n</predict>\n<think>{think}</think>\n\\boxed{{{answer}}}'


def test_fold_is_declared_by_zero_budget():
    assert declared_abstention(response('', budget=0))
    assert not declared_abstention(response('', budget=0.000061))
    assert not parse_predict(response('', budget=1.1))[2]


def test_empty_and_truncated_think_costs():
    tokenizer = Tokenizer()
    assert think_token_proxy(response(''), tokenizer) == 0
    assert think_token_proxy('<think>abc', tokenizer) == 3
    assert think_token_proxy('abc</think>\\boxed{1}', tokenizer) == 3


def test_group_cost_and_all_rewards_use_same_tokenizer():
    completions = [response('x' * n) for n in range(1, 11)]
    profiles = compute_group_profiles(['p'] * 10, completions, ['1'] * 10,
                                      max_completion_tokens=100, tokenizer=Tokenizer())
    assert profiles['p'].efficient_cost == 2
    rewards = compute_bet_rewards(['p'] * 10, completions, ['1'] * 10,
                                 BETRewardConfig(max_completion_tokens=100), tokenizer=Tokenizer())
    assert rewards[0].efficiency == pytest.approx(0.15)
    assert rewards[-1].efficiency == 0
    failure = compute_bet_rewards(['p'], [response('x' * 20, '2')], ['1'],
                                 BETRewardConfig(max_completion_tokens=100), tokenizer=Tokenizer())[0]
    assert failure.value == pytest.approx(-0.04)


def test_one_success_flips_fold_gate_and_keeps_low_solvability_depth():
    completions = [response('work')] + [response('', 'Unsolvable', 0)] * 15
    rewards = compute_bet_rewards(['p'] * 16, completions, ['1'] * 16)
    assert rewards[0].value == 1
    assert rewards[0].efficiency == 0
    assert all(r.value == -0.8 for r in rewards[1:])


def test_single_sample_cold_start_retains_all_three_regimes():
    samples = [dict(problem=str(i), answer='1', completion=response('x' * n, answer), difficulty=0.5)
               for i, (n, answer) in enumerate([(1, '1'), (10, '1'), (20, '1'), (30, '2')])]
    profiles = build_cold_start_profiles(samples, tokenizer=Tokenizer(), split_fraction=0.5)
    assert [p.regime for p in profiles] == ['short_solve', 'short_solve', 'hero_call', 'nice_fold']
    targets = [profile_to_sft_target(p) for p in profiles]
    assert not declared_abstention(targets[0]['completion'])
    assert declared_abstention(targets[-1]['completion'])
    assert len(targets) == len(samples)


def test_chat_template_starts_with_predict_in_both_stages():
    tokenizer = Tokenizer()
    prompt = apply_chat_template(tokenizer, '1+1?')
    assert prompt.endswith('Assistant:\n')
    assert normalize_sft_record({'problem': '1+1?', 'completion': response('add')}, tokenizer)['prompt'] == prompt


def test_eval_counts_empty_folds_as_zero_tokens_and_wrong():
    metrics = compute_metrics([{'completion': response('', 'Unsolvable', 0), 'answer': '1'}], Tokenizer())
    assert metrics['accuracy'] == 0
    assert metrics['avg_think_tokens'] == 0
    assert metrics['n'] == 1


def test_reward_functions_have_trl_names():
    assert [f.__name__ for f in make_trl_reward_functions()] == [
        'bet_format', 'bet_value', 'bet_efficiency', 'bet_calibration',
    ]


def test_fraction_correctness_and_invalid_denominators():
    assert math_equal(r'\frac{1}{2}', '0.5')
    assert math_equal(r'\frac{1}{2}', '1/2')
    assert not math_equal(r'\frac{1}{0}', '1')
