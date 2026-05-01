from bet.evaluation.metrics import compute_metrics, relative_accuracy_efficiency


def test_metrics():
    records = [
        {"answer": r"\boxed{3}", "completion": "<predict>\nSolvability: 0.1\nBudget: 0.1\n</predict>\n<think>\n1+2.\n</think>\n\\boxed{3}"},
        {"answer": r"\boxed{4}", "completion": "<predict>\nSolvability: 0.1\nBudget: 0.1\n</predict>\n<think>\nwrong.\n</think>\n\\boxed{5}"},
    ]
    m = compute_metrics(records)
    assert m['accuracy'] == 0.5
    assert m['format_rate'] == 1.0
    assert relative_accuracy_efficiency(m, m) == 1.0
