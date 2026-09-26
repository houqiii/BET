from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from ..math_eval import is_correct
from ..parsing import parse_response, think_token_proxy


def safe_mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_metrics(records: Sequence[Dict[str, Any]], tokenizer: Any = None) -> Dict[str, float]:
    correct = []
    lengths = []
    folds = []
    formats = []
    for r in records:
        completion = r.get('completion') or r.get('prediction') or r.get('response', '')
        answer = r.get('answer') or r.get('gold') or r.get('target', '')
        parsed = parse_response(completion)
        correct.append(float(is_correct(completion, answer)))
        lengths.append(float(think_token_proxy(completion, tokenizer)))
        folds.append(float(parsed.is_fold))
        formats.append(float(parsed.format_ok))
    return {
        'accuracy': safe_mean(correct),
        ('avg_think_tokens' if tokenizer is not None else 'avg_think_tokens_proxy'): safe_mean(lengths),
        'fold_rate': safe_mean(folds),
        'format_rate': safe_mean(formats),
        'n': float(len(records)),
    }


def relative_accuracy_efficiency(method: Dict[str, float], baseline: Dict[str, float]) -> float:
    acc0 = max(1e-12, baseline.get('accuracy', 0.0))
    key = 'avg_think_tokens' if 'avg_think_tokens' in method else 'avg_think_tokens_proxy'
    if key not in baseline:
        raise ValueError('Method and baseline must use the same token-counting mode.')
    tok = max(1e-12, method[key])
    tok0 = max(1e-12, baseline[key])
    return (method.get('accuracy', 0.0) / acc0) * (tok0 / tok)
