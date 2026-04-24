from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Optional

from .constants import UNSOLVABLE_TOKEN
from .parsing import extract_boxed, get_text


def normalize_math(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\mathrm", "").replace("\\text", "")
    s = re.sub(r"[\$\s\n\t]+", "", s)
    s = s.strip("{}")
    return s


def _try_fraction(s: str) -> Optional[Fraction]:
    s = normalize_math(s)
    m = re.fullmatch(r"(-?\d+)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        whole, num, den = map(int, m.groups())
        return Fraction(whole, 1) + Fraction(num, den)
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return Fraction(int(s), 1)
    return None


def math_equal(pred: Any, gold: Any) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    p = str(pred).strip()
    g = str(gold).strip()
    if p == g:
        return True
    if normalize_math(p) == normalize_math(g):
        return True
    fp = _try_fraction(p)
    fg = _try_fraction(g)
    return fp is not None and fg is not None and fp == fg


def canonical_gold(answer: Any) -> str:
    text = get_text(answer)
    boxed = extract_boxed(text)
    return boxed if boxed is not None else text.strip()


def is_correct(completion: Any, answer: Any) -> bool:
    pred = extract_boxed(completion)
    if pred is None:
        return False
    if pred in {UNSOLVABLE_TOKEN, f"<{UNSOLVABLE_TOKEN}>"}:
        return False
    return math_equal(pred, canonical_gold(answer))
