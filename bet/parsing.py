from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .constants import PREDICT_END, PREDICT_START, THINK_END, THINK_START, UNSOLVABLE_TOKEN
from .schemas import ParsedResponse

PREDICT_BLOCK_RE = re.compile(
    r"^\s*<predict>\s*"
    r"Solvability:\s*([0-9]*\.?[0-9]+)\s*"
    r"Budget:\s*([0-9]*\.?[0-9]+)\s*"
    r"</predict>",
    re.DOTALL,
)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def get_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("content", x))
    if isinstance(x, list):
        if not x:
            return ""
        if isinstance(x[-1], dict):
            return str(x[-1].get("content", x[-1]))
        return str(x[-1])
    return str(x)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_boxed(text: Any) -> Optional[str]:
    text = get_text(text)
    needle = r"\boxed{"
    idx = text.rfind(needle)
    if idx < 0:
        return None
    start = idx + len(needle)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def parse_predict(text: Any) -> Tuple[Optional[float], Optional[float], bool]:
    text = get_text(text)
    m = PREDICT_BLOCK_RE.search(text)
    if not m:
        return None, None, False
    try:
        d = clamp01(float(m.group(1)))
        b = clamp01(float(m.group(2)))
    except ValueError:
        return None, None, False
    return d, b, True


def inspect_format(text: Any) -> Dict[str, Any]:
    text = get_text(text)
    stripped = text.lstrip()
    d, b, predict_parse_ok = parse_predict(text)
    think_matches = list(THINK_BLOCK_RE.finditer(text))
    boxed = extract_boxed(text)

    predict_count_ok = text.count(PREDICT_START) == 1 and text.count(PREDICT_END) == 1
    think_count_ok = text.count(THINK_START) == 1 and text.count(THINK_END) == 1
    boxed_count_ok = text.count(r"\boxed{") == 1
    boxed_ok = boxed is not None and len(boxed.strip()) > 0

    order_ok = False
    if predict_count_ok and think_count_ok and boxed_ok:
        pred_s = text.find(PREDICT_START)
        pred_e = text.find(PREDICT_END)
        think_s = text.find(THINK_START)
        think_e = text.find(THINK_END)
        boxed_s = text.rfind(r"\boxed{")
        order_ok = pred_s < pred_e < think_s < think_e < boxed_s

    full_format_ok = (
        stripped.startswith(PREDICT_START)
        and predict_count_ok
        and predict_parse_ok
        and think_count_ok
        and len(think_matches) == 1
        and boxed_ok
        and boxed_count_ok
        and order_ok
    )

    return {
        "starts_with_predict": stripped.startswith(PREDICT_START),
        "predict_count_ok": predict_count_ok,
        "predict_parse_ok": predict_parse_ok,
        "think_count_ok": think_count_ok,
        "think_parse_ok": len(think_matches) == 1,
        "boxed_ok": boxed_ok,
        "boxed_count_ok": boxed_count_ok,
        "order_ok": order_ok,
        "format_ok": full_format_ok,
        "solvability_pred": d,
        "budget": b,
        "boxed": boxed,
        "think": think_matches[0].group(1).strip() if len(think_matches) == 1 else "",
    }


def parse_response(text: Any) -> ParsedResponse:
    raw = get_text(text)
    info = inspect_format(raw)
    boxed = info["boxed"]
    return ParsedResponse(
        text=raw,
        format_ok=bool(info["format_ok"]),
        solvability_pred=info["solvability_pred"],
        budget=info["budget"],
        think=info["think"],
        boxed=boxed,
        is_fold=(boxed == UNSOLVABLE_TOKEN or boxed == f"<{UNSOLVABLE_TOKEN}>"),
        diagnostics=info,
    )


def think_token_proxy(text: Any) -> float:
    parsed = parse_response(text)
    if parsed.think:
        return len(parsed.think) / 4.0
    return len(get_text(text)) / 4.0
