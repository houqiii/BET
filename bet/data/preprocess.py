from __future__ import annotations

from typing import Any, Dict, Optional

from ..prompts import build_user_prompt, sharegpt_to_prompt_completion


def normalize_grpo_record(record: Dict[str, Any]) -> Dict[str, str]:
    problem = record.get('problem') or record.get('question') or record.get('prompt')
    answer = record.get('answer') or record.get('gold') or record.get('target')
    if problem is None or answer is None:
        raise ValueError(f"GRPO record must contain problem/question/prompt and answer/gold/target: {record}")
    return {'prompt': build_user_prompt(str(problem)), 'answer': str(answer), 'id': str(record.get('id', ''))}


def normalize_sft_record(record: Dict[str, Any], tokenizer: Optional[Any] = None) -> Dict[str, str]:
    if 'conversations' in record or ('prompt' in record and 'completion' in record):
        return sharegpt_to_prompt_completion(record, tokenizer=tokenizer)
    problem = record.get('problem') or record.get('question')
    completion = record.get('completion') or record.get('response')
    if problem is None or completion is None:
        raise ValueError(f"SFT record must contain conversations, prompt/completion, or problem/completion: {record}")
    return {'prompt': build_user_prompt(str(problem)), 'completion': str(completion)}
