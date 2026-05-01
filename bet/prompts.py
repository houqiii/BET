from __future__ import annotations

from typing import Any, Dict, List, Optional

BET_SYSTEM_PROMPT = """Solve the problem efficiently.
Before reasoning, output a self-assessment block:
<predict>
Solvability: <number in [0,1]>
Budget: <number in [0,1]>
</predict>
Then reason inside <think>...</think>.
End with exactly one final answer in \\boxed{}.
If the problem is beyond your current reliable capability, keep the reasoning short and output \\boxed{Unsolvable}.
"""


def build_user_prompt(problem: str, system_prompt: str = BET_SYSTEM_PROMPT) -> str:
    return f"{system_prompt.strip()}\n\nProblem:\n{problem.strip()}"


def apply_chat_template(tokenizer: Any, problem: str, *, add_generation_prompt: bool = True) -> str:
    messages = [{"role": "user", "content": build_user_prompt(problem)}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return build_user_prompt(problem) + "\n\nAssistant:\n"


def sharegpt_to_prompt_completion(example: Dict[str, Any], tokenizer: Optional[Any] = None) -> Dict[str, str]:
    if "prompt" in example and "completion" in example:
        return {"prompt": example["prompt"], "completion": example["completion"]}
    conversations: List[Dict[str, str]] = example["conversations"]
    user = conversations[0].get("value") or conversations[0].get("content")
    assistant = conversations[1].get("value") or conversations[1].get("content")
    if "Problem:" in user:
        problem = user.split("Problem:", 1)[1]
    else:
        problem = user
    if tokenizer is not None:
        prompt = apply_chat_template(tokenizer, problem)
    else:
        prompt = build_user_prompt(problem)
    return {"prompt": prompt, "completion": assistant}
