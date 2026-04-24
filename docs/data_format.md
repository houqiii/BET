# Data format

## GRPO records

Required fields:

- `problem`: natural-language query.
- `answer`: canonical final answer, preferably already wrapped as `\boxed{...}`.

Optional fields:

- `id`: stable record identifier.
- `tags`: list of task labels used only for filtering or analysis.

## SFT records

The SFT loader accepts either ShareGPT-style records or direct prompt/completion pairs.

### ShareGPT style

```json
{
  "conversations": [
    {"from": "human", "value": "Problem:\n..."},
    {"from": "gpt", "value": "<predict>..."}
  ]
}
```

### Prompt/completion style

```json
{"prompt":"...","completion":"<predict>..."}
```

## Rollout-profile records

A rollout-profile file groups multiple completions for the same prompt:

```json
{
  "prompt": "Problem:\n...",
  "answer": "\\boxed{...}",
  "rollouts": [{"id":"r1","completion":"..."}]
}
```
