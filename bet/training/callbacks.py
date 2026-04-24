from __future__ import annotations

from typing import Any, Dict

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    class TrainerCallback:  # type: ignore
        pass


class BETConsoleCallback(TrainerCallback):
    """Compact console logger for reward-heavy GRPO runs."""

    def on_log(self, args, state, control, logs: Dict[str, Any] | None = None, **kwargs):  # pragma: no cover
        if not logs or not getattr(state, 'is_world_process_zero', True):
            return
        keys = [k for k in logs if k.startswith('rewards/') or k in {'loss', 'grad_norm', 'learning_rate'}]
        if not keys:
            return
        print('\n' + '=' * 72)
        print(f"BET training step {getattr(state, 'global_step', 0)}")
        for k in sorted(keys):
            v = logs[k]
            if isinstance(v, float):
                print(f"  {k:36s}: {v: .6f}")
            else:
                print(f"  {k:36s}: {v}")
        print('=' * 72 + '\n')
