from __future__ import annotations

from typing import Any, Dict

from ..rewards import BETRewardConfig


def reward_config_from_dict(config: Dict[str, Any]) -> BETRewardConfig:
    reward = config.get('reward', {})
    return BETRewardConfig(**{k: v for k, v in reward.items() if hasattr(BETRewardConfig, k)})
