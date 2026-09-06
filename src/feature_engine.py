from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AccountFeatures:
    account_id: str
    total_received: float = 0.0
    total_sent: float = 0.0
    last_received_time: int | None = None
    last_sent_time: int | None = None

    # Sliding window queues: (timestamp, amount, counterparty)
    recent_incoming: deque = field(default_factory=deque)
    recent_outgoing: deque = field(default_factory=deque)


class FeatureEngine:
    def __init__(self, time_window: int = 1):
        # time_window: window length in PaySim steps (1 step = 1 hour)
        self.time_window = time_window
        self.features: dict[str, AccountFeatures] = {}

    def get_or_create(self, account_id: str) -> AccountFeatures:
        if account_id not in self.features:
            self.features[account_id] = AccountFeatures(account_id=account_id)
        return self.features[account_id]

    def get_features(self, account_id: str) -> AccountFeatures | None:
        return self.features.get(account_id)

    def account_count(self) -> int:
        return len(self.features)
