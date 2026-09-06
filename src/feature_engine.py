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
    forwarding_delay: int | None = None

    # Sliding window queues: (timestamp, amount, counterparty)
    recent_incoming: deque = field(default_factory=deque)
    recent_outgoing: deque = field(default_factory=deque)
    txn_count: int = 0

    def receive(self, timestamp: int, amount: float = 0.0, sender: str = ""):
        self.last_received_time = timestamp
        self.total_received += amount
        self.txn_count += 1
        self.recent_incoming.append((timestamp, amount, sender))

    def send(self, timestamp: int, amount: float = 0.0, receiver: str = ""):
        self.last_sent_time = timestamp
        self.total_sent += amount
        self.txn_count += 1
        self.recent_outgoing.append((timestamp, amount, receiver))

        # ponytail: forwarding_delay computed in O(1) time when funds move out
        if self.last_received_time is not None:
            self.forwarding_delay = self.last_sent_time - self.last_received_time


class FeatureEngine:
    def __init__(self, time_window: int = 1):
        # time_window: window length in PaySim steps (1 step = 1 hour)
        self.time_window = time_window
        self.features: dict[str, AccountFeatures] = {}

    def get_or_create(self, account_id: str) -> AccountFeatures:
        if account_id not in self.features:
            self.features[account_id] = AccountFeatures(account_id=account_id)
        return self.features[account_id]

    def update_transaction(self, sender: str, receiver: str, amount: float, timestamp: int):
        self.get_or_create(sender).send(timestamp, amount=amount, receiver=receiver)
        self.get_or_create(receiver).receive(timestamp, amount=amount, sender=sender)

    def get_features(self, account_id: str) -> AccountFeatures | None:
        return self.features.get(account_id)

    def account_count(self) -> int:
        return len(self.features)
