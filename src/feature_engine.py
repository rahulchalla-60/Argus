from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Any


@dataclass
class AccountFeatures:
    account_id: str

    # Lifetime aggregates
    total_received: float = 0.0
    total_sent: float = 0.0
    unique_senders: set[str] = field(default_factory=set)
    unique_receivers: set[str] = field(default_factory=set)
    txn_count: int = 0

    # Timestamps & delays
    last_received_time: int | None = None
    last_sent_time: int | None = None
    forwarding_delay: int | None = None

    # Rolling window metrics
    rolling_received: float = 0.0
    rolling_sent: float = 0.0
    pass_through_ratio: float = 0.0

    # Sliding window queues: (timestamp, amount, counterparty)
    recent_incoming: Deque[tuple[int, float, str]] = field(default_factory=deque)
    recent_outgoing: Deque[tuple[int, float, str]] = field(default_factory=deque)

    def receive(self, timestamp: int, amount: float, sender: str):
        self.last_received_time = timestamp
        self.total_received += amount
        self.rolling_received += amount
        self.txn_count += 1
        self.unique_senders.add(sender)
        self.recent_incoming.append((timestamp, amount, sender))

    def send(self, timestamp: int, amount: float, receiver: str):
        self.last_sent_time = timestamp
        self.total_sent += amount
        self.rolling_sent += amount
        self.txn_count += 1
        self.unique_receivers.add(receiver)
        self.recent_outgoing.append((timestamp, amount, receiver))

        if self.last_received_time is not None:
            self.forwarding_delay = self.last_sent_time - self.last_received_time

    def prune(self, current_time: int, window: int):
        cutoff = current_time - window

        while self.recent_incoming and self.recent_incoming[0][0] < cutoff:
            _, amt, _ = self.recent_incoming.popleft()
            self.rolling_received -= amt

        while self.recent_outgoing and self.recent_outgoing[0][0] < cutoff:
            _, amt, _ = self.recent_outgoing.popleft()
            self.rolling_sent -= amt

        self.pass_through_ratio = (
            self.rolling_sent / self.rolling_received
            if self.rolling_received > 0
            else 0.0
        )

    @property
    def velocity(self) -> int:
        return len(self.recent_incoming) + len(self.recent_outgoing)

    @property
    def fan_in(self) -> int:
        return len({sender for _, _, sender in self.recent_incoming})

    @property
    def fan_out(self) -> int:
        return len({receiver for _, _, receiver in self.recent_outgoing})

    def get_lifetime_pass_through_ratio(self) -> float:
        return (self.total_sent / self.total_received) if self.total_received > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "total_received": self.total_received,
            "total_sent": self.total_sent,
            "rolling_received": self.rolling_received,
            "rolling_sent": self.rolling_sent,
            "pass_through_ratio": self.pass_through_ratio,
            "lifetime_pass_through_ratio": self.get_lifetime_pass_through_ratio(),
            "forwarding_delay": self.forwarding_delay,
            "velocity": self.velocity,
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "lifetime_fan_in": len(self.unique_senders),
            "lifetime_fan_out": len(self.unique_receivers),
            "txn_count": self.txn_count,
        }


class FeatureEngine:
    def __init__(self, time_window: int = 24):
        # time_window: window length in PaySim steps (1 step = 1 hour; default = 24 steps / 1 day)
        self.time_window = time_window
        self.features: dict[str, AccountFeatures] = {}

    def get_or_create(self, account_id: str) -> AccountFeatures:
        if account_id not in self.features:
            self.features[account_id] = AccountFeatures(account_id=account_id)
        return self.features[account_id]

    def update_transaction(self, sender: str, receiver: str, amount: float, timestamp: int):
        sender_acc = self.get_or_create(sender)
        sender_acc.send(timestamp, amount=amount, receiver=receiver)
        sender_acc.prune(timestamp, self.time_window)

        receiver_acc = self.get_or_create(receiver)
        receiver_acc.receive(timestamp, amount=amount, sender=sender)
        receiver_acc.prune(timestamp, self.time_window)

    def get_features(self, account_id: str) -> AccountFeatures | None:
        return self.features.get(account_id)

    def get_features_dict(self, account_id: str) -> dict[str, Any] | None:
        acc = self.get_features(account_id)
        return acc.to_dict() if acc else None

    def account_count(self) -> int:
        return len(self.features)
