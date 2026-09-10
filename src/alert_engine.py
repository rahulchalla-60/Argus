import json
from dataclasses import dataclass, field, asdict
from typing import Any
from rule_engine import RuleEngine
from database import Database


@dataclass
class Alert:
    account_id: str
    risk_score: float
    timestamp: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertEngine:
    def __init__(
        self,
        threshold: float = 0.80,
        rule_engine: RuleEngine | None = None,
        db: Database | None = None,
        risk_engine: Any = None,
        risk_threshold: float | None = None,
    ):
        self.threshold = risk_threshold if risk_threshold is not None else threshold
        self.rule_engine = rule_engine or RuleEngine()
        self.risk_engine = risk_engine
        self.db = db
        self.alerts: list[Alert] = []

    def evaluate_and_alert(
        self,
        account_id: str,
        risk_score: float,
        timestamp: int,
        features: dict[str, Any] | None = None,
    ) -> Alert | None:
        reasons = []

        # 1. Extract rule-based reasons if features are provided
        rule_res = None
        if features:
            rule_res = self.rule_engine.evaluate(features)
            reasons.extend(rule_res.get("reasons", []))

        # 2. Check if alert condition is met (ML threshold or Rule HIGH)
        is_ml_triggered = risk_score >= self.threshold
        is_rule_triggered = rule_res and rule_res.get("risk_level") == "HIGH"

        if not (is_ml_triggered or is_rule_triggered):
            return None

        if is_ml_triggered and f"ML risk score ({risk_score:.2f}) >= {self.threshold}" not in reasons:
            reasons.insert(0, f"ML risk score ({risk_score:.2f}) >= {self.threshold}")

        alert = Alert(
            account_id=account_id,
            risk_score=float(risk_score),
            timestamp=int(timestamp),
            reasons=reasons,
        )

        self.alerts.append(alert)

        # 3. Optional SQLite Persistence
        if self.db and hasattr(self.db, "save_alert"):
            try:
                self.db.save_alert(alert.account_id, alert.risk_score, alert.timestamp, alert.reasons)
            except Exception:
                pass

        return alert

    def process_transaction(
        self,
        txn: dict,
        orig_features: dict[str, Any] | None = None,
        dest_features: dict[str, Any] | None = None,
    ) -> list[Alert]:
        alerts_created = []
        step = int(txn.get("step", txn.get("timestamp", 0)))

        # Score & evaluate sender
        if orig_features:
            sender = txn.get("nameOrig", orig_features.get("account_id"))
            score = 0.0
            if self.risk_engine and hasattr(self.risk_engine, "score_account"):
                score = self.risk_engine.score_account(orig_features)
            alt = self.evaluate_and_alert(sender, score, step, features=orig_features)
            if alt:
                alerts_created.append(alt)

        # Score & evaluate receiver
        if dest_features:
            receiver = txn.get("nameDest", dest_features.get("account_id"))
            score = 0.0
            if self.risk_engine and hasattr(self.risk_engine, "score_account"):
                score = self.risk_engine.score_account(dest_features)
            alt = self.evaluate_and_alert(receiver, score, step, features=dest_features)
            if alt:
                alerts_created.append(alt)

        return alerts_created

    @property
    def alert_queue(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.alerts]

    def get_alerts(self, limit: int | None = None) -> list[Alert]:
        return self.alerts[-limit:] if limit else list(self.alerts)

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.get_alerts(limit=limit)]

    def count(self) -> int:
        return len(self.alerts)
