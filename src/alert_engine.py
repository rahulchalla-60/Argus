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
        db: Database | None = None
    ):
        self.threshold = threshold
        self.rule_engine = rule_engine or RuleEngine()
        self.db = db
        self.alerts: list[Alert] = []

    def evaluate_and_alert(
        self,
        account_id: str,
        risk_score: float,
        timestamp: int,
        features: dict[str, Any] | None = None
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
            reasons=reasons
        )

        self.alerts.append(alert)

        # 3. Optional SQLite Persistence
        if self.db:
            with self.db.conn:
                self.db.conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        account_id TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        step INTEGER NOT NULL,
                        reasons TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.db.conn.execute("""
                    INSERT INTO alerts (account_id, risk_score, step, reasons)
                    VALUES (?, ?, ?, ?)
                """, (alert.account_id, alert.risk_score, alert.timestamp, json.dumps(alert.reasons)))

        return alert

    def get_alerts(self, limit: int | None = None) -> list[Alert]:
        return self.alerts[-limit:] if limit else list(self.alerts)

    def count(self) -> int:
        return len(self.alerts)
