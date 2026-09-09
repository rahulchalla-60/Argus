import sys
import json
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from alert_engine import AlertEngine, Alert
from database import Database
from rule_engine import RuleEngine
from feature_engine import FeatureEngine
from risk_engine import RiskEngine

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"


def test_alert_generation_and_queue():
    db = Database(":memory:")
    alert_engine = AlertEngine(threshold=0.80, db=db)

    # 1. Normal retail user -> No alert generated
    normal_features = {
        "pass_through_ratio": 0.0,
        "forwarding_delay": -1,
        "velocity": 1,
        "fan_in": 0,
        "fan_out": 1,
        "counterparties": 1,
        "total_received": 0.0,
        "total_sent": 150.0
    }
    alert_normal = alert_engine.evaluate_and_alert(
        account_id="NORMAL_CUST_1",
        risk_score=0.15,
        timestamp=10,
        features=normal_features
    )
    assert alert_normal is None
    assert alert_engine.count() == 0

    # 2. Injected Mule Account -> High risk score + High pass-through ratio
    mule_features = {
        "pass_through_ratio": 0.99,
        "forwarding_delay": 5,
        "velocity": 8,
        "fan_in": 1,
        "fan_out": 1,
        "counterparties": 2,
        "total_received": 1000000.0,
        "total_sent": 990000.0
    }
    alert_mule = alert_engine.evaluate_and_alert(
        account_id="MULE_ACC_888",
        risk_score=0.96,
        timestamp=15,
        features=mule_features
    )

    assert isinstance(alert_mule, Alert)
    assert alert_mule.account_id == "MULE_ACC_888"
    assert alert_mule.risk_score == 0.96
    assert alert_mule.timestamp == 15
    assert len(alert_mule.reasons) >= 2
    assert any("pass_through_ratio" in r for r in alert_mule.reasons)

    # 3. Verify in-memory queue
    assert alert_engine.count() == 1
    recent_alerts = alert_engine.get_alerts()
    assert len(recent_alerts) == 1
    assert recent_alerts[0].account_id == "MULE_ACC_888"

    # 4. Verify SQLite persistence
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM alerts WHERE account_id = ?", ("MULE_ACC_888",))
    row = cursor.fetchone()
    assert row is not None
    assert row["risk_score"] == 0.96
    assert row["step"] == 15
    db_reasons = json.loads(row["reasons"])
    assert isinstance(db_reasons, list)
    assert len(db_reasons) >= 2

    db.close()
    print("[PASS] AlertEngine generation, reasons, queue, and SQLite persistence.")


def test_end_to_end_mule_injection_alert():
    # End-to-end: FeatureEngine -> RiskEngine -> AlertEngine
    fe = FeatureEngine(time_window=24)
    re = RiskEngine(MODEL_PATH)
    db = Database(":memory:")
    alert_engine = AlertEngine(threshold=0.80, db=db)

    # Inject money mule pattern:
    # 1. Victim transfers $1,000,000 to Mule at step 5
    fe.update_transaction("VICTIM_123", "MULE_XYZ", amount=1000000.0, timestamp=5)
    # 2. Mule cashes out $990,000 at step 5 (immediate cash out)
    fe.update_transaction("MULE_XYZ", "SINK_AGENT", amount=990000.0, timestamp=5)

    # Score mule account
    mule_score = re.score_account("MULE_XYZ", fe, db=db, step=5)
    mule_feat = fe.get_features_dict("MULE_XYZ")

    # Evaluate for alert
    alert = alert_engine.evaluate_and_alert(
        account_id="MULE_XYZ",
        risk_score=mule_score,
        timestamp=5,
        features=mule_feat
    )

    assert alert is not None
    assert alert.account_id == "MULE_XYZ"
    assert alert.risk_score > 0.80
    print(f"Generated Alert: {alert.to_dict()}")

    db.close()
    print("[PASS] End-to-End Mule Injection Alert successfully generated.")


if __name__ == "__main__":
    test_alert_generation_and_queue()
    test_end_to_end_mule_injection_alert()
