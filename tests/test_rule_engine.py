import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from rule_engine import RuleEngine
from feature_engine import AccountFeatures


def test_rule_engine_evaluations():
    re = RuleEngine()

    # 1. Classic Mule: ratio=0.98, delay=8s -> HIGH
    mule_feat = {
        "pass_through_ratio": 0.98,
        "forwarding_delay": 8,
        "fan_in": 3,
        "velocity": 2
    }
    res_mule = re.evaluate(mule_feat)
    assert res_mule["risk_level"] == "HIGH"
    assert res_mule["risk_score"] >= 70
    assert len(res_mule["reasons"]) == 2
    assert "pass_through_ratio" in res_mule["reasons"][0]

    # 2. Single Medium trigger: velocity=7 -> MEDIUM
    burst_feat = {
        "pass_through_ratio": 0.20,
        "forwarding_delay": 120,
        "fan_in": 2,
        "velocity": 7
    }
    res_burst = re.evaluate(burst_feat)
    assert res_burst["risk_level"] == "MEDIUM"
    assert res_burst["risk_score"] == 30
    assert len(res_burst["reasons"]) == 1

    # 3. Two Medium triggers: delay=15s, fan_in=25 -> HIGH
    fan_in_feat = {
        "pass_through_ratio": 0.50,
        "forwarding_delay": 15,
        "fan_in": 25,
        "velocity": 3
    }
    res_fan_in = re.evaluate(fan_in_feat)
    assert res_fan_in["risk_level"] == "HIGH"
    assert res_fan_in["risk_score"] == 60
    assert len(res_fan_in["reasons"]) == 2

    # 4. Normal account -> LOW
    normal_feat = {
        "pass_through_ratio": 0.10,
        "forwarding_delay": 500,
        "fan_in": 1,
        "velocity": 1
    }
    res_normal = re.evaluate(normal_feat)
    assert res_normal["risk_level"] == "LOW"
    assert res_normal["risk_score"] == 0
    assert len(res_normal["reasons"]) == 0

    # 5. AccountFeatures object evaluation
    acc = AccountFeatures(account_id="ACC_MULE")
    acc.receive(timestamp=10, amount=10000.0, sender="A")
    acc.send(timestamp=18, amount=9900.0, receiver="B")
    acc.prune(current_time=18, window=24)

    res_acc = re.evaluate(acc)
    assert res_acc["risk_level"] == "HIGH"
    assert res_acc["risk_score"] == 100  # 70 (ratio > 0.95) + 30 (delay 8s < 30)

    print("[PASS] RuleEngine evaluations and risk scoring passed.")


if __name__ == "__main__":
    test_rule_engine_evaluations()
