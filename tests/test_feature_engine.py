import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_engine import FeatureEngine, AccountFeatures


def test_feature_engine_setup():
    fe = FeatureEngine(time_window=24)

    assert fe.get_features("USER_A") is None
    assert fe.account_count() == 0

    feat_a = fe.get_or_create("USER_A")
    assert feat_a.account_id == "USER_A"
    assert feat_a.total_received == 0.0
    assert feat_a.total_sent == 0.0
    assert feat_a.pass_through_ratio == 0.0
    assert feat_a.forwarding_delay is None
    assert feat_a.velocity == 0
    assert feat_a.fan_in == 0
    assert feat_a.fan_out == 0
    print("[PASS] FeatureEngine setup & AccountFeatures state.")


def test_forwarding_delay_cases():
    fe = FeatureEngine(time_window=24)

    # Case 1: Account only receives (A -> B at step 10)
    fe.update_transaction("A", "B", amount=500.0, timestamp=10)
    feat_b = fe.get_features("B")
    assert feat_b.last_received_time == 10
    assert feat_b.last_sent_time is None
    assert feat_b.forwarding_delay is None

    # Case 2: Account only sends (A has only sent so far)
    feat_a = fe.get_features("A")
    assert feat_a.last_sent_time == 10
    assert feat_a.last_received_time is None
    assert feat_a.forwarding_delay is None

    # Case 3: Receive then send (B sends to C at step 25) -> delay = 25 - 10 = 15
    fe.update_transaction("B", "C", amount=480.0, timestamp=25)
    assert feat_b.last_sent_time == 25
    assert feat_b.forwarding_delay == 15

    print("[PASS] Forwarding delay edge cases.")


def test_pass_through_ratio():
    fe = FeatureEngine(time_window=24)

    fe.update_transaction("SRC", "MULE", amount=10000.0, timestamp=1)
    fe.update_transaction("MULE", "SINK", amount=9800.0, timestamp=2)

    mule = fe.get_features("MULE")
    assert mule.total_received == 10000.0
    assert mule.total_sent == 9800.0
    assert round(mule.get_lifetime_pass_through_ratio(), 2) == 0.98
    assert round(mule.pass_through_ratio, 2) == 0.98

    print("[PASS] Pass-through ratio.")


def test_fan_in_fan_out_and_velocity():
    fe = FeatureEngine(time_window=24)

    # 1. Fan-In: 5 unique accounts send to B at timestamp 1
    for i, sender in enumerate(["A", "C", "D", "E", "F"]):
        fe.update_transaction(sender, "B", amount=100.0, timestamp=1)

    feat_b = fe.get_features("B")
    assert feat_b.fan_in == 5
    assert feat_b.velocity == 5
    assert feat_b.fan_out == 0

    # 2. Fan-Out: B sends to 4 unique accounts at timestamp 2
    for receiver in ["R1", "R2", "R3", "R4"]:
        fe.update_transaction("B", receiver, amount=50.0, timestamp=2)

    assert feat_b.fan_out == 4
    assert feat_b.velocity == 9  # 5 in + 4 out

    # 3. Add 1 more transaction to reach 10 total transactions
    fe.update_transaction("B", "R1", amount=25.0, timestamp=3)
    assert feat_b.velocity == 10
    assert feat_b.fan_out == 4  # R1 was already a unique receiver

    # 4. Window Expiry: at timestamp 30 (cutoff = 30 - 24 = 6), txns at t=1, 2, 3 expire
    fe.update_transaction("NEW_SENDER", "B", amount=500.0, timestamp=30)
    assert feat_b.fan_in == 1
    assert feat_b.fan_out == 0
    assert feat_b.velocity == 1
    # Lifetime counts remain preserved
    assert len(feat_b.unique_senders) == 6  # A, C, D, E, F, NEW_SENDER
    assert len(feat_b.unique_receivers) == 4  # R1, R2, R3, R4

    # 5. to_dict verification
    d = fe.get_features_dict("B")
    assert d["account_id"] == "B"
    assert d["velocity"] == 1
    assert d["fan_in"] == 1
    assert d["lifetime_fan_in"] == 6

    print("[PASS] Fan-in, fan-out, velocity, and window expiry.")


if __name__ == "__main__":
    test_feature_engine_setup()
    test_forwarding_delay_cases()
    test_pass_through_ratio()
    test_fan_in_fan_out_and_velocity()
    print("\nAll feature engine tests passed successfully.")
