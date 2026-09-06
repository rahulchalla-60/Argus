import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_engine import FeatureEngine, AccountFeatures


def test_feature_engine_setup():
    fe = FeatureEngine(time_window=1)

    assert fe.get_features("USER_A") is None
    assert fe.account_count() == 0

    feat_a = fe.get_or_create("USER_A")
    assert feat_a.account_id == "USER_A"
    assert feat_a.total_received == 0.0
    assert feat_a.total_sent == 0.0
    assert feat_a.forwarding_delay is None
    print("[PASS] FeatureEngine setup & AccountFeatures state.")


def test_forwarding_delay_cases():
    fe = FeatureEngine(time_window=1)

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

    # Case 4: Multiple receives before send (D receives at 100, 110, 120; sends at 125) -> delay = 125 - 120 = 5
    fe.update_transaction("SRC1", "D", amount=100.0, timestamp=100)
    fe.update_transaction("SRC2", "D", amount=200.0, timestamp=110)
    fe.update_transaction("SRC3", "D", amount=300.0, timestamp=120)
    fe.update_transaction("D", "SINK", amount=580.0, timestamp=125)

    feat_d = fe.get_features("D")
    assert feat_d.forwarding_delay == 5

    print("[PASS] Forwarding delay edge cases & stream updates.")


if __name__ == "__main__":
    test_feature_engine_setup()
    test_forwarding_delay_cases()
    print("\nAll feature engine tests passed successfully.")
