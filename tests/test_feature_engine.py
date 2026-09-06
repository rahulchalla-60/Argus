import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_engine import FeatureEngine, AccountFeatures


def test_feature_engine_setup():
    fe = FeatureEngine(time_window=1)

    # 1. Non-existent account returns None
    assert fe.get_features("USER_A") is None
    assert fe.account_count() == 0

    # 2. Get or create initializes AccountFeatures
    feat_a = fe.get_or_create("USER_A")
    assert isinstance(feat_a, AccountFeatures)
    assert feat_a.account_id == "USER_A"
    assert feat_a.total_received == 0.0
    assert feat_a.total_sent == 0.0
    assert feat_a.last_received_time is None
    assert feat_a.last_sent_time is None
    assert len(feat_a.recent_incoming) == 0
    assert len(feat_a.recent_outgoing) == 0
    assert fe.account_count() == 1

    # 3. Modify and retrieve
    feat_a.total_received += 5000.0
    feat_a.last_received_time = 1
    feat_a.recent_incoming.append((1, 5000.0, "USER_B"))

    retrieved = fe.get_features("USER_A")
    assert retrieved is not None
    assert retrieved.total_received == 5000.0
    assert retrieved.last_received_time == 1
    assert len(retrieved.recent_incoming) == 1

    print("[PASS] FeatureEngine setup & AccountFeatures state.")


if __name__ == "__main__":
    test_feature_engine_setup()
