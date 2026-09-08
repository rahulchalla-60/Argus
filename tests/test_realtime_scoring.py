import sys
import time
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from database import Database
from feature_engine import FeatureEngine
from risk_engine import RiskEngine
from stream_simulator import StreamSimulator

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"


def test_mule_and_normal_profiles():
    re = RiskEngine(MODEL_PATH)
    fe = FeatureEngine(time_window=24)

    # 1. Known Mule Profile: Massive inbound stolen transfer ($1,000,000) -> fan_in=1
    fe.update_transaction("VICTIM_ACCOUNT", "MULE_ACCOUNT", amount=1000000.0, timestamp=10)
    mule_score = re.score_account("MULE_ACCOUNT", fe)
    print(f"Mule Account Score: {mule_score:.4f}")
    assert mule_score > 0.80, f"Expected mule score > 0.80, got {mule_score}"

    # 2. Legitimate Retail User: Small single payment ($120) -> fan_in=0, fan_out=1
    fe.update_transaction("NORMAL_USER", "MERCHANT_XYZ", amount=120.0, timestamp=10)
    normal_score = re.score_account("NORMAL_USER", fe)
    print(f"Normal User Score: {normal_score:.4f}")
    assert normal_score < 0.30, f"Expected normal score < 0.30, got {normal_score}"

    print("[PASS] Mule and Normal profile scoring verification.")


def test_realtime_scoring_stream_and_latency():
    re = RiskEngine(MODEL_PATH)
    fe = FeatureEngine(time_window=24)
    db = Database(":memory:")
    sim = StreamSimulator(DATASET_PATH, nrows=1000)

    scoring_latencies_ms = []

    print("\nStreaming 1,000 transactions and measuring scoring latency...")
    for txn in sim.stream():
        sender = txn["nameOrig"]
        receiver = txn["nameDest"]
        amount = float(txn["amount"])
        step = int(txn["step"])

        # 1. Feature Update
        fe.update_transaction(sender, receiver, amount=amount, timestamp=step)

        # 2. Score Sender (Measure inference time)
        t0 = time.perf_counter()
        re.score_account(sender, fe, db=db, step=step)
        t1 = time.perf_counter()
        scoring_latencies_ms.append((t1 - t0) * 1000.0)

        # 3. Score Receiver (Measure inference time)
        t0 = time.perf_counter()
        re.score_account(receiver, fe, db=db, step=step)
        t1 = time.perf_counter()
        scoring_latencies_ms.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(scoring_latencies_ms))
    p95_latency = float(np.percentile(scoring_latencies_ms, 95))
    max_latency = float(np.max(scoring_latencies_ms))

    print("\n--- Latency Benchmark Results (2,000 score_account evaluations) ---")
    print(f"Avg Latency : {avg_latency:.4f} ms  (Target: < 10.0 ms)")
    print(f"P95 Latency : {p95_latency:.4f} ms  (Target: < 10.0 ms)")
    print(f"Max Latency : {max_latency:.4f} ms  (Target: < 20.0 ms)")

    # Assert Latency SLA
    assert avg_latency < 10.0, f"Average latency exceeded SLA: {avg_latency:.4f} ms"
    assert p95_latency < 10.0, f"P95 latency exceeded SLA: {p95_latency:.4f} ms"

    # Verify Option B SQLite Persistence
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM risk_scores")
    stored_count = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(risk_score) FROM risk_scores")
    min_score_row = cursor.fetchone()
    min_stored_score = min_score_row[0] if min_score_row and min_score_row[0] is not None else 0.0

    print(f"\nOption B DB Persistence: {stored_count} accounts saved with risk >= 0.30 (Min stored score: {min_stored_score:.4f})")
    if stored_count > 0:
        assert min_stored_score >= 0.30

    high_risk_list = db.get_high_risk_accounts(min_score=0.80)
    print(f"High-Risk Accounts Flagged (Score >= 0.80): {len(high_risk_list)}")

    db.close()
    print("\n[PASS] Phase 3.4 Real-Time Scoring & Latency Benchmark passed.")


if __name__ == "__main__":
    test_mule_and_normal_profiles()
    test_realtime_scoring_stream_and_latency()
