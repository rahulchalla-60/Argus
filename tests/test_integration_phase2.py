import sys
import time
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from stream_simulator import StreamSimulator
from feature_engine import FeatureEngine
from graph_engine import GraphEngine

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"


def test_integration_and_latency():
    print("=== Phase 2.7 Integration & Latency Benchmark ===")
    sim = StreamSimulator(DATASET_PATH, nrows=1000)
    fe = FeatureEngine(time_window=24)
    ge = GraphEngine(window_size=50000, feature_engine=fe)

    latencies_ms = []

    for txn in sim.stream():
        t0 = time.perf_counter()
        ge.add_transaction(
            sender=txn["nameOrig"],
            receiver=txn["nameDest"],
            amount=txn["amount"],
            timestamp=txn["step"],
            type=txn["type"],
            isFraud=txn["isFraud"]
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_lat = float(np.mean(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    max_lat = float(np.max(latencies_ms))

    print(f"Processed: {len(latencies_ms)} transactions")
    print(f"Avg Latency : {avg_lat:.4f} ms")
    print(f"P95 Latency : {p95_lat:.4f} ms")
    print(f"Max Latency : {max_lat:.4f} ms")
    print(f"Graph Nodes : {ge.get_node_count()} | Edges: {ge.get_edge_count()}")
    print(f"Feature Store Accounts: {fe.account_count()}")

    # Target: < 5ms average latency
    assert avg_lat < 5.0, f"Average latency exceeded 5ms: {avg_lat:.4f} ms"
    assert p95_lat < 10.0, f"P95 latency exceeded 10ms: {p95_lat:.4f} ms"

    # Find busiest account (highest transaction count)
    busiest_id = max(fe.features.keys(), key=lambda acc_id: fe.features[acc_id].txn_count)
    busiest_features = ge.get_account_features_dict(busiest_id)
    print(f"\nBusiest Account: {busiest_id}")
    for k, v in busiest_features.items():
        print(f"  {k:28s}: {v}")

    print("\n[PASS] Phase 2.7 Integration and Latency Benchmark successfully passed.")


if __name__ == "__main__":
    test_integration_and_latency()
