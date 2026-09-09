import sys
from pathlib import Path
import networkx as nx

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from risk_propagation import propagate_risk_detailed, compute_neighbor_risk
from visualize_graph import visualize_risk_propagation
from risk_engine import RiskEngine

BASE_DIR = Path(__file__).resolve().parent.parent
CHAIN_HTML = BASE_DIR / "data" / "propagation_chain.html"
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"


def test_propagation_chain_visualization():
    # 4-tier chain: Payer (A) -> Mule 1 (B) -> Mule 2 (C) -> Cashout (D)
    g = nx.MultiDiGraph()
    g.add_edge("Orig_Payer_A", "Mule_1_B", amount=500000.0, timestamp=1)
    g.add_edge("Mule_1_B", "Mule_2_C", amount=490000.0, timestamp=2)
    g.add_edge("Mule_2_C", "Cashout_D", amount=480000.0, timestamp=3)

    # Propagate from seed A (1.0)
    details = propagate_risk_detailed(g, {"Orig_Payer_A": 1.0}, decay_factor=0.5, max_hops=3, direction="downstream")

    assert details["Orig_Payer_A"]["risk"] == 1.0
    assert details["Orig_Payer_A"]["hop"] == 0
    assert details["Mule_1_B"]["risk"] == 0.5
    assert details["Mule_1_B"]["hop"] == 1
    assert details["Mule_2_C"]["risk"] == 0.25
    assert details["Mule_2_C"]["hop"] == 2
    assert details["Cashout_D"]["risk"] == 0.125
    assert details["Cashout_D"]["hop"] == 3

    # Generate interactive HTML visualization
    visualize_risk_propagation(g, details, title="Argus — Laundering Chain Risk Propagation", output_html=CHAIN_HTML)
    assert CHAIN_HTML.exists()
    print(f"[PASS] Propagation chain visualization generated at: {CHAIN_HTML}")


def test_neighbor_risk_feature_inference():
    if not MODEL_PATH.exists():
        print("[SKIP] Model not trained yet.")
        return

    re = RiskEngine(MODEL_PATH)
    
    # High-risk account with elevated neighbor risk
    feat_with_neighbor = {
        "forwarding_delay": 5,
        "velocity": 5,
        "fan_in": 1,
        "fan_out": 1,
        "pass_through_ratio": 0.98,
        "counterparties": 2,
        "total_received": 500000.0,
        "total_sent": 490000.0,
        "neighbor_risk": 0.95
    }
    score = re.predict_risk(feat_with_neighbor)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    print(f"[PASS] Graph-Enhanced 9-feature inference test passed (Score: {score:.4f}).")


if __name__ == "__main__":
    test_propagation_chain_visualization()
    test_neighbor_risk_feature_inference()
    print("\nAll Phase 4 unit tests passed successfully.")
