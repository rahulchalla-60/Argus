import sys
from pathlib import Path
import networkx as nx

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from risk_propagation import propagate_risk, propagate_risk_detailed, compute_neighbor_risk
from visualize_graph import visualize_risk_propagation

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_HTML = BASE_DIR / "data" / "propagation_poc.html"


def test_linear_chain_propagation():
    # Linear chain: A -> B -> C -> D -> E
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", amount=100.0, timestamp=1)
    g.add_edge("B", "C", amount=100.0, timestamp=2)
    g.add_edge("C", "D", amount=100.0, timestamp=3)
    g.add_edge("D", "E", amount=100.0, timestamp=4)

    # 1. Downstream propagation from A (Risk = 1.0) with decay = 0.5, max_hops = 3
    risk_map = propagate_risk(g, flagged_accounts={"A": 1.0}, decay_factor=0.5, max_hops=3, direction="downstream")

    assert risk_map["A"] == 1.0
    assert risk_map["B"] == 0.5
    assert risk_map["C"] == 0.25
    assert risk_map["D"] == 0.125
    assert "E" not in risk_map  # Beyond 3 hops

    # Detailed with hop distances
    details = propagate_risk_detailed(g, flagged_accounts={"A": 1.0}, decay_factor=0.5, max_hops=3)
    assert details["A"]["hop"] == 0
    assert details["B"]["hop"] == 1
    assert details["C"]["hop"] == 2
    assert details["D"]["hop"] == 3

    print("[PASS] Linear chain BFS risk decay.")


def test_multi_source_convergence():
    # Converging graph: A -> B, X -> B
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", amount=50000.0, timestamp=1)
    g.add_edge("X", "B", amount=30000.0, timestamp=2)
    g.add_edge("B", "C", amount=75000.0, timestamp=3)

    risk_map = propagate_risk(
        g,
        flagged_accounts={"A": 1.0, "X": 0.8},
        decay_factor=0.5,
        max_hops=2,
        direction="downstream"
    )

    # B receives max(1.0 * 0.5, 0.8 * 0.5) = 0.5
    assert risk_map["A"] == 1.0
    assert risk_map["X"] == 0.8
    assert risk_map["B"] == 0.5
    assert risk_map["C"] == 0.25

    print("[PASS] Multi-source convergence (max risk selection).")


def test_neighbor_risk_and_visualization():
    g = nx.MultiDiGraph()
    g.add_edge("ACC_A", "ACC_B", amount=100.0)
    g.add_edge("ACC_B", "ACC_C", amount=100.0)

    risk_map = {"ACC_A": 1.0, "ACC_B": 0.5, "ACC_C": 0.25}

    # B's neighbors are A (1.0) and C (0.25) -> max neighbor risk is 1.0
    nbr_risk_b = compute_neighbor_risk(g, "ACC_B", risk_map)
    assert nbr_risk_b == 1.0

    # C's neighbor is B (0.5) -> max neighbor risk is 0.5
    nbr_risk_c = compute_neighbor_risk(g, "ACC_C", risk_map)
    assert nbr_risk_c == 0.5

    # Test visualization output
    details = propagate_risk_detailed(g, flagged_accounts={"ACC_A": 1.0}, decay_factor=0.5, max_hops=2)
    visualize_risk_propagation(g, details, title="Argus Risk Propagation POC", output_html=OUTPUT_HTML)
    assert OUTPUT_HTML.exists()

    print(f"[PASS] Neighbor risk calculation and visualization generated at: {OUTPUT_HTML}")


if __name__ == "__main__":
    test_linear_chain_propagation()
    test_multi_source_convergence()
    test_neighbor_risk_and_visualization()
    print("\nAll Phase 4.1 Risk Propagation tests passed successfully.")
