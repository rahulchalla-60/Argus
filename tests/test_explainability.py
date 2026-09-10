import networkx as nx
from src.explainability import ExplainabilityEngine


def test_explain_account():
    engine = ExplainabilityEngine(risk_threshold=0.80)
    features = {
        "pass_through_ratio": 0.98,
        "forwarding_delay": 12,
        "neighbor_risk": 0.85,
        "total_received": 750000.0,
        "total_sent": 735000.0,
    }
    explanation = engine.explain_account(
        account_id="MULE_001",
        features=features,
        risk_score=0.92,
        rule_reasons=["High pass-through ratio (>95%)", "Rapid forwarding (<30s)"]
    )

    assert explanation["account_id"] == "MULE_001"
    assert explanation["risk_score"] == 0.92
    assert explanation["risk_level"] == "HIGH"
    assert len(explanation["contributing_factors"]) >= 3
    assert len(explanation["rule_reasons"]) == 2
    assert "HIGH RISK" in explanation["summary"]


def test_extract_flow_subgraph():
    engine = ExplainabilityEngine()
    G = nx.MultiDiGraph()

    # Flow: Payer_1 -> Mule_1 -> Cashout_1
    G.add_edge("Payer_1", "Mule_1", amount=10000.0, timestamp=100, type="TRANSFER")
    G.add_edge("Mule_1", "Cashout_1", amount=9800.0, timestamp=110, type="CASH_OUT")
    # Additional branch: Payer_2 -> Mule_1
    G.add_edge("Payer_2", "Mule_1", amount=5000.0, timestamp=105, type="TRANSFER")

    flow = engine.extract_flow_subgraph("Mule_1", G, k_hops=2)

    assert flow["target"] == "Mule_1"
    assert flow["node_count"] == 4
    assert flow["edge_count"] == 3

    roles = {n["account_id"]: n["role"] for n in flow["nodes"]}
    assert roles["Mule_1"] == "Focal Account"
    assert roles["Payer_1"] == "Payer / Source"
    assert roles["Payer_2"] == "Payer / Source"
    assert roles["Cashout_1"] == "Cashout / Sink"


def test_detect_clusters():
    engine = ExplainabilityEngine(risk_threshold=0.80)
    G = nx.MultiDiGraph()

    # Create a dense mule ring (A -> B -> C -> A)
    G.add_edge("A", "B", amount=50000)
    G.add_edge("B", "C", amount=49000)
    G.add_edge("C", "A", amount=48000)

    # Create another separate cluster (D -> E -> F)
    G.add_edge("D", "E", amount=1000)
    G.add_edge("E", "F", amount=1000)

    risk_scores = {
        "A": 0.95,
        "B": 0.90,
        "C": 0.85,
        "D": 0.10,
        "E": 0.05,
        "F": 0.02,
    }

    clusters = engine.detect_clusters(G, risk_scores=risk_scores, min_size=3)

    assert len(clusters) == 2
    # The first cluster should be the high-risk syndicate
    syndicate = clusters[0]
    assert syndicate["high_risk_accounts"] == 3
    assert syndicate["label"] == "Mule Ring / Smurfing Syndicate"
    assert set(syndicate["members"]) == {"A", "B", "C"}
    assert syndicate["avg_risk"] > 0.85
