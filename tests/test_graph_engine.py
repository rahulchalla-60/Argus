import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from database import Database
from graph_engine import GraphEngine


def test_base_graph_operations():
    ge = GraphEngine(window_size=100)

    transactions = [
        ("A", "B", 100.0, 1),
        ("B", "C", 200.0, 2),
        ("C", "D", 300.0, 3),
        ("D", "E", 400.0, 4),
        ("E", "F", 500.0, 5),
        ("F", "G", 600.0, 6),
        ("G", "H", 700.0, 7),
        ("H", "I", 800.0, 8),
        ("I", "J", 900.0, 9),
        ("J", "K", 1000.0, 10),
    ]

    for txn in transactions:
        ge.add_transaction(*txn)

    assert ge.get_node_count() == 11
    assert ge.get_edge_count() == 10

    neighbors_c = ge.get_neighbors("C")
    assert neighbors_c["incoming"] == ["B"]
    assert neighbors_c["outgoing"] == ["D"]

    txns_c = ge.get_transactions("C")
    assert len(txns_c) == 2

    subgraph = ge.get_subgraph("F", k_hops=2)
    assert subgraph.number_of_nodes() == 5
    assert subgraph.number_of_edges() == 4
    print("[PASS] Base graph operations.")


def test_sliding_window_eviction():
    ge = GraphEngine(window_size=3)

    ge.add_transaction("A", "B", 100.0, 1)
    ge.add_transaction("B", "C", 200.0, 2)
    ge.add_transaction("C", "D", 300.0, 3)

    assert ge.get_edge_count() == 3
    assert ge.get_node_count() == 4

    ge.add_transaction("D", "E", 400.0, 4)
    assert ge.get_edge_count() == 3
    assert ge.get_node_count() == 4
    assert "A" not in ge.graph
    assert "B" in ge.graph
    print("[PASS] Sliding window eviction.")


def test_database_and_warm_start():
    with Database(":memory:") as db:
        db.insert_transaction(
            step=1, type_="TRANSFER", amount=500.0, name_orig="USER1", name_dest="USER2",
            oldbalance_org=1000.0, newbalance_orig=500.0, oldbalance_dest=0.0, newbalance_dest=500.0
        )
        db.insert_transaction(
            step=2, type_="CASH_OUT", amount=450.0, name_orig="USER2", name_dest="MERCHANT",
            oldbalance_org=500.0, newbalance_orig=50.0, oldbalance_dest=100.0, newbalance_dest=550.0
        )
        # Legitimate duplicate flow allowed
        db.insert_transaction(
            step=1, type_="TRANSFER", amount=500.0, name_orig="USER1", name_dest="USER2"
        )
        assert db.count() == 3

        # Test GraphEngine warm start from DB
        ge = GraphEngine(window_size=10, db=db)
        loaded = ge.warm_start()
        assert loaded == 3
        assert ge.get_edge_count() == 3
        assert ge.get_node_count() == 3

        history = ge.get_transactions("USER2", include_history=True)
        assert len(history) == 3

    print("[PASS] Database persistence and warm start.")


if __name__ == "__main__":
    test_base_graph_operations()
    test_sliding_window_eviction()
    test_database_and_warm_start()
    print("\nAll tests passed successfully.")
