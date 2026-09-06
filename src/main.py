from pathlib import Path
from stream_simulator import StreamSimulator
from graph_engine import GraphEngine
from visualize_graph import visualize_subgraph

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"
OUTPUT_HTML = BASE_DIR / "data" / "live_subgraph.html"


def main():
    print("=== Argus Phase 1.4: Stream -> Graph Live Integration ===")
    sim = StreamSimulator(DATASET_PATH, nrows=50)
    ge = GraphEngine(window_size=1000)

    target_account = None

    for i, txn in enumerate(sim.stream(), start=1):
        ge.add_transaction(
            sender=txn["nameOrig"],
            receiver=txn["nameDest"],
            amount=txn["amount"],
            timestamp=txn["step"],
            type=txn["type"],
            isFraud=txn["isFraud"]
        )
        if i == 1:
            target_account = txn["nameOrig"]

        if i % 10 == 0 or i == 50:
            print(f"Processed {i:2d} txns | Graph Nodes: {ge.get_node_count():3d} | Graph Edges: {ge.get_edge_count():3d}")

    # Extract 2-hop neighborhood for visualization POC
    subgraph = ge.get_subgraph(target_account, k_hops=2)
    print(f"\nExtracted 2-hop subgraph around '{target_account}': {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges.")

    visualize_subgraph(subgraph, title=f"Argus Subgraph — Neighborhood of {target_account}", output_html=OUTPUT_HTML)
    print(f"Interactive visualization saved to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()