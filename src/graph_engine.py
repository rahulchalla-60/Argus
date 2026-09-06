from collections import deque
import networkx as nx
from typing import Any
from database import Database
from feature_engine import FeatureEngine, AccountFeatures


class GraphEngine:
    def __init__(
        self,
        window_size: int = 50000,
        db: Database | None = None,
        feature_engine: FeatureEngine | None = None
    ):
        self.graph = nx.MultiDiGraph()
        self.window_size = window_size
        self.window = deque()
        self.db = db
        self.feature_engine = feature_engine

    def add_transaction(self, sender: str, receiver: str, amount: float, timestamp: int, **metadata):
        # 1. Update Graph & sliding window
        self.graph.add_node(sender)
        self.graph.add_node(receiver)

        edge_key = self.graph.add_edge(
            sender,
            receiver,
            amount=amount,
            timestamp=timestamp,
            **metadata
        )
        self.window.append((sender, receiver, edge_key))

        if len(self.window) > self.window_size:
            self._evict_oldest()

        # 2. Incrementally update feature engine in O(1)
        if self.feature_engine:
            self.feature_engine.update_transaction(sender, receiver, amount=amount, timestamp=timestamp)

    def _evict_oldest(self):
        old_sender, old_receiver, old_key = self.window.popleft()
        if self.graph.has_edge(old_sender, old_receiver, key=old_key):
            self.graph.remove_edge(old_sender, old_receiver, key=old_key)

        if self.graph.has_node(old_sender) and self.graph.degree(old_sender) == 0:
            self.graph.remove_node(old_sender)
        if self.graph.has_node(old_receiver) and self.graph.degree(old_receiver) == 0:
            self.graph.remove_node(old_receiver)

    def warm_start(self, limit: int | None = None) -> int:
        if not self.db:
            return 0
        fetch_limit = limit or self.window_size
        rows = self.db.get_recent_transactions(limit=fetch_limit)
        for r in rows:
            self.add_transaction(
                sender=r["nameOrig"],
                receiver=r["nameDest"],
                amount=r["amount"],
                timestamp=r["step"],
                type=r["type"],
                oldbalanceOrg=r["oldbalanceOrg"],
                newbalanceOrig=r["newbalanceOrig"],
                oldbalanceDest=r["oldbalanceDest"],
                newbalanceDest=r["newbalanceDest"],
                isFraud=r["isFraud"],
                isFlaggedFraud=r["isFlaggedFraud"]
            )
        return len(rows)

    def get_node_count(self) -> int:
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_neighbors(self, account_id: str) -> dict[str, list[str]]:
        if account_id not in self.graph:
            return {"incoming": [], "outgoing": []}
        return {
            "incoming": list(self.graph.predecessors(account_id)),
            "outgoing": list(self.graph.successors(account_id)),
        }

    def get_transactions(self, account_id: str, include_history: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        if include_history and self.db:
            return self.db.get_account_history(account_id, limit=limit)

        if account_id not in self.graph:
            return []

        txns = []
        for src, tgt, data in self.graph.in_edges(account_id, data=True):
            txns.append({"sender": src, "receiver": tgt, **data})
        for src, tgt, data in self.graph.out_edges(account_id, data=True):
            txns.append({"sender": src, "receiver": tgt, **data})
        return txns

    def get_account_features(self, account_id: str) -> AccountFeatures | None:
        if self.feature_engine:
            return self.feature_engine.get_features(account_id)
        return None

    def get_account_features_dict(self, account_id: str) -> dict[str, Any] | None:
        acc = self.get_account_features(account_id)
        return acc.to_dict() if acc else None

    def get_subgraph(self, account_id: str, k_hops: int = 1) -> nx.MultiDiGraph:
        if account_id not in self.graph:
            return nx.MultiDiGraph()

        nodes = {account_id}
        frontier = {account_id}

        for _ in range(k_hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.predecessors(node))
                next_frontier.update(self.graph.successors(node))
            nodes.update(next_frontier)
            frontier = next_frontier

        return self.graph.subgraph(nodes).copy()
