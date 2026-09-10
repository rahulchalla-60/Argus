from typing import Any
import networkx as nx
from networkx.algorithms.community import louvain_communities


class ExplainabilityEngine:
    def __init__(self, risk_threshold: float = 0.80):
        self.risk_threshold = risk_threshold

    def explain_account(
        self,
        account_id: str,
        features: dict[str, Any],
        risk_score: float,
        rule_reasons: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Generates human-readable contributing factors, ranking factors by risk severity.
        """
        contributing_factors = []
        reasons = list(rule_reasons or [])

        # 1. Pass-Through Ratio Analysis
        ratio = features.get("pass_through_ratio", 0.0)
        if ratio >= 0.90:
            contributing_factors.append({
                "factor": "Pass-Through Ratio",
                "value": f"{ratio * 100:.1f}%",
                "severity": "HIGH",
                "detail": f"{ratio * 100:.1f}% of received funds were forwarded out (classic mule behavior)."
            })

        # 2. Forwarding Delay Analysis
        delay = features.get("forwarding_delay")
        if delay is not None and delay >= 0 and delay <= 30:
            contributing_factors.append({
                "factor": "Rapid Forwarding Delay",
                "value": f"{delay}s",
                "severity": "HIGH",
                "detail": f"Funds were forwarded within {delay} seconds of receipt (rapid liquidation)."
            })

        # 3. Neighbor Risk Exposure
        neighbor_risk = features.get("neighbor_risk", 0.0)
        if neighbor_risk >= 0.50:
            contributing_factors.append({
                "factor": "Risky Counterparty Proximity",
                "value": f"{neighbor_risk:.2f}",
                "severity": "HIGH" if neighbor_risk >= 0.80 else "MEDIUM",
                "detail": f"Account has direct transaction links to confirmed high-risk accounts ({neighbor_risk:.2f})."
            })

        # 4. Inbound / Outbound Flow Magnitude
        tot_recv = features.get("total_received", 0.0)
        tot_sent = features.get("total_sent", 0.0)
        if tot_recv > 500000.0 or tot_sent > 500000.0:
            contributing_factors.append({
                "factor": "High Transaction Volume",
                "value": f"${max(tot_recv, tot_sent):,.2f}",
                "severity": "MEDIUM",
                "detail": f"Account processed abnormal volume (${tot_recv:,.0f} in / ${tot_sent:,.0f} out)."
            })

        # Summary Synthesis
        if risk_score >= self.risk_threshold:
            summary = (
                f"Account '{account_id}' is flagged as HIGH RISK ({risk_score:.2%}). "
                f"Key drivers include {len(contributing_factors)} abnormal behavioral and graph signals."
            )
        elif risk_score >= 0.50:
            summary = f"Account '{account_id}' exhibits MEDIUM RISK ({risk_score:.2%}) with notable exposure."
        else:
            summary = f"Account '{account_id}' exhibits LOW RISK ({risk_score:.2%}) with standard retail patterns."

        return {
            "account_id": account_id,
            "risk_score": float(risk_score),
            "risk_level": "HIGH" if risk_score >= self.risk_threshold else ("MEDIUM" if risk_score >= 0.50 else "LOW"),
            "contributing_factors": contributing_factors,
            "rule_reasons": reasons,
            "summary": summary
        }

    def extract_flow_subgraph(
        self,
        account_id: str,
        graph: nx.MultiDiGraph,
        k_hops: int = 2
    ) -> dict[str, Any]:
        """
        Extracts k-hop neighborhood and automatically assigns money-laundering roles:
        - Source / Payer: in_deg == 0, out_deg > 0
        - Mule / Relay: in_deg > 0, out_deg > 0
        - Sink / Cashout: in_deg > 0, out_deg == 0
        """
        if account_id not in graph:
            return {"nodes": [], "edges": [], "target": account_id}

        # BFS neighborhood extraction
        nodes = {account_id}
        frontier = {account_id}
        for _ in range(k_hops):
            nxt = set()
            for n in frontier:
                nxt.update(graph.predecessors(n))
                nxt.update(graph.successors(n))
            nodes.update(nxt)
            frontier = nxt

        sub_g = graph.subgraph(nodes)

        node_list = []
        for n in sub_g.nodes():
            in_deg = sub_g.in_degree(n)
            out_deg = sub_g.out_degree(n)

            if n == account_id:
                role = "Focal Account"
            elif in_deg == 0 and out_deg > 0:
                role = "Payer / Source"
            elif in_deg > 0 and out_deg > 0:
                role = "Mule / Relay"
            else:
                role = "Cashout / Sink"

            node_list.append({
                "account_id": n,
                "role": role,
                "in_degree": in_deg,
                "out_degree": out_deg
            })

        edge_list = []
        for u, v, data in sub_g.edges(data=True):
            edge_list.append({
                "source": u,
                "target": v,
                "amount": data.get("amount", 0.0),
                "timestamp": data.get("timestamp", 0),
                "type": data.get("type", "TXN")
            })

        return {
            "target": account_id,
            "node_count": len(node_list),
            "edge_count": len(edge_list),
            "nodes": node_list,
            "edges": edge_list
        }

    def detect_clusters(
        self,
        graph: nx.MultiDiGraph,
        risk_scores: dict[str, float] | None = None,
        min_size: int = 3
    ) -> list[dict[str, Any]]:
        """
        Discovers dense communities using Louvain and classifies suspicious mule rings.
        """
        if graph.number_of_nodes() < min_size:
            return []

        scores = risk_scores or {}
        # Convert to undirected graph for community detection
        undirected_g = graph.to_undirected()
        communities = louvain_communities(undirected_g, seed=42)

        clusters = []
        for i, comm in enumerate(communities, start=1):
            if len(comm) < min_size:
                continue

            sub_g = graph.subgraph(comm)
            total_vol = sum(d.get("amount", 0.0) for _, _, d in sub_g.edges(data=True))
            cluster_scores = [scores.get(n, 0.0) for n in comm]
            avg_risk = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
            high_risk_count = sum(1 for s in cluster_scores if s >= self.risk_threshold)

            # Heuristic labeling
            if high_risk_count >= 2:
                label = "Mule Ring / Smurfing Syndicate"
            elif high_risk_count == 1:
                label = "Suspicious Cluster"
            else:
                label = "Standard Customer Group"

            clusters.append({
                "cluster_id": f"CLUSTER_{i:03d}",
                "size": len(comm),
                "members": list(comm),
                "total_volume": total_vol,
                "avg_risk": round(avg_risk, 4),
                "high_risk_accounts": high_risk_count,
                "label": label
            })

        # Sort by avg_risk DESC
        clusters.sort(key=lambda x: (x["high_risk_accounts"], x["avg_risk"]), reverse=True)
        return clusters
