from collections import deque
from typing import Any
import networkx as nx


def propagate_risk(
    graph: nx.MultiDiGraph,
    flagged_accounts: dict[str, float] | list[str],
    decay_factor: float = 0.5,
    max_hops: int = 3,
    direction: str = "downstream"
) -> dict[str, float]:
    """
    Spreads risk outward from flagged seed accounts across graph topology using BFS with exponential decay.

    Args:
        graph: NetworkX MultiDiGraph
        flagged_accounts: Dict of {account_id: initial_risk} or list of flagged account IDs (default risk 1.0)
        decay_factor: Multiplier applied per hop (default 0.5)
        max_hops: Maximum BFS exploration depth (default 3)
        direction: "downstream" (successors/money out), "upstream" (predecessors/money in), or "both"

    Returns:
        Dict of {account_id: max_propagated_risk}
    """
    details = propagate_risk_detailed(
        graph=graph,
        flagged_accounts=flagged_accounts,
        decay_factor=decay_factor,
        max_hops=max_hops,
        direction=direction
    )
    return {acc: data["risk"] for acc, data in details.items()}


def propagate_risk_detailed(
    graph: nx.MultiDiGraph,
    flagged_accounts: dict[str, float] | list[str],
    decay_factor: float = 0.5,
    max_hops: int = 3,
    direction: str = "downstream"
) -> dict[str, dict[str, Any]]:
    """
    Same as propagate_risk, but returns both risk score and hop distance:
    {account_id: {"risk": float, "hop": int}}
    """
    if isinstance(flagged_accounts, list):
        seeds = {acc: 1.0 for acc in flagged_accounts}
    else:
        seeds = dict(flagged_accounts)

    best_results: dict[str, dict[str, Any]] = {}
    queue = deque()

    for acc, risk in seeds.items():
        if acc in graph:
            r_val = float(risk)
            best_results[acc] = {"risk": r_val, "hop": 0}
            queue.append((acc, r_val, 0))

    while queue:
        curr_acc, curr_risk, hop = queue.popleft()

        if hop >= max_hops:
            continue

        next_risk = curr_risk * decay_factor
        if next_risk <= 0.001:
            continue

        # Directional exploration
        if direction == "downstream":
            neighbors = list(graph.successors(curr_acc))
        elif direction == "upstream":
            neighbors = list(graph.predecessors(curr_acc))
        else:
            neighbors = list(set(graph.predecessors(curr_acc)) | set(graph.successors(curr_acc)))

        for nbr in neighbors:
            prev_risk = best_results.get(nbr, {}).get("risk", 0.0)
            if next_risk > prev_risk:
                best_results[nbr] = {"risk": next_risk, "hop": hop + 1}
                queue.append((nbr, next_risk, hop + 1))

    return best_results


def compute_neighbor_risk(
    graph: nx.MultiDiGraph,
    account_id: str,
    risk_scores: dict[str, float],
    direction: str = "both"
) -> float:
    """
    Computes the maximum risk among an account's immediate 1-hop counterparties.
    """
    if account_id not in graph:
        return 0.0

    if direction == "downstream":
        neighbors = set(graph.successors(account_id))
    elif direction == "upstream":
        neighbors = set(graph.predecessors(account_id))
    else:
        neighbors = set(graph.predecessors(account_id)) | set(graph.successors(account_id))

    if not neighbors:
        return 0.0

    return max(risk_scores.get(nbr, 0.0) for nbr in neighbors)
