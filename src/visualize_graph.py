from pathlib import Path
from typing import Any
import networkx as nx
import plotly.graph_objects as go


def visualize_subgraph(graph: nx.MultiDiGraph, title: str = "Argus — Subgraph Visualization", output_html: str | Path | None = None) -> go.Figure:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty Graph")
        return fig

    # Compute 2D spring layout positions
    pos = nx.spring_layout(graph, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    # Midpoint annotations for transaction amounts
    annotations = []

    for src, tgt, key, data in graph.edges(keys=True, data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        amount = data.get("amount", 0.0)
        txn_type = data.get("type", "TXN")
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=f"${amount:,.0f}<br>({txn_type})",
                showarrow=False,
                font=dict(size=9, color="#636EFA"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#d3d3d3",
                borderwidth=1,
                borderpad=2
            )
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        node_text.append(f"Account: {node}<br>In-Degree: {in_deg}<br>Out-Degree: {out_deg}")
        # Color based on flow (source, sink, mule/relay)
        if in_deg > 0 and out_deg > 0:
            node_color.append("#EF553B")  # Mule / Relay (Red)
        elif out_deg > 0:
            node_color.append("#00CC96")  # Origin / Payer (Green)
        else:
            node_color.append("#AB63FA")  # Cashout / Sink (Purple)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n if len(n) <= 10 else f"{n[:6]}..." for n in graph.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=22,
            line=dict(width=2, color="#333")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations
        )
    )

    if output_html:
        fig.write_html(str(output_html))

    return fig


def visualize_risk_propagation(
    graph: nx.MultiDiGraph,
    risk_details: dict[str, Any],
    title: str = "Argus — Risk Propagation Network",
    output_html: str | Path | None = None
) -> go.Figure:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty Graph")
        return fig

    pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    for src, tgt in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        detail = risk_details.get(node)
        if isinstance(detail, dict):
            risk = detail.get("risk", 0.0)
            hop = detail.get("hop", "N/A")
        elif isinstance(detail, (int, float)):
            risk = float(detail)
            hop = "N/A"
        else:
            risk = 0.0
            hop = "N/A"

        node_text.append(f"Account: {node}<br>Propagated Risk: {risk:.4f}<br>Hop Distance: {hop}")

        if risk >= 0.80:
            node_color.append("#EF553B")  # Red (Seed / High Risk)
        elif risk >= 0.50:
            node_color.append("#FFA15A")  # Orange (1-hop Exposure)
        elif risk >= 0.25:
            node_color.append("#FFD700")  # Yellow (2-hop Exposure)
        else:
            node_color.append("#00CC96")  # Green (Low Exposure)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n if len(n) <= 10 else f"{n[:6]}..." for n in graph.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=24,
            line=dict(width=2, color="#333")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    if output_html:
        fig.write_html(str(output_html))

    return fig


if __name__ == "__main__":
    # Self-test POC graph: A -> B -> C, A -> D
    g = nx.MultiDiGraph()
    g.add_edge("Orig_A", "Mule_B", amount=50000.0, type="TRANSFER", timestamp=1)
    g.add_edge("Mule_B", "Sink_C", amount=49000.0, type="CASH_OUT", timestamp=2)
    g.add_edge("Orig_A", "Sink_D", amount=1200.0, type="PAYMENT", timestamp=3)

    out_file = Path(__file__).resolve().parent.parent / "data" / "poc_graph.html"
    visualize_subgraph(g, title="Argus POC — Mule Flow (A -> B -> C)", output_html=out_file)
    print(f"[PASS] Graph visualization generated at: {out_file}")
