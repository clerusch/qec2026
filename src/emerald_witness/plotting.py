from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize

from .auth import resolve_token
from .config import ResonanceConfig
from .io_utils import graph_from_subgraph_dict, load_json
from .paths import plot_path
from .subgraph import select_emerald_subgraph


def _format_t1(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1e-6:
        return f"{value * 1e9:.1f} ns"
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


def save_subgraph_png(
    graph: nx.Graph,
    output_path: str | Path = plot_path("emerald_subgraph.png"),
) -> Path:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render an empty graph.")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    layout = nx.kamada_kawai_layout(graph)
    node_order = sorted(graph.nodes)
    edge_order = sorted(graph.edges)

    t1_values = [float(graph.nodes[node]["t1"]) for node in node_order]
    node_scores = [float(graph.nodes[node].get("node_score", 0.5)) for node in node_order]
    fidelities = [float(graph.edges[edge].get("cz_fidelity", 0.0)) for edge in edge_order]

    t1_min = min(t1_values)
    t1_max = max(t1_values)
    if t1_min == t1_max:
        t1_min -= 1.0
        t1_max += 1.0

    fidelity_min = min(fidelities) if fidelities else 0.0
    fidelity_max = max(fidelities) if fidelities else 1.0
    if fidelity_min == fidelity_max:
        fidelity_min -= 1.0
        fidelity_max += 1.0

    node_norm = Normalize(vmin=t1_min, vmax=t1_max)
    edge_norm = Normalize(vmin=fidelity_min, vmax=fidelity_max)

    node_colors = [cm.YlGnBu(node_norm(value)) for value in t1_values]
    node_sizes = [1100 + 1600 * score for score in node_scores]
    edge_colors = [cm.YlOrRd_r(edge_norm(value)) for value in fidelities]
    edge_widths = [2.5 + 5.0 * max(0.0, value) for value in fidelities]

    figure, axis = plt.subplots(figsize=(11, 8), dpi=200)
    axis.set_facecolor("#fbfbf8")

    nx.draw_networkx_edges(
        graph,
        pos=layout,
        ax=axis,
        edgelist=edge_order,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        ax=axis,
        nodelist=node_order,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.5,
        edgecolors="#17324d",
    )
    nx.draw_networkx_labels(
        graph,
        pos=layout,
        ax=axis,
        labels={
            node: f"{node}\nT1 {_format_t1(graph.nodes[node].get('t1'))}"
            for node in node_order
        },
        font_size=8,
        font_weight="bold",
        font_color="#13212e",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos=layout,
        ax=axis,
        edge_labels={
            edge: f"{graph.edges[edge].get('cz_fidelity', 0.0):.4f}"
            for edge in edge_order
        },
        font_size=7,
        rotate=False,
        font_color="#6b2d14",
        bbox={"facecolor": "#fbfbf8", "edgecolor": "none", "alpha": 0.8},
    )

    title = f"IQM Emerald Selected {graph.number_of_nodes()}-Qubit Subgraph"
    subtitle = (
        f"T1 >= {graph.graph.get('t1_min', 0.0):.3e}, "
        f"edge error <= {graph.graph.get('max_edge_error_rate', 0.0):.3e}"
    )
    axis.set_title(f"{title}\n{subtitle}", fontsize=15, weight="bold", color="#13212e", pad=18)
    axis.axis("off")

    node_sm = cm.ScalarMappable(norm=node_norm, cmap=cm.YlGnBu)
    node_sm.set_array([])
    edge_sm = cm.ScalarMappable(norm=edge_norm, cmap=cm.YlOrRd_r)
    edge_sm.set_array([])

    figure.colorbar(node_sm, ax=axis, fraction=0.046, pad=0.02).set_label("T1")
    figure.colorbar(edge_sm, ax=axis, fraction=0.046, pad=0.10).set_label("CZ fidelity")

    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def draw_subgraph_from_plan(
    plan_path: str | Path,
    output_path: str | Path = plot_path("emerald_subgraph.png"),
) -> Path:
    payload = load_json(plan_path)
    if "subgraph" not in payload:
        raise KeyError(f"No 'subgraph' entry found in plan file: {plan_path}")
    graph = graph_from_subgraph_dict(payload["subgraph"])
    return save_subgraph_png(graph, output_path=output_path)


def draw_emerald_subgraph(
    *,
    target_size: int,
    config: ResonanceConfig,
    output_path: str | Path = plot_path("emerald_subgraph.png"),
    token: str | None = None,
) -> Path:
    resolved_token = resolve_token(token)
    graph = select_emerald_subgraph(
        target_size=target_size,
        config=config,
        token=resolved_token,
    )
    return save_subgraph_png(graph, output_path=output_path)
