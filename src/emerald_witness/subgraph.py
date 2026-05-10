from __future__ import annotations

from statistics import mean
from typing import Any

import networkx as nx

from .characterization import get_characterization
from .config import ResonanceConfig


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty list.")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _normalize_scores(values: dict[str, float], default: float = 0.5) -> dict[str, float]:
    if not values:
        return {}

    min_value = min(values.values())
    max_value = max(values.values())
    if max_value == min_value:
        return {key: default for key in values}

    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in values.items()
    }


def _extract_qubit_metric(metric_block: dict[str, Any]) -> dict[str, float]:
    qubit_metric: dict[str, float] = {}
    for qubit, payload in metric_block.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        if isinstance(qubit, str) and qubit.startswith("QB") and isinstance(value, (int, float)):
            qubit_metric[qubit] = float(value)
    return qubit_metric


def _extract_edge_metric(metric_block: dict[str, Any]) -> dict[tuple[str, str], float]:
    edge_metric: dict[tuple[str, str], float] = {}
    for payload in metric_block.values():
        if not isinstance(payload, dict):
            continue

        qubits = payload.get("qubits")
        value = payload.get("value")
        if (
            isinstance(qubits, list)
            and len(qubits) == 2
            and all(isinstance(qubit, str) and qubit.startswith("QB") for qubit in qubits)
            and isinstance(value, (int, float))
        ):
            edge_metric[tuple(sorted(qubits))] = float(value)
    return edge_metric


def _build_filtered_graph(
    *,
    t1_times: dict[str, float],
    prx_fidelities: dict[str, float],
    edge_fidelities: dict[tuple[str, str], float],
    t1_min: float,
    max_edge_error: float,
) -> nx.Graph:
    allowed_qubits = {qubit for qubit, t1 in t1_times.items() if t1 >= t1_min}
    t1_scores = _normalize_scores(t1_times)
    prx_scores = _normalize_scores(prx_fidelities)

    graph = nx.Graph()
    for qubit in allowed_qubits:
        node_score = 0.75 * t1_scores.get(qubit, 0.5) + 0.25 * prx_scores.get(qubit, 0.5)
        graph.add_node(
            qubit,
            t1=t1_times[qubit],
            prx_fidelity=prx_fidelities.get(qubit),
            node_score=node_score,
        )

    for edge, fidelity in edge_fidelities.items():
        q0, q1 = edge
        if q0 not in allowed_qubits or q1 not in allowed_qubits:
            continue
        error_rate = 1.0 - fidelity
        if error_rate > max_edge_error:
            continue
        graph.add_edge(
            q0,
            q1,
            cz_fidelity=fidelity,
            edge_error_rate=error_rate,
            edge_score=fidelity,
        )

    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)
    return graph


def _grow_connected_subset(component_graph: nx.Graph, target_size: int) -> nx.Graph:
    if component_graph.number_of_nodes() <= target_size:
        return component_graph.copy()

    seed = max(
        component_graph.nodes,
        key=lambda node: (
            component_graph.nodes[node]["node_score"],
            component_graph.degree[node],
            node,
        ),
    )

    selected = {seed}
    while len(selected) < target_size:
        frontier: dict[str, float] = {}
        for node in selected:
            for neighbor in component_graph.neighbors(node):
                if neighbor in selected:
                    continue
                best_edge_score = max(
                    component_graph.edges[neighbor, selected_node]["edge_score"]
                    for selected_node in selected
                    if component_graph.has_edge(neighbor, selected_node)
                )
                candidate_score = (
                    0.65 * component_graph.nodes[neighbor]["node_score"]
                    + 0.30 * best_edge_score
                    + 0.05 * component_graph.degree[neighbor]
                )
                current = frontier.get(neighbor)
                if current is None or candidate_score > current:
                    frontier[neighbor] = candidate_score

        if not frontier:
            break

        selected.add(max(frontier, key=lambda node: (frontier[node], node)))

    return component_graph.subgraph(selected).copy()


def _candidate_key(graph: nx.Graph, target_size: int) -> tuple[float, float, float, float]:
    if graph.number_of_nodes() == 0:
        return (-1e9, -1e9, -1e9, -1e9)

    avg_node_score = mean(graph.nodes[node]["node_score"] for node in graph.nodes)
    avg_edge_fidelity = (
        mean(graph.edges[edge]["cz_fidelity"] for edge in graph.edges)
        if graph.number_of_edges() > 0
        else 0.0
    )
    size_penalty = -abs(graph.number_of_nodes() - target_size)
    return size_penalty, avg_node_score, avg_edge_fidelity, graph.number_of_edges()


def select_emerald_subgraph(
    *,
    target_size: int,
    config: ResonanceConfig,
    token: str,
) -> nx.Graph:
    characterization = get_characterization(config, token)
    t1_times = _extract_qubit_metric(characterization["t1_times"])
    prx = _extract_qubit_metric(characterization["gate_fidelities"]["single_qubit_prx"])
    edge_fidelities = _extract_edge_metric(characterization["gate_fidelities"]["two_qubit_cz"])
    if not edge_fidelities:
        edge_fidelities = _extract_edge_metric(
            characterization["gate_fidelities"]["two_qubit_clifford"]
        )

    if not t1_times:
        raise ValueError("No T1 times were found in the characterization payload.")
    if not edge_fidelities:
        raise ValueError("No two-qubit fidelity data were found in the characterization payload.")

    t1_quantiles = [0.35, 0.25, 0.15, 0.05, 0.0]
    edge_error_quantiles = [0.70, 0.80, 0.90, 0.95, 1.0]
    edge_errors = [1.0 - fidelity for fidelity in edge_fidelities.values()]

    best_graph: nx.Graph | None = None
    best_key: tuple[float, float, float, float] | None = None
    best_thresholds: dict[str, float] | None = None

    for t1_q in t1_quantiles:
        t1_min = _quantile(list(t1_times.values()), t1_q)
        for edge_q in edge_error_quantiles:
            max_edge_error = _quantile(edge_errors, edge_q)
            filtered_graph = _build_filtered_graph(
                t1_times=t1_times,
                prx_fidelities=prx,
                edge_fidelities=edge_fidelities,
                t1_min=t1_min,
                max_edge_error=max_edge_error,
            )
            if filtered_graph.number_of_nodes() == 0:
                continue

            for component_nodes in nx.connected_components(filtered_graph):
                component_graph = filtered_graph.subgraph(component_nodes).copy()
                candidate_graph = _grow_connected_subset(component_graph, target_size)
                candidate_key = _candidate_key(candidate_graph, target_size)
                if best_key is None or candidate_key > best_key:
                    best_graph = candidate_graph
                    best_key = candidate_key
                    best_thresholds = {
                        "t1_min": t1_min,
                        "max_edge_error_rate": max_edge_error,
                        "t1_quantile": t1_q,
                        "edge_error_quantile": edge_q,
                    }

    if best_graph is None or best_thresholds is None:
        raise ValueError("Could not construct a connected Emerald subgraph from the available metrics.")

    best_graph.graph.update(
        {
            "quantum_computer": characterization["quantum_computer"],
            "calibration_set_id": characterization["calibration_set_id"],
            "api_layout": characterization["api_layout"],
            "root_url": characterization["root_url"],
            "target_size": target_size,
            **best_thresholds,
        }
    )
    return best_graph
