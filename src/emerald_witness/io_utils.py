from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx


def qubit_sort_key(qubit_label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", qubit_label)
    if match:
        return int(match.group(1)), qubit_label
    return 10**9, qubit_label


def edge_sort_key(edge: tuple[str, str]) -> tuple[tuple[int, str], tuple[int, str]]:
    q0, q1 = sorted(edge, key=qubit_sort_key)
    return qubit_sort_key(q0), qubit_sort_key(q1)


def ordered_qubit_names(qubit_index_map: dict[str, int]) -> list[str]:
    return [
        qubit_name
        for qubit_name, _ in sorted(qubit_index_map.items(), key=lambda item: item[1])
    ]


def subgraph_to_dict(graph: nx.Graph) -> dict[str, Any]:
    return {
        "graph_attrs": dict(graph.graph),
        "qubits": [
            {"qubit": node, **dict(graph.nodes[node])}
            for node in sorted(graph.nodes, key=qubit_sort_key)
        ],
        "edges": [
            {"qubits": [u, v], **dict(graph.edges[u, v])}
            for u, v in sorted(graph.edges, key=edge_sort_key)
        ],
    }


def graph_from_subgraph_dict(subgraph: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.graph.update(subgraph.get("graph_attrs", {}))

    for qubit_info in subgraph.get("qubits", []):
        node = str(qubit_info["qubit"])
        attrs = {key: value for key, value in qubit_info.items() if key != "qubit"}
        graph.add_node(node, **attrs)

    for edge_info in subgraph.get("edges", []):
        q0, q1 = [str(name) for name in edge_info["qubits"]]
        attrs = {key: value for key, value in edge_info.items() if key != "qubits"}
        graph.add_edge(q0, q1, **attrs)

    return graph


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())
