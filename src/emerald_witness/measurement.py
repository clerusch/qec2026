from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx
from qiskit import ClassicalRegister, QuantumCircuit, transpile

from .auth import explicit_token_environment, resolve_token
from .config import ResonanceConfig
from .io_utils import (
    edge_sort_key,
    graph_from_subgraph_dict,
    ordered_qubit_names,
    qubit_sort_key,
    save_json,
    subgraph_to_dict,
    load_json,
)
from .models import MeasurementSetting
from .paths import measurement_data_path
from .subgraph import select_emerald_subgraph


DEFAULT_PLAN_PATH = measurement_data_path("emerald_stabilizer_measurement_plan.json")
DEFAULT_RESULTS_PATH = measurement_data_path("emerald_stabilizer_measurement_results.json")


def build_graph_state_circuit(
    graph: nx.Graph,
    *,
    add_barrier: bool = True,
    circuit_name: str = "emerald_graph_state",
) -> tuple[QuantumCircuit, dict[str, int]]:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot build a graph-state circuit from an empty graph.")

    ordered_qubits = sorted(graph.nodes, key=qubit_sort_key)
    ordered_edges = sorted(graph.edges, key=edge_sort_key)
    qubit_index_map = {qubit: index for index, qubit in enumerate(ordered_qubits)}

    circuit = QuantumCircuit(len(ordered_qubits), name=circuit_name)
    for qubit in ordered_qubits:
        circuit.h(qubit_index_map[qubit])

    if add_barrier and ordered_edges:
        circuit.barrier()

    for q0, q1 in ordered_edges:
        circuit.cz(qubit_index_map[q0], qubit_index_map[q1])

    circuit.metadata = {
        "emerald_qubit_order": ordered_qubits,
        "emerald_qubit_index_map": qubit_index_map,
        "subgraph": subgraph_to_dict(graph),
    }
    return circuit, qubit_index_map


def build_emerald_graph_state_circuit(
    *,
    target_size: int,
    config: ResonanceConfig,
    token: str,
    add_barrier: bool = True,
    circuit_name: str = "emerald_graph_state",
) -> tuple[QuantumCircuit, nx.Graph, dict[str, int]]:
    graph = select_emerald_subgraph(target_size=target_size, config=config, token=token)
    circuit, qubit_index_map = build_graph_state_circuit(
        graph,
        add_barrier=add_barrier,
        circuit_name=circuit_name,
    )
    return circuit, graph, qubit_index_map


def generate_stabilizer_generators(
    graph: nx.Graph,
    qubit_index_map: dict[str, int],
) -> list[str]:
    stabilizers: list[str] = []
    for node in graph.nodes:
        node_index = qubit_index_map[node]
        stabilizer = f"X{node_index}"
        for neighbor_index in sorted(qubit_index_map[neighbor] for neighbor in graph.neighbors(node)):
            stabilizer += f"Z{neighbor_index}"
        stabilizers.append(stabilizer)
    stabilizers.sort(key=lambda label: int(label.split("Z")[0][1:]))
    return stabilizers


def group_measurement_settings(
    graph: nx.Graph,
    qubit_index_map: dict[str, int],
) -> list[MeasurementSetting]:
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    num_colors = max(coloring.values()) + 1

    settings: list[MeasurementSetting] = []
    for color_index in range(num_colors):
        bases: dict[int, str] = {}
        measured_stabilizers: list[int] = []
        for node in graph.nodes:
            qubit_index = qubit_index_map[node]
            if coloring[node] == color_index:
                bases[qubit_index] = "X"
                measured_stabilizers.append(qubit_index)
            else:
                bases[qubit_index] = "Z"
        settings.append(
            MeasurementSetting(
                setting_id=color_index,
                bases=bases,
                measured_stabilizers=measured_stabilizers,
            )
        )
    return settings


def build_measurement_circuit(
    base_circuit: QuantumCircuit,
    setting: MeasurementSetting,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    circuit = base_circuit.copy()
    circuit.name = f"stabilizer_setting_{setting.setting_id:02d}"

    measurement_register = ClassicalRegister(base_circuit.num_qubits, "m")
    circuit.add_register(measurement_register)
    circuit.barrier()

    for qubit_index, basis in sorted(setting.bases.items()):
        if basis == "X":
            circuit.h(qubit_index)
        elif basis != "Z":
            raise ValueError(f"Unsupported measurement basis {basis!r}. Expected X or Z.")

    circuit.barrier()
    circuit.measure(range(base_circuit.num_qubits), measurement_register)

    measurement_info = setting.to_dict()
    circuit.metadata = dict(base_circuit.metadata or {})
    circuit.metadata.update(measurement_info)
    return circuit, measurement_info


def build_measurement_circuits(
    graph: nx.Graph,
    base_circuit: QuantumCircuit,
    qubit_index_map: dict[str, int],
) -> tuple[list[QuantumCircuit], list[dict[str, Any]], list[str]]:
    settings = group_measurement_settings(graph, qubit_index_map)
    stabilizers = generate_stabilizer_generators(graph, qubit_index_map)

    circuits: list[QuantumCircuit] = []
    measurement_settings: list[dict[str, Any]] = []
    for setting in settings:
        circuit, measurement_info = build_measurement_circuit(base_circuit, setting)
        circuits.append(circuit)
        measurement_settings.append(measurement_info)

    return circuits, measurement_settings, stabilizers


def _physical_index_from_name(qubit_name: str) -> int:
    match = re.fullmatch(r"QB(\d+)", qubit_name)
    if match is None:
        raise ValueError(f"Unsupported IQM qubit label {qubit_name!r}.")
    return int(match.group(1)) - 1


def _resolve_physical_qubit_indices(backend: Any, ordered_names: list[str]) -> list[int]:
    if hasattr(backend, "qubit_name_to_index"):
        return [backend.qubit_name_to_index(name) for name in ordered_names]
    return [_physical_index_from_name(name) for name in ordered_names]


def _reduced_coupling_map(
    backend: Any,
    physical_qubit_indices: list[int],
) -> list[list[int]] | None:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is None:
        return None
    selected = set(physical_qubit_indices)
    reduced_edges = [
        list(edge)
        for edge in coupling_map
        if set(edge).issubset(selected)
    ]
    return reduced_edges or None


def _transpile_restricted_circuits(
    circuits: list[QuantumCircuit],
    *,
    backend: Any,
    ordered_names: list[str],
    physical_qubit_indices: list[int],
) -> list[QuantumCircuit]:
    try:
        from iqm.qiskit_iqm import transpile_to_IQM
    except ImportError:
        transpile_to_IQM = None

    if transpile_to_IQM is not None:
        return [
            transpile_to_IQM(
                circuit,
                backend=backend,
                restrict_to_qubits=ordered_names,
                optimization_level=0,
            )
            for circuit in circuits
        ]

    transpiled = transpile(
        circuits,
        backend=backend,
        initial_layout=physical_qubit_indices,
        coupling_map=_reduced_coupling_map(backend, physical_qubit_indices),
        optimization_level=0,
    )
    if isinstance(transpiled, QuantumCircuit):
        return [transpiled]
    return list(transpiled)


def build_plan_payload(
    *,
    graph: nx.Graph,
    qubit_index_map: dict[str, int],
    stabilizers: list[str],
    measurement_settings: list[dict[str, Any]],
    shots_per_setting: int,
    config: ResonanceConfig,
    submitted_jobs: list[dict[str, Any]] | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    ordered_names = ordered_qubit_names(qubit_index_map)
    payload: dict[str, Any] = {
        "server_url": config.server_url,
        "quantum_computer": config.quantum_computer,
        "num_qubits": len(ordered_names),
        "ordered_qubit_names": ordered_names,
        "emerald_qubit_index_map": qubit_index_map,
        "shots_per_setting": shots_per_setting,
        "num_measurement_settings": len(measurement_settings),
        "total_shots": len(measurement_settings) * shots_per_setting,
        "stabilizers": stabilizers,
        "measurement_settings": measurement_settings,
        "subgraph": subgraph_to_dict(graph),
        "graph_attrs": dict(graph.graph),
    }
    if submitted_jobs is not None:
        payload["submitted_jobs"] = submitted_jobs
        payload["num_jobs"] = len(submitted_jobs)
    if backend is not None:
        payload["backend_name"] = getattr(backend, "name", None)
        payload["backend_max_circuits"] = getattr(backend, "max_circuits", None)
        payload["physical_qubit_indices"] = _resolve_physical_qubit_indices(
            backend,
            ordered_names,
        )
    return payload


def prepare_measurement_plan(
    *,
    target_size: int,
    shots_per_setting: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
) -> tuple[dict[str, Any], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    base_circuit, graph, qubit_index_map = build_emerald_graph_state_circuit(
        target_size=target_size,
        config=config,
        token=resolved_token,
        add_barrier=False,
    )
    circuits, measurement_settings, stabilizers = build_measurement_circuits(
        graph,
        base_circuit,
        qubit_index_map,
    )
    plan = build_plan_payload(
        graph=graph,
        qubit_index_map=qubit_index_map,
        stabilizers=stabilizers,
        measurement_settings=measurement_settings,
        shots_per_setting=shots_per_setting,
        config=config,
    )
    save_json(plan, plan_path)
    return plan, circuits


def submit_measurement_job(
    *,
    target_size: int,
    shots_per_setting: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
) -> tuple[Any, dict[str, Any], list[QuantumCircuit], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    plan, circuits = prepare_measurement_plan(
        target_size=target_size,
        shots_per_setting=shots_per_setting,
        config=config,
        token=resolved_token,
        plan_path=plan_path,
    )

    from iqm.qiskit_iqm import IQMProvider

    with explicit_token_environment(resolved_token):
        provider = IQMProvider(
            config.server_url,
            quantum_computer=config.quantum_computer,
            token=resolved_token,
        )
        backend = provider.get_backend()
        ordered_names = plan["ordered_qubit_names"]
        physical_qubit_indices = _resolve_physical_qubit_indices(backend, ordered_names)
        transpiled_circuits = _transpile_restricted_circuits(
            circuits,
            backend=backend,
            ordered_names=ordered_names,
            physical_qubit_indices=physical_qubit_indices,
        )
        job = backend.run(
            transpiled_circuits,
            shots=shots_per_setting,
            qubit_index_to_name=dict(enumerate(ordered_names)),
        )

    submitted_jobs = [
        {
            "job_id": job.job_id(),
            "num_circuits": len(transpiled_circuits),
            "shots_per_circuit": shots_per_setting,
            "total_shots": len(transpiled_circuits) * shots_per_setting,
            "setting_index_start": 0,
            "setting_index_end": len(transpiled_circuits) - 1,
        }
    ]
    saved_plan = load_json(plan_path)
    graph = graph_from_subgraph_dict(saved_plan["subgraph"])
    qubit_index_map = {
        str(qubit_name): int(index)
        for qubit_name, index in saved_plan["emerald_qubit_index_map"].items()
    }
    plan = build_plan_payload(
        graph=graph,
        qubit_index_map=qubit_index_map,
        stabilizers=saved_plan["stabilizers"],
        measurement_settings=saved_plan["measurement_settings"],
        shots_per_setting=shots_per_setting,
        config=config,
        submitted_jobs=submitted_jobs,
        backend=backend,
    )
    save_json(plan, plan_path)
    return job, plan, circuits, transpiled_circuits


def retrieve_measurement_results(
    *,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    job_id: str | None = None,
    timeout: float = 10800.0,
) -> dict[str, Any]:
    resolved_token = resolve_token(token)
    plan = load_json(plan_path)
    submitted_jobs = plan.get("submitted_jobs", [])
    resolved_job_id = job_id or (submitted_jobs[0]["job_id"] if submitted_jobs else None)
    if resolved_job_id is None:
        raise RuntimeError("No job_id given and no submitted job exists in the plan.")

    from iqm.qiskit_iqm import IQMProvider

    with explicit_token_environment(resolved_token):
        provider = IQMProvider(
            config.server_url,
            quantum_computer=config.quantum_computer,
            token=resolved_token,
        )
        backend = provider.get_backend()
        job = backend.retrieve_job(resolved_job_id)
        result = job.result(timeout=timeout)

    measurement_results: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(plan["measurement_settings"]):
        counts = {
            str(bitstring): int(count)
            for bitstring, count in result.get_counts(setting_index).items()
        }
        measurement_results.append(
            {
                "job_id": resolved_job_id,
                "setting_index": setting_index,
                "setting_id": setting["setting_id"],
                "bases": setting["bases"],
                "measured_stabilizers": setting["measured_stabilizers"],
                "shots": sum(counts.values()),
                "counts": counts,
            }
        )

    payload = {
        "server_url": config.server_url,
        "quantum_computer": config.quantum_computer,
        "job_id": resolved_job_id,
        "status": str(job.status()),
        "num_qubits": plan["num_qubits"],
        "ordered_qubit_names": plan["ordered_qubit_names"],
        "stabilizers": plan["stabilizers"],
        "measurement_settings": plan["measurement_settings"],
        "measurement_results": measurement_results,
    }
    save_json(payload, results_path)
    return payload
