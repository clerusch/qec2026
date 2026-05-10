from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .auth import explicit_token_environment, resolve_token
from .config import DEFAULT_QUANTUM_COMPUTER, DEFAULT_SERVER_URL, ResonanceConfig
from .io_utils import (
    graph_from_subgraph_dict,
    ordered_qubit_names,
    save_json,
    subgraph_to_dict,
)
from .measurement import (
    _resolve_physical_qubit_indices,
    _transpile_restricted_circuits,
    build_emerald_graph_state_circuit,
    build_graph_state_circuit,
)
from .paths import measurement_data_path


DEFAULT_PLAN_PATH = measurement_data_path("emerald_alpha_projector_witness_plan.json")
DEFAULT_RESULTS_PATH = measurement_data_path("emerald_alpha_projector_witness_results.json")
DEFAULT_RESULTS_CSV_PATH = measurement_data_path("emerald_alpha_projector_witness_results.csv")
DEFAULT_SUMMARY_PATH = measurement_data_path("emerald_alpha_projector_witness_summary.json")
DEFAULT_TERMS_CSV_PATH = measurement_data_path("emerald_alpha_projector_witness_terms.csv")
DEFAULT_MATRIX_PATH = measurement_data_path("emerald_alpha_projector_witness_matrix.npy")


def _chunk_sequence(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _circuit_order_pauli_chars(pauli_label: str) -> list[str]:
    return list(reversed(pauli_label))


def _apply_pauli_measurement_rotation(
    circuit: QuantumCircuit,
    qubit_index: int,
    pauli_char: str,
) -> None:
    if pauli_char in {"I", "Z"}:
        return
    if pauli_char == "X":
        circuit.h(qubit_index)
        return
    if pauli_char == "Y":
        circuit.sdg(qubit_index)
        circuit.h(qubit_index)
        return
    raise ValueError(f"Unsupported Pauli character {pauli_char!r}.")


def _clean_bitstring(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def _bit_for_qubit(bitstring: str, qubit_index: int) -> int:
    cleaned = _clean_bitstring(bitstring)
    if qubit_index >= len(cleaned):
        raise ValueError(f"Bitstring {bitstring!r} is too short for qubit index {qubit_index}.")
    return int(cleaned[-1 - qubit_index])


def _parity_eigenvalue(bitstring: str, active_qubit_indices: list[int]) -> int:
    parity = sum(_bit_for_qubit(bitstring, qubit_index) for qubit_index in active_qubit_indices) % 2
    return 1 if parity == 0 else -1


def expectation_value_from_counts(
    counts: dict[str, int],
    active_qubit_indices: list[int],
) -> float:
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("Cannot compute expectation value from zero shots.")
    weighted_sum = sum(
        _parity_eigenvalue(bitstring, active_qubit_indices) * count
        for bitstring, count in counts.items()
    )
    return weighted_sum / total_shots


def graph_state_statevector(
    graph: Any,
) -> tuple[np.ndarray, dict[str, int], QuantumCircuit]:
    circuit, qubit_index_map = build_graph_state_circuit(
        graph,
        add_barrier=False,
        circuit_name="emerald_graph_state_alpha_projector",
    )
    statevector = Statevector.from_instruction(circuit).data
    return np.asarray(statevector, dtype=complex), qubit_index_map, circuit


def graph_state_projector(
    graph: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], QuantumCircuit]:
    statevector, qubit_index_map, circuit = graph_state_statevector(graph)
    projector = np.outer(statevector, np.conjugate(statevector))
    return projector, statevector, qubit_index_map, circuit


def max_biseparable_overlap(
    statevector: np.ndarray,
    *,
    num_qubits: int,
) -> tuple[float, tuple[int, ...]]:
    tensor = statevector.reshape([2] * num_qubits)

    best_overlap = 0.0
    best_partition: tuple[int, ...] = ()
    for part_size in range(1, num_qubits // 2 + 1):
        for partition in combinations(range(num_qubits), part_size):
            if part_size == num_qubits - part_size and partition[0] != 0:
                continue
            complement = tuple(qubit for qubit in range(num_qubits) if qubit not in partition)
            permutation = list(partition) + list(complement)
            reshaped = np.transpose(tensor, permutation).reshape(
                2**part_size,
                2 ** (num_qubits - part_size),
            )
            singular_values = np.linalg.svd(reshaped, compute_uv=False)
            overlap = float(singular_values[0] ** 2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_partition = partition

    return best_overlap, best_partition


def alpha_projector_witness_matrix(
    graph: Any,
) -> tuple[float, tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, dict[str, int], QuantumCircuit]:
    projector, statevector, qubit_index_map, base_circuit = graph_state_projector(graph)
    num_qubits = base_circuit.num_qubits
    alpha, alpha_partition = max_biseparable_overlap(statevector, num_qubits=num_qubits)
    dimension = projector.shape[0]
    witness = alpha * np.eye(dimension, dtype=complex) - projector
    return alpha, alpha_partition, witness, projector, statevector, qubit_index_map, base_circuit


def alpha_projector_sparse_pauli_decomposition(
    witness: np.ndarray,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> SparsePauliOp:
    decomposition = SparsePauliOp.from_operator(witness, atol=atol, rtol=rtol)
    return decomposition.simplify(atol=atol, rtol=rtol)


def build_pauli_measurement_circuit(
    base_circuit: QuantumCircuit,
    pauli_label: str,
    coefficient: complex,
    term_index: int,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    num_qubits = base_circuit.num_qubits
    if len(pauli_label) != num_qubits:
        raise ValueError(
            f"Pauli label length {len(pauli_label)} does not match the circuit size {num_qubits}."
        )

    circuit = base_circuit.copy()
    circuit.name = f"alpha_witness_term_{term_index:04d}_{pauli_label}"

    measurement_register = ClassicalRegister(num_qubits, "m")
    circuit.add_register(measurement_register)

    basis_by_qubit_index = _circuit_order_pauli_chars(pauli_label)
    active_qubit_indices = [
        qubit_index
        for qubit_index, pauli_char in enumerate(basis_by_qubit_index)
        if pauli_char != "I"
    ]

    circuit.barrier()
    for qubit_index, pauli_char in enumerate(basis_by_qubit_index):
        _apply_pauli_measurement_rotation(circuit, qubit_index, pauli_char)
    circuit.barrier()
    circuit.measure(range(num_qubits), measurement_register)

    measurement_info = {
        "term_index": int(term_index),
        "pauli_label_qiskit_order": str(pauli_label),
        "pauli_label_circuit_order": "".join(basis_by_qubit_index),
        "coefficient_real": float(np.real(coefficient)),
        "coefficient_imag": float(np.imag(coefficient)),
        "active_qubit_indices": active_qubit_indices,
    }
    circuit.metadata = dict(base_circuit.metadata or {})
    circuit.metadata.update(measurement_info)
    return circuit, measurement_info


def build_witness_measurement_circuits(
    base_circuit: QuantumCircuit,
    decomposition: SparsePauliOp,
) -> tuple[list[QuantumCircuit], list[dict[str, Any]]]:
    circuits: list[QuantumCircuit] = []
    measurement_terms: list[dict[str, Any]] = []
    for term_index, (pauli_label, coefficient) in enumerate(decomposition.to_list()):
        circuit, measurement_info = build_pauli_measurement_circuit(
            base_circuit,
            pauli_label=pauli_label,
            coefficient=coefficient,
            term_index=term_index,
        )
        circuits.append(circuit)
        measurement_terms.append(measurement_info)
    return circuits, measurement_terms


def prepare_alpha_projector_plan(
    *,
    target_size: int,
    shots_per_term: int,
    circuits_per_job: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> tuple[dict[str, Any], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    _, graph, _ = build_emerald_graph_state_circuit(
        target_size=target_size,
        config=config,
        token=resolved_token,
        add_barrier=False,
    )
    alpha, alpha_partition, witness, projector, statevector, qubit_index_map, base_circuit = (
        alpha_projector_witness_matrix(graph)
    )
    decomposition = alpha_projector_sparse_pauli_decomposition(witness)
    circuits, measurement_terms = build_witness_measurement_circuits(base_circuit, decomposition)

    matrix_output = Path(matrix_path).expanduser().resolve()
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_output, witness)

    payload = {
        "measurement_mode": "alpha_projector_witness",
        "server_url": config.server_url,
        "quantum_computer": config.quantum_computer,
        "num_qubits": base_circuit.num_qubits,
        "ordered_qubit_names": ordered_qubit_names(qubit_index_map),
        "emerald_qubit_index_map": qubit_index_map,
        "shots_per_term": int(shots_per_term),
        "circuits_per_job": int(circuits_per_job),
        "num_pauli_terms": len(measurement_terms),
        "total_shots": len(measurement_terms) * int(shots_per_term),
        "alpha": float(alpha),
        "alpha_partition": list(alpha_partition),
        "ideal_projector_expectation": 1.0,
        "ideal_witness_expectation": float(alpha - 1.0),
        "witness_matrix_path": str(matrix_output),
        "measurement_terms": measurement_terms,
        "subgraph": subgraph_to_dict(graph),
        "graph_attrs": dict(graph.graph),
        "decomposition_summary": {
            "num_pauli_terms": len(measurement_terms),
            "num_identity_terms": sum(
                1 for term in measurement_terms if set(term["pauli_label_circuit_order"]) == {"I"}
            ),
            "max_abs_coefficient": max(abs(term["coefficient_real"]) for term in measurement_terms),
            "min_abs_nonzero_coefficient": min(
                abs(term["coefficient_real"])
                for term in measurement_terms
                if abs(term["coefficient_real"]) > 0.0
            ),
        },
        "statevector_norm": float(np.linalg.norm(statevector)),
        "projector_trace": float(np.trace(projector).real),
    }
    save_json(payload, plan_path)
    return payload, circuits


def submit_alpha_projector_jobs(
    *,
    target_size: int,
    shots_per_term: int,
    circuits_per_job: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> tuple[dict[str, Any], list[QuantumCircuit], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    plan, circuits = prepare_alpha_projector_plan(
        target_size=target_size,
        shots_per_term=shots_per_term,
        circuits_per_job=circuits_per_job,
        config=config,
        token=resolved_token,
        plan_path=plan_path,
        matrix_path=matrix_path,
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
        backend_max_circuits = getattr(backend, "max_circuits", None)
        effective_circuits_per_job = circuits_per_job
        if backend_max_circuits is not None:
            effective_circuits_per_job = min(effective_circuits_per_job, backend_max_circuits)
        if effective_circuits_per_job <= 0:
            raise RuntimeError("No usable circuit batch size was available for the backend.")

        qubit_index_to_name = dict(enumerate(ordered_names))
        circuit_chunks = _chunk_sequence(circuits, effective_circuits_per_job)
        measurement_term_chunks = _chunk_sequence(plan["measurement_terms"], effective_circuits_per_job)
        submitted_jobs: list[dict[str, Any]] = []
        transpiled_circuits: list[QuantumCircuit] = []
        for batch_index, (raw_circuit_chunk, term_chunk) in enumerate(
            zip(circuit_chunks, measurement_term_chunks),
            start=1,
        ):
            start_term_index = int(term_chunk[0]["term_index"])
            end_term_index = int(term_chunk[-1]["term_index"])
            transpiled_chunk = _transpile_restricted_circuits(
                raw_circuit_chunk,
                backend=backend,
                ordered_names=ordered_names,
                physical_qubit_indices=physical_qubit_indices,
            )
            transpiled_circuits.extend(transpiled_chunk)
            job = backend.run(
                transpiled_chunk,
                shots=shots_per_term,
                qubit_index_to_name=qubit_index_to_name,
            )
            submitted_jobs.append(
                {
                    "batch_index": batch_index,
                    "job_id": job.job_id(),
                    "num_circuits": len(transpiled_chunk),
                    "shots_per_circuit": int(shots_per_term),
                    "total_shots": len(transpiled_chunk) * int(shots_per_term),
                    "term_index_start": start_term_index,
                    "term_index_end": end_term_index,
                }
            )
            plan["submitted_jobs"] = submitted_jobs
            plan["num_jobs"] = len(submitted_jobs)
            plan["circuits_per_job"] = effective_circuits_per_job
            plan["backend_name"] = getattr(backend, "name", None)
            plan["backend_max_circuits"] = getattr(backend, "max_circuits", None)
            plan["physical_qubit_indices"] = _resolve_physical_qubit_indices(
                backend,
                plan["ordered_qubit_names"],
            )
            save_json(plan, plan_path)

    plan["submitted_jobs"] = submitted_jobs
    plan["num_jobs"] = len(submitted_jobs)
    plan["circuits_per_job"] = effective_circuits_per_job
    plan["backend_name"] = getattr(backend, "name", None)
    plan["backend_max_circuits"] = getattr(backend, "max_circuits", None)
    plan["physical_qubit_indices"] = _resolve_physical_qubit_indices(
        backend,
        plan["ordered_qubit_names"],
    )
    save_json(plan, plan_path)
    return plan, circuits, transpiled_circuits


def retrieve_alpha_projector_results(
    *,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    results_csv_path: str | Path = DEFAULT_RESULTS_CSV_PATH,
    timeout: float = 10800.0,
) -> dict[str, Any]:
    resolved_token = resolve_token(token)
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    submitted_jobs = plan.get("submitted_jobs", [])
    if not submitted_jobs:
        raise RuntimeError("No submitted jobs were found in the plan.")

    from iqm.qiskit_iqm import IQMProvider

    measurement_results: list[dict[str, Any]] = []
    job_statuses: list[dict[str, Any]] = []

    with explicit_token_environment(resolved_token):
        provider = IQMProvider(
            config.server_url,
            quantum_computer=config.quantum_computer,
            token=resolved_token,
        )
        backend = provider.get_backend()
        for job_info in submitted_jobs:
            job = backend.retrieve_job(job_info["job_id"])
            result = job.result(timeout=timeout)
            job_statuses.append(
                {
                    "batch_index": int(job_info["batch_index"]),
                    "job_id": str(job_info["job_id"]),
                    "status": str(job.status()),
                }
            )
            for circuit_index in range(int(job_info["num_circuits"])):
                term_index = int(job_info["term_index_start"]) + circuit_index
                term = plan["measurement_terms"][term_index]
                counts = {
                    str(bitstring): int(count)
                    for bitstring, count in result.get_counts(circuit_index).items()
                }
                expectation_value = expectation_value_from_counts(
                    counts,
                    [int(index) for index in term["active_qubit_indices"]],
                )
                measurement_results.append(
                    {
                        "job_id": str(job_info["job_id"]),
                        "batch_index": int(job_info["batch_index"]),
                        "circuit_index_in_job": circuit_index,
                        "term_index": term_index,
                        "pauli_label_qiskit_order": term["pauli_label_qiskit_order"],
                        "pauli_label_circuit_order": term["pauli_label_circuit_order"],
                        "active_qubit_indices": term["active_qubit_indices"],
                        "coefficient_real": term["coefficient_real"],
                        "coefficient_imag": term["coefficient_imag"],
                        "shots": sum(counts.values()),
                        "counts": counts,
                        "expectation_value": expectation_value,
                    }
                )

    measurement_results.sort(key=lambda item: int(item["term_index"]))
    payload = {
        "measurement_mode": plan["measurement_mode"],
        "server_url": config.server_url,
        "quantum_computer": config.quantum_computer,
        "num_qubits": plan["num_qubits"],
        "ordered_qubit_names": plan["ordered_qubit_names"],
        "alpha": plan["alpha"],
        "alpha_partition": plan["alpha_partition"],
        "job_statuses": job_statuses,
        "measurement_results": measurement_results,
    }
    save_json(payload, results_path)
    save_measurement_results_csv(measurement_results, results_csv_path)
    return payload


def _normalize_distribution(weights: list[float]) -> list[float]:
    total_weight = sum(weights)
    if total_weight == 0.0:
        raise ValueError("Cannot normalize a zero-weight distribution.")
    return [weight / total_weight for weight in weights]


def _total_variation_distance(dist_a: list[float], dist_b: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(dist_a, dist_b))


def _cosine_similarity(values_a: list[float], values_b: list[float]) -> float:
    norm_a = math.sqrt(sum(value * value for value in values_a))
    norm_b = math.sqrt(sum(value * value for value in values_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity with a zero vector.")
    dot_product = sum(a * b for a, b in zip(values_a, values_b))
    return dot_product / (norm_a * norm_b)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def evaluate_alpha_projector_results(
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    terms_csv_path: str | Path = DEFAULT_TERMS_CSV_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    results = json.loads(Path(results_path).expanduser().resolve().read_text())
    graph = graph_from_subgraph_dict(plan["subgraph"])
    statevector_data, _, _ = graph_state_statevector(graph)
    statevector = Statevector(statevector_data)

    rows = sorted(results["measurement_results"], key=lambda item: int(item["term_index"]))
    if len(rows) != len(plan["measurement_terms"]):
        raise RuntimeError(
            f"Results contain {len(rows)} rows but the plan contains {len(plan['measurement_terms'])} Pauli terms."
        )

    coefficient_abs_weights: list[float] = []
    measured_abs_weights: list[float] = []
    expectation_errors: list[float] = []
    contribution_errors: list[float] = []
    term_comparisons: list[dict[str, Any]] = []

    alpha = float(plan["alpha"])
    ideal_witness_expectation = float(plan["ideal_witness_expectation"])
    measured_witness_expectation = 0.0

    for row, plan_term in zip(rows, plan["measurement_terms"]):
        coefficient_real = float(plan_term["coefficient_real"])
        coefficient_imag = float(plan_term["coefficient_imag"])
        if abs(coefficient_imag) > 1e-8:
            raise RuntimeError(
                f"Encountered a non-real witness coefficient for term {row['term_index']}: "
                f"{coefficient_real} + {coefficient_imag}j"
            )

        pauli_label = str(plan_term["pauli_label_qiskit_order"])
        ideal_expectation = statevector.expectation_value(
            SparsePauliOp.from_list([(pauli_label, 1.0)])
        )
        ideal_expectation_real = float(np.real_if_close(ideal_expectation))
        measured_expectation = float(row["expectation_value"])

        ideal_contribution = coefficient_real * ideal_expectation_real
        measured_contribution = coefficient_real * measured_expectation
        expectation_error = measured_expectation - ideal_expectation_real
        contribution_error = measured_contribution - ideal_contribution

        coefficient_abs_weights.append(abs(coefficient_real))
        measured_abs_weights.append(abs(measured_contribution))
        expectation_errors.append(expectation_error)
        contribution_errors.append(contribution_error)
        measured_witness_expectation += measured_contribution

        term_comparisons.append(
            {
                "term_index": int(row["term_index"]),
                "job_id": row["job_id"],
                "batch_index": row["batch_index"],
                "circuit_index_in_job": row["circuit_index_in_job"],
                "pauli_label_qiskit_order": pauli_label,
                "pauli_label_circuit_order": plan_term["pauli_label_circuit_order"],
                "coefficient_real": coefficient_real,
                "shots": int(row["shots"]),
                "ideal_expectation_value": ideal_expectation_real,
                "measured_expectation_value": measured_expectation,
                "expectation_error": expectation_error,
                "ideal_contribution": ideal_contribution,
                "measured_contribution": measured_contribution,
                "contribution_error": contribution_error,
                "abs_expectation_error": abs(expectation_error),
                "abs_contribution_error": abs(contribution_error),
                "counts": row["counts"],
            }
        )

    coefficient_distribution = _normalize_distribution(coefficient_abs_weights)
    measured_contribution_distribution = _normalize_distribution(measured_abs_weights)
    fidelity_estimate = alpha - measured_witness_expectation
    fidelity_sigma = math.sqrt(
        sum(
            (float(term["coefficient_real"]) ** 2)
            * (1.0 - float(row["expectation_value"]) ** 2)
            / int(row["shots"])
            for term, row in zip(plan["measurement_terms"], rows)
        )
    )

    worst_expectation_terms = sorted(
        term_comparisons,
        key=lambda term: term["abs_expectation_error"],
        reverse=True,
    )[:20]
    worst_contribution_terms = sorted(
        term_comparisons,
        key=lambda term: term["abs_contribution_error"],
        reverse=True,
    )[:20]

    summary = {
        "measurement_mode": plan["measurement_mode"],
        "plan_path": str(Path(plan_path).expanduser().resolve()),
        "results_path": str(Path(results_path).expanduser().resolve()),
        "num_qubits": int(plan["num_qubits"]),
        "num_terms": len(rows),
        "ordered_qubit_names": plan["ordered_qubit_names"],
        "graph_attrs": plan["graph_attrs"],
        "alpha": alpha,
        "alpha_partition": plan["alpha_partition"],
        "ideal_projector_expectation": 1.0,
        "estimated_graph_state_projector_expectation": fidelity_estimate,
        "projector_expectation_sigma": fidelity_sigma,
        "ideal_witness_expectation": ideal_witness_expectation,
        "measured_witness_expectation": measured_witness_expectation,
        "witness_expectation_error": measured_witness_expectation - ideal_witness_expectation,
        "is_entangled_by_alpha_witness": measured_witness_expectation < 0.0,
        "mean_expectation_value": _mean(
            [term["measured_expectation_value"] for term in term_comparisons]
        ),
        "mean_abs_expectation_error": _mean([abs(error) for error in expectation_errors]),
        "rmse_expectation_error": _rmse(expectation_errors),
        "mean_abs_contribution_error": _mean([abs(error) for error in contribution_errors]),
        "rmse_contribution_error": _rmse(contribution_errors),
        "coefficient_distribution_total_variation": _total_variation_distance(
            coefficient_distribution,
            measured_contribution_distribution,
        ),
        "coefficient_distribution_cosine_similarity": _cosine_similarity(
            coefficient_distribution,
            measured_contribution_distribution,
        ),
        "worst_expectation_terms_preview": worst_expectation_terms,
        "worst_contribution_terms_preview": worst_contribution_terms,
    }
    save_json(summary, summary_path)
    save_term_comparisons_csv(term_comparisons, terms_csv_path)
    return summary, term_comparisons


def save_measurement_results_csv(
    measurement_results: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_RESULTS_CSV_PATH,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "batch_index",
        "circuit_index_in_job",
        "term_index",
        "pauli_label_qiskit_order",
        "pauli_label_circuit_order",
        "active_qubit_indices_json",
        "coefficient_real",
        "coefficient_imag",
        "shots",
        "expectation_value",
        "counts_json",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in measurement_results:
            writer.writerow(
                {
                    "job_id": row["job_id"],
                    "batch_index": row["batch_index"],
                    "circuit_index_in_job": row["circuit_index_in_job"],
                    "term_index": row["term_index"],
                    "pauli_label_qiskit_order": row["pauli_label_qiskit_order"],
                    "pauli_label_circuit_order": row["pauli_label_circuit_order"],
                    "active_qubit_indices_json": json.dumps(row["active_qubit_indices"]),
                    "coefficient_real": row["coefficient_real"],
                    "coefficient_imag": row["coefficient_imag"],
                    "shots": row["shots"],
                    "expectation_value": row["expectation_value"],
                    "counts_json": json.dumps(row["counts"], sort_keys=True),
                }
            )
    return output


def save_term_comparisons_csv(
    term_comparisons: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_TERMS_CSV_PATH,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "term_index",
        "job_id",
        "batch_index",
        "circuit_index_in_job",
        "pauli_label_qiskit_order",
        "pauli_label_circuit_order",
        "coefficient_real",
        "shots",
        "ideal_expectation_value",
        "measured_expectation_value",
        "expectation_error",
        "ideal_contribution",
        "measured_contribution",
        "contribution_error",
        "abs_expectation_error",
        "abs_contribution_error",
        "counts_json",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in term_comparisons:
            writer.writerow(
                {
                    "term_index": row["term_index"],
                    "job_id": row["job_id"],
                    "batch_index": row["batch_index"],
                    "circuit_index_in_job": row["circuit_index_in_job"],
                    "pauli_label_qiskit_order": row["pauli_label_qiskit_order"],
                    "pauli_label_circuit_order": row["pauli_label_circuit_order"],
                    "coefficient_real": row["coefficient_real"],
                    "shots": row["shots"],
                    "ideal_expectation_value": row["ideal_expectation_value"],
                    "measured_expectation_value": row["measured_expectation_value"],
                    "expectation_error": row["expectation_error"],
                    "ideal_contribution": row["ideal_contribution"],
                    "measured_contribution": row["measured_contribution"],
                    "contribution_error": row["contribution_error"],
                    "abs_expectation_error": row["abs_expectation_error"],
                    "abs_contribution_error": row["abs_contribution_error"],
                    "counts_json": json.dumps(row["counts"], sort_keys=True),
                }
            )
    return output


def run_alpha_projector_pipeline(
    *,
    target_size: int,
    shots_per_term: int,
    circuits_per_job: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    results_csv_path: str | Path = DEFAULT_RESULTS_CSV_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    terms_csv_path: str | Path = DEFAULT_TERMS_CSV_PATH,
    result_timeout: float = 10800.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, _, _ = submit_alpha_projector_jobs(
        target_size=target_size,
        shots_per_term=shots_per_term,
        circuits_per_job=circuits_per_job,
        config=config,
        token=token,
        plan_path=plan_path,
        matrix_path=matrix_path,
    )
    results_payload = retrieve_alpha_projector_results(
        config=config,
        token=token,
        plan_path=plan_path,
        results_path=results_path,
        results_csv_path=results_csv_path,
        timeout=result_timeout,
    )
    summary, _ = evaluate_alpha_projector_results(
        plan_path=plan_path,
        results_path=results_path,
        summary_path=summary_path,
        terms_csv_path=terms_csv_path,
    )
    return plan, results_payload, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a live alpha-projector graph-state witness workflow on IQM."
    )
    parser.add_argument("--target-size", type=int, default=10)
    parser.add_argument("--shots", type=int, default=10, dest="shots_per_term")
    parser.add_argument("--circuits-per-job", type=int, default=100)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument("--matrix-path", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--results-csv-path", default=DEFAULT_RESULTS_CSV_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--terms-csv-path", default=DEFAULT_TERMS_CSV_PATH)
    parser.add_argument("--result-timeout", type=float, default=10800.0)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--quantum-computer", default=DEFAULT_QUANTUM_COMPUTER)
    parser.add_argument("--calibration-set-id")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--token")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = ResonanceConfig(
        server_url=args.server_url,
        quantum_computer=args.quantum_computer,
        calibration_set_id=args.calibration_set_id,
        timeout=args.timeout,
    )
    plan, results_payload, summary = run_alpha_projector_pipeline(
        target_size=args.target_size,
        shots_per_term=args.shots_per_term,
        circuits_per_job=args.circuits_per_job,
        config=config,
        token=args.token,
        plan_path=args.plan_path,
        matrix_path=args.matrix_path,
        results_path=args.results_path,
        results_csv_path=args.results_csv_path,
        summary_path=args.summary_path,
        terms_csv_path=args.terms_csv_path,
        result_timeout=args.result_timeout,
    )
    print(f"Saved plan to: {args.plan_path}")
    print(f"Saved witness matrix to: {args.matrix_path}")
    print(f"Saved measurement results to: {args.results_path}")
    print(f"Saved measurement results CSV to: {args.results_csv_path}")
    print(f"Saved witness summary to: {args.summary_path}")
    print(f"Saved per-term comparison CSV to: {args.terms_csv_path}")
    print(f"Qubits: {plan['num_qubits']}")
    print(f"Pauli terms: {plan['num_pauli_terms']}")
    print(f"Jobs submitted: {plan['num_jobs']}")
    print(f"Alpha: {plan['alpha']:.6f}")
    print(
        "Job statuses: "
        + ", ".join(status["status"] for status in results_payload["job_statuses"])
    )
    print(f"Estimated projector expectation: {summary['estimated_graph_state_projector_expectation']:.6f}")
    print(f"Measured alpha-witness expectation: {summary['measured_witness_expectation']:.6f}")
    print(f"Entangled by alpha witness: {summary['is_entangled_by_alpha_witness']}")


if __name__ == "__main__":
    main()
