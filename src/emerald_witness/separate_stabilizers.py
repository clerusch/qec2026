from __future__ import annotations

from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from .auth import explicit_token_environment, resolve_token
from .config import ResonanceConfig
from .io_utils import load_json, save_json
from .measurement import (
    _resolve_physical_qubit_indices,
    _transpile_restricted_circuits,
    build_emerald_graph_state_circuit,
    build_measurement_circuit,
    build_plan_payload,
    generate_stabilizer_generators,
)
from .models import MeasurementSetting
from .paths import measurement_data_path
from .witness import DEFAULT_SUMMARY_PATH, evaluate_stabilizer_witness


DEFAULT_SEPARATE_PLAN_PATH = measurement_data_path("emerald_separate_stabilizer_measurement_plan.json")
DEFAULT_SEPARATE_RESULTS_PATH = measurement_data_path("emerald_separate_stabilizer_measurement_results.json")


def _chunk_sequence(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def build_separate_stabilizer_settings(num_qubits: int) -> list[MeasurementSetting]:
    settings: list[MeasurementSetting] = []
    for stabilizer_index in range(num_qubits):
        bases = {qubit_index: "Z" for qubit_index in range(num_qubits)}
        bases[stabilizer_index] = "X"
        settings.append(
            MeasurementSetting(
                setting_id=stabilizer_index,
                bases=bases,
                measured_stabilizers=[stabilizer_index],
            )
        )
    return settings


def build_separate_stabilizer_circuits(
    base_circuit: QuantumCircuit,
    num_qubits: int,
) -> tuple[list[QuantumCircuit], list[dict[str, Any]]]:
    settings = build_separate_stabilizer_settings(num_qubits)

    circuits: list[QuantumCircuit] = []
    measurement_settings: list[dict[str, Any]] = []
    for setting in settings:
        circuit, measurement_info = build_measurement_circuit(base_circuit, setting)
        circuits.append(circuit)
        measurement_settings.append(measurement_info)

    return circuits, measurement_settings


def prepare_separate_stabilizer_plan(
    *,
    target_size: int,
    shots_per_stabilizer: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_SEPARATE_PLAN_PATH,
) -> tuple[dict[str, Any], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    base_circuit, graph, qubit_index_map = build_emerald_graph_state_circuit(
        target_size=target_size,
        config=config,
        token=resolved_token,
        add_barrier=False,
    )
    circuits, measurement_settings = build_separate_stabilizer_circuits(
        base_circuit,
        base_circuit.num_qubits,
    )
    stabilizers = generate_stabilizer_generators(graph, qubit_index_map)
    plan = build_plan_payload(
        graph=graph,
        qubit_index_map=qubit_index_map,
        stabilizers=stabilizers,
        measurement_settings=measurement_settings,
        shots_per_setting=shots_per_stabilizer,
        config=config,
    )
    plan["measurement_mode"] = "separate_stabilizers"
    save_json(plan, plan_path)
    return plan, circuits


def submit_separate_stabilizer_jobs(
    *,
    target_size: int,
    shots_per_stabilizer: int,
    circuits_per_job: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_SEPARATE_PLAN_PATH,
) -> tuple[dict[str, Any], list[QuantumCircuit], list[QuantumCircuit]]:
    resolved_token = resolve_token(token)
    plan, circuits = prepare_separate_stabilizer_plan(
        target_size=target_size,
        shots_per_stabilizer=shots_per_stabilizer,
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

        backend_max_circuits = getattr(backend, "max_circuits", None)
        effective_circuits_per_job = circuits_per_job
        if backend_max_circuits is not None:
            effective_circuits_per_job = min(effective_circuits_per_job, backend_max_circuits)
        if effective_circuits_per_job <= 0:
            raise RuntimeError("No usable circuit batch size was available for the backend.")

        qubit_index_to_name = dict(enumerate(ordered_names))
        transpiled_chunks = _chunk_sequence(transpiled_circuits, effective_circuits_per_job)
        submitted_jobs: list[dict[str, Any]] = []
        for batch_index, circuit_chunk in enumerate(transpiled_chunks, start=1):
            setting_index_start = (batch_index - 1) * effective_circuits_per_job
            setting_index_end = setting_index_start + len(circuit_chunk) - 1
            job = backend.run(
                circuit_chunk,
                shots=shots_per_stabilizer,
                qubit_index_to_name=qubit_index_to_name,
            )
            submitted_jobs.append(
                {
                    "batch_index": batch_index,
                    "job_id": job.job_id(),
                    "num_circuits": len(circuit_chunk),
                    "shots_per_circuit": shots_per_stabilizer,
                    "total_shots": len(circuit_chunk) * shots_per_stabilizer,
                    "setting_index_start": setting_index_start,
                    "setting_index_end": setting_index_end,
                }
            )

    saved_plan = load_json(plan_path)
    saved_plan["measurement_mode"] = "separate_stabilizers"
    saved_plan["circuits_per_job"] = effective_circuits_per_job
    saved_plan["submitted_jobs"] = submitted_jobs
    saved_plan["num_jobs"] = len(submitted_jobs)
    saved_plan["backend_name"] = getattr(backend, "name", None)
    saved_plan["backend_max_circuits"] = getattr(backend, "max_circuits", None)
    saved_plan["physical_qubit_indices"] = _resolve_physical_qubit_indices(
        backend,
        saved_plan["ordered_qubit_names"],
    )
    save_json(saved_plan, plan_path)

    return saved_plan, circuits, transpiled_circuits


def retrieve_separate_stabilizer_results(
    *,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_SEPARATE_PLAN_PATH,
    results_path: str | Path = DEFAULT_SEPARATE_RESULTS_PATH,
    timeout: float = 10800.0,
) -> dict[str, Any]:
    resolved_token = resolve_token(token)
    plan = load_json(plan_path)
    submitted_jobs = plan.get("submitted_jobs", [])
    if not submitted_jobs:
        raise RuntimeError("No submitted jobs were found in the plan.")

    from iqm.qiskit_iqm import IQMProvider

    with explicit_token_environment(resolved_token):
        provider = IQMProvider(
            config.server_url,
            quantum_computer=config.quantum_computer,
            token=resolved_token,
        )
        backend = provider.get_backend()

        measurement_results: list[dict[str, Any]] = []
        job_statuses: list[dict[str, Any]] = []
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
                setting_index = int(job_info["setting_index_start"]) + circuit_index
                setting = plan["measurement_settings"][setting_index]
                counts = {
                    str(bitstring): int(count)
                    for bitstring, count in result.get_counts(circuit_index).items()
                }
                measurement_results.append(
                    {
                        "job_id": str(job_info["job_id"]),
                        "batch_index": int(job_info["batch_index"]),
                        "setting_index": setting_index,
                        "setting_id": int(setting["setting_id"]),
                        "bases": setting["bases"],
                        "measured_stabilizers": setting["measured_stabilizers"],
                        "shots": sum(counts.values()),
                        "counts": counts,
                    }
                )

    measurement_results.sort(key=lambda item: int(item["setting_index"]))
    payload = {
        "server_url": config.server_url,
        "quantum_computer": config.quantum_computer,
        "measurement_mode": plan.get("measurement_mode", "separate_stabilizers"),
        "num_qubits": plan["num_qubits"],
        "ordered_qubit_names": plan["ordered_qubit_names"],
        "stabilizers": plan["stabilizers"],
        "measurement_settings": plan["measurement_settings"],
        "job_statuses": job_statuses,
        "measurement_results": measurement_results,
    }
    save_json(payload, results_path)
    return payload


def run_separate_stabilizer_pipeline(
    *,
    target_size: int,
    shots_per_stabilizer: int,
    circuits_per_job: int,
    config: ResonanceConfig,
    token: str | None = None,
    plan_path: str | Path = DEFAULT_SEPARATE_PLAN_PATH,
    results_path: str | Path = DEFAULT_SEPARATE_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    result_timeout: float = 10800.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, _, _ = submit_separate_stabilizer_jobs(
        target_size=target_size,
        shots_per_stabilizer=shots_per_stabilizer,
        circuits_per_job=circuits_per_job,
        config=config,
        token=token,
        plan_path=plan_path,
    )
    results_payload = retrieve_separate_stabilizer_results(
        config=config,
        token=token,
        plan_path=plan_path,
        results_path=results_path,
        timeout=result_timeout,
    )
    summary = evaluate_stabilizer_witness(results_payload)
    save_json(summary, summary_path)
    return plan, results_payload, summary
