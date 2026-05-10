from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import load_json, save_json
from .paths import measurement_data_path


DEFAULT_SUMMARY_PATH = measurement_data_path("emerald_stabilizer_witness_summary.json")


def _clean_bitstring(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def _bit_for_qubit(bitstring: str, qubit_index: int) -> int:
    cleaned = _clean_bitstring(bitstring)
    if qubit_index >= len(cleaned):
        raise ValueError(f"Bitstring {bitstring!r} is too short for qubit index {qubit_index}.")
    return int(cleaned[-1 - qubit_index])


def _parity_eigenvalue(bitstring: str, qubit_indices: list[int]) -> int:
    parity = sum(_bit_for_qubit(bitstring, qubit_index) for qubit_index in qubit_indices) % 2
    return 1 if parity == 0 else -1


def expectation_value_from_counts(
    counts: dict[str, int],
    qubit_indices: list[int],
) -> float:
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("Cannot compute an expectation value from zero shots.")
    weighted_sum = sum(
        _parity_eigenvalue(bitstring, qubit_indices) * count
        for bitstring, count in counts.items()
    )
    return weighted_sum / shots


def stabilizer_qubits_from_label(stabilizer: str) -> list[int]:
    indices: list[int] = []
    current_digits: list[str] = []
    for character in stabilizer:
        if character in {"X", "Y", "Z"}:
            if current_digits:
                indices.append(int("".join(current_digits)))
                current_digits = []
        elif character.isdigit():
            current_digits.append(character)
        else:
            raise ValueError(f"Unsupported stabilizer character {character!r}.")
    if current_digits:
        indices.append(int("".join(current_digits)))
    return indices


def calculate_stabilizer_expectations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stabilizers = payload["stabilizers"]
    results_by_setting_id = {
        int(result["setting_id"]): result
        for result in payload["measurement_results"]
    }

    expectations: list[dict[str, Any]] = []
    for setting in payload["measurement_settings"]:
        setting_id = int(setting["setting_id"])
        result = results_by_setting_id[setting_id]
        counts = {str(bitstring): int(count) for bitstring, count in result["counts"].items()}

        for stabilizer_index in setting["measured_stabilizers"]:
            stabilizer = stabilizers[int(stabilizer_index)]
            qubit_indices = stabilizer_qubits_from_label(stabilizer)
            expectations.append(
                {
                    "stabilizer_index": int(stabilizer_index),
                    "stabilizer": stabilizer,
                    "setting_id": setting_id,
                    "qubit_indices": qubit_indices,
                    "shots": sum(counts.values()),
                    "expectation_value": expectation_value_from_counts(counts, qubit_indices),
                }
            )

    expectations.sort(key=lambda item: item["stabilizer_index"])
    return expectations


def evaluate_stabilizer_witness(payload: dict[str, Any]) -> dict[str, Any]:
    expectations = calculate_stabilizer_expectations(payload)
    num_qubits = int(payload["num_qubits"])
    stabilizer_sum = sum(item["expectation_value"] for item in expectations)
    witness_value = (num_qubits - 1) - stabilizer_sum

    return {
        "num_qubits": num_qubits,
        "num_stabilizers": len(expectations),
        "ordered_qubit_names": payload.get("ordered_qubit_names"),
        "stabilizer_expectation_sum": stabilizer_sum,
        "entanglement_threshold_sum": num_qubits - 1,
        "stabilizer_witness_value": witness_value,
        "is_entangled_by_witness": witness_value < 0,
        "ideal_graph_state_witness_value": -1.0,
        "stabilizer_expectations": expectations,
    }


def evaluate_results_file(
    results_path: str | Path,
    output_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> tuple[dict[str, Any], Path]:
    payload = load_json(results_path)
    summary = evaluate_stabilizer_witness(payload)
    output = save_json(summary, output_path)
    return summary, output
