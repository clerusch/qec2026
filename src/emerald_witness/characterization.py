from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests

from .config import ResonanceConfig


class IQMCharacterizationError(RuntimeError):
    """Raised when IQM characterization data cannot be fetched or parsed."""


def _normalize_server_url(
    base_url: str,
    quantum_computer: str | None,
) -> tuple[str, str]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise IQMCharacterizationError(
            f"Unsupported URL scheme in {base_url!r}. Expected http or https."
        )

    hostname = parsed.hostname or ""
    path_segments = [segment for segment in parsed.path.split("/") if segment]

    quantum_computer_from_path = None
    if path_segments:
        quantum_computer_from_path = path_segments[-1].removesuffix(":timeslot")
        if quantum_computer_from_path:
            path_segments = path_segments[:-1]

    if hostname.startswith("cocos."):
        hostname = hostname.removeprefix("cocos.")

    resolved_quantum_computer = quantum_computer or quantum_computer_from_path
    if not resolved_quantum_computer:
        raise IQMCharacterizationError(
            "Quantum computer alias is missing. Pass one explicitly or use a URL that ends with the alias."
        )

    port_suffix = f":{parsed.port}" if parsed.port else ""
    base_path = f"/{'/'.join(path_segments)}" if path_segments else ""
    root_url = f"{parsed.scheme}://{hostname}{port_suffix}{base_path}".rstrip("/")
    return root_url, resolved_quantum_computer


def _legacy_backend_url(root_url: str, quantum_computer: str) -> str:
    parsed = urlparse(root_url)
    hostname = parsed.hostname or ""
    if not hostname.startswith("cocos."):
        hostname = f"cocos.{hostname}"
    port_suffix = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{hostname}{port_suffix}/{quantum_computer}"


def _fetch_json(url: str, token: str, timeout: float) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = f"Request to {url} failed with status {response.status_code}"
        details = response.text.strip()
        if details:
            message = f"{message}: {details}"
        if response.status_code == 401:
            message = (
                f"{message}. Check that the token is valid and does not already include the 'Bearer ' prefix."
            )
        raise IQMCharacterizationError(message) from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise IQMCharacterizationError(f"Unexpected JSON payload from {url}: {payload!r}")
    return payload


def _api_v1_url(root_url: str, path: str) -> str:
    return f"{root_url.rstrip('/')}/api/v1/{path.lstrip('/')}"


def _fetch_current_api_payloads(
    root_url: str,
    quantum_computer: str,
    calibration_set_id: str | None,
    token: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_segment = calibration_set_id or "default"
    architecture_payload = _fetch_json(
        _api_v1_url(
            root_url,
            f"calibration-sets/{quantum_computer}/{calibration_segment}/dynamic-quantum-architecture",
        ),
        token=token,
        timeout=timeout,
    )
    metrics_payload = _fetch_json(
        _api_v1_url(
            root_url,
            f"calibration-sets/{quantum_computer}/{calibration_segment}/metrics",
        ),
        token=token,
        timeout=timeout,
    )
    return architecture_payload, metrics_payload


def _fetch_legacy_api_payloads(
    root_url: str,
    quantum_computer: str,
    calibration_set_id: str | None,
    token: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend_url = _legacy_backend_url(root_url, quantum_computer)
    metrics_suffix = (
        f"calibration/metrics/{calibration_set_id}"
        if calibration_set_id
        else "calibration/metrics/latest"
    )
    architecture_payload = _fetch_json(
        f"{backend_url}/quantum-architecture",
        token=token,
        timeout=timeout,
    )
    metrics_payload = _fetch_json(
        f"{backend_url}/{metrics_suffix}",
        token=token,
        timeout=timeout,
    )
    return architecture_payload, metrics_payload


def _find_metric_block(metrics: dict[str, Any], *suffixes: str) -> dict[str, Any]:
    for suffix in suffixes:
        if suffix in metrics and isinstance(metrics[suffix], dict):
            return metrics[suffix]

    for suffix in suffixes:
        for key, value in metrics.items():
            if key.endswith(suffix) and isinstance(value, dict):
                return value

    return {}


def _extract_components(text: str) -> list[str]:
    seen: set[str] = set()
    components: list[str] = []
    for component in re.findall(r"(?:QB|COMPR)\d+", text):
        if component not in seen:
            components.append(component)
            seen.add(component)
    return components


def _normalize_metric_block(
    metric_block: dict[str, Any],
    *,
    is_edge_metric: bool,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for component_name, entry in metric_block.items():
        if not isinstance(entry, dict):
            continue

        value = entry.get("value")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = value

        item: dict[str, Any] = {
            "value": numeric_value,
            "unit": entry.get("unit"),
            "uncertainty": entry.get("uncertainty", entry.get("uncertainity")),
            "timestamp": entry.get("timestamp"),
            "implementation": entry.get("implementation"),
        }
        if is_edge_metric:
            item["qubits"] = _extract_components(component_name)
        normalized[component_name] = item
    return normalized


def _normalize_observation(
    observation: dict[str, Any],
    *,
    key: str,
    components: list[str],
    is_edge_metric: bool,
) -> tuple[str, dict[str, Any]] | None:
    dut_field = observation.get("dut_field")
    if not isinstance(dut_field, str):
        return None

    value = observation.get("value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = value

    item: dict[str, Any] = {
        "value": numeric_value,
        "unit": observation.get("unit"),
        "uncertainty": observation.get("uncertainty"),
        "created_timestamp": observation.get("created_timestamp"),
        "modified_timestamp": observation.get("modified_timestamp"),
        "dut_field": dut_field,
    }
    if is_edge_metric:
        item["qubits"] = components
    return key, item


def _split_observation_name(dut_field: str) -> list[str]:
    return dut_field.split(":", 1)[0].split(".")


def _parse_metrics_from_observations(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    t1_times: dict[str, Any] = {}
    single_qubit_prx_fidelities: dict[str, Any] = {}
    two_qubit_cz_fidelities: dict[str, Any] = {}
    two_qubit_clifford_fidelities: dict[str, Any] = {}

    for observation in observations:
        dut_field = observation.get("dut_field")
        if not isinstance(dut_field, str):
            continue

        parts = _split_observation_name(dut_field)
        if len(parts) >= 4 and parts[:2] == ["characterization", "model"] and parts[-1] == "t1_time":
            component = parts[2]
            normalized = _normalize_observation(
                observation,
                key=component,
                components=[component],
                is_edge_metric=False,
            )
            if normalized is not None:
                key, item = normalized
                t1_times[key] = item
            continue

        if len(parts) >= 6 and parts[0] == "metrics" and parts[-1] == "fidelity":
            method = parts[1]
            gate = parts[2]
            implementation = parts[3]
            locus_str = parts[4]
            components = locus_str.split("__")
            normalized = _normalize_observation(
                observation,
                key=locus_str,
                components=components,
                is_edge_metric=len(components) == 2,
            )
            if normalized is None:
                continue

            key, item = normalized
            item["implementation"] = implementation
            item["method"] = method
            item["gate"] = gate
            if method == "rb" and gate == "prx":
                single_qubit_prx_fidelities[key] = item
            elif method == "irb" and gate == "cz":
                two_qubit_cz_fidelities[key] = item
            elif method == "rb" and gate == "clifford":
                two_qubit_clifford_fidelities[key] = item

    return (
        t1_times,
        single_qubit_prx_fidelities,
        two_qubit_cz_fidelities,
        two_qubit_clifford_fidelities,
    )


def _parse_quality_metrics_payload(
    metrics_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    observations = metrics_payload.get("observations")
    if isinstance(observations, list):
        observation_dicts = [obs for obs in observations if isinstance(obs, dict)]
        return _parse_metrics_from_observations(observation_dicts)

    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, dict):
        t1_times = _normalize_metric_block(
            _find_metric_block(metrics, "t1_time"),
            is_edge_metric=False,
        )
        prx = _normalize_metric_block(
            _find_metric_block(metrics, "prx_rb_fidelity"),
            is_edge_metric=False,
        )
        cz = _normalize_metric_block(
            _find_metric_block(metrics, "cz_irb_fidelity"),
            is_edge_metric=True,
        )
        clifford = _normalize_metric_block(
            _find_metric_block(metrics, "clifford_rb_fidelity"),
            is_edge_metric=True,
        )
        return t1_times, prx, cz, clifford

    raise IQMCharacterizationError(
        "Malformed calibration metrics payload: expected either an 'observations' list or a 'metrics' mapping."
    )


def _extract_connectivity_from_dynamic_architecture(
    architecture: dict[str, Any],
) -> list[list[str]]:
    connectivity: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    gates = architecture.get("gates", {})
    if not isinstance(gates, dict):
        return connectivity

    for gate_info in gates.values():
        if not isinstance(gate_info, dict):
            continue
        implementations = gate_info.get("implementations", {})
        if not isinstance(implementations, dict):
            continue
        for implementation_info in implementations.values():
            if not isinstance(implementation_info, dict):
                continue
            loci = implementation_info.get("loci", [])
            if not isinstance(loci, (list, tuple)):
                continue
            for locus in loci:
                if not isinstance(locus, (list, tuple)) or len(locus) != 2:
                    continue
                edge = [str(component) for component in locus]
                edge_key = tuple(edge)
                if edge_key not in seen:
                    connectivity.append(edge)
                    seen.add(edge_key)

    return connectivity


def _extract_calibrated_connectivity(
    cz_fidelities: dict[str, Any],
    clifford_fidelities: dict[str, Any],
) -> list[list[str]]:
    connectivity: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    edge_names = list(cz_fidelities) or list(clifford_fidelities)
    for edge_name in edge_names:
        edge = _extract_components(edge_name)
        edge_key = tuple(edge)
        if len(edge) == 2 and edge_key not in seen:
            connectivity.append(edge)
            seen.add(edge_key)
    return connectivity


def get_characterization(config: ResonanceConfig, token: str) -> dict[str, Any]:
    root_url, quantum_computer = _normalize_server_url(
        config.server_url,
        config.quantum_computer,
    )

    try:
        architecture_payload, metrics_payload = _fetch_current_api_payloads(
            root_url=root_url,
            quantum_computer=quantum_computer,
            calibration_set_id=config.calibration_set_id,
            token=token,
            timeout=config.timeout,
        )
        api_layout = "current"
    except IQMCharacterizationError as current_error:
        if "status 404" not in str(current_error):
            raise

        architecture_payload, metrics_payload = _fetch_legacy_api_payloads(
            root_url=root_url,
            quantum_computer=quantum_computer,
            calibration_set_id=config.calibration_set_id,
            token=token,
            timeout=config.timeout,
        )
        api_layout = "legacy"

    t1_times, prx, cz, clifford = _parse_quality_metrics_payload(metrics_payload)

    if api_layout == "current":
        architecture = architecture_payload
        if not isinstance(architecture, dict):
            raise IQMCharacterizationError("Malformed dynamic quantum architecture payload.")
        hardware_connectivity = _extract_connectivity_from_dynamic_architecture(architecture)
        resolved_calibration_set_id = architecture.get("calibration_set_id")
        qubits = architecture.get("qubits", [])
        resonators = architecture.get("computational_resonators", [])
    else:
        architecture = architecture_payload.get("quantum_architecture", architecture_payload)
        if not isinstance(architecture, dict):
            raise IQMCharacterizationError("Malformed quantum architecture payload.")
        hardware_connectivity = [
            list(edge)
            for edge in architecture.get("qubit_connectivity", architecture.get("connectivity", []))
            if isinstance(edge, (list, tuple))
        ]
        resolved_calibration_set_id = metrics_payload.get("calibration_set_id")
        qubits = architecture.get("qubits", [])
        resonators = architecture.get("computational_resonators", [])

    return {
        "quantum_computer": quantum_computer,
        "root_url": root_url,
        "api_layout": api_layout,
        "calibration_set_id": resolved_calibration_set_id,
        "qubits": qubits,
        "computational_resonators": resonators,
        "hardware_connectivity": hardware_connectivity,
        "calibrated_connectivity": _extract_calibrated_connectivity(cz, clifford),
        "t1_times": t1_times,
        "gate_fidelities": {
            "single_qubit_prx": prx,
            "two_qubit_cz": cz,
            "two_qubit_clifford": clifford,
        },
    }
