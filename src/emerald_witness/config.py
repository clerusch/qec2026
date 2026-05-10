from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SERVER_URL = "https://resonance.meetiqm.com"
DEFAULT_QUANTUM_COMPUTER = "emerald"
DEFAULT_TOKEN_ENV_VARS = ("IQM_TOKEN", "QRISP_API_TOKEN")


@dataclass(frozen=True)
class ResonanceConfig:
    server_url: str = DEFAULT_SERVER_URL
    quantum_computer: str = DEFAULT_QUANTUM_COMPUTER
    calibration_set_id: str | None = None
    timeout: float = 30.0
