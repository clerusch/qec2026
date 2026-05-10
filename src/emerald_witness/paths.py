from __future__ import annotations

from pathlib import Path


MEASUREMENT_DATA_DIR = Path("measurement_data")
PLOTS_DIR = Path("plots")


def measurement_data_path(filename: str) -> str:
    return str(MEASUREMENT_DATA_DIR / filename)


def plot_path(filename: str) -> str:
    return str(PLOTS_DIR / filename)
