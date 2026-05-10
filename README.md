# Clean Emerald Witness Workflow

This directory contains a clean, modular codebase for the workflow we have been building in the root repo:

1. fetch IQM Emerald calibration data,
2. choose a good connected subgraph,
3. build a graph-state preparation circuit,
4. group graph-state stabilizer measurements into a small number of basis settings,
5. submit those settings to IQM,
6. retrieve fresh counts,
7. compute the stabilizer witness,
8. plot the exact subgraph used in the run.

## Layout

```text
clean/
├── README.md
├── pyproject.toml
└── src/emerald_witness/
    ├── __init__.py
    ├── __main__.py
    ├── auth.py
    ├── characterization.py
    ├── cli.py
    ├── config.py
    ├── io_utils.py
    ├── measurement.py
    ├── models.py
    ├── plotting.py
    ├── subgraph.py
    └── witness.py
```

## Install

Use an environment with `qiskit`, `networkx`, `requests`, `matplotlib`, and the IQM Qiskit integration.

Example:

```bash
cd clean
python -m pip install -e '.[iqm]'
```

If you do not want to install the package, you can also run it with:

```bash
PYTHONPATH=clean/src python -m emerald_witness ...
```

## Authentication

The CLI accepts `--token`, or reads from:

```bash
export IQM_TOKEN='...'
```

It also accepts `QRISP_API_TOKEN` as a fallback for token resolution.

## Typical usage

Prepare a 15-qubit grouped measurement plan:

```bash
PYTHONPATH=clean/src python -m emerald_witness prepare \
  --target-size 15 \
  --shots 1000 \
  --plan-path measurement_data/clean_emerald_15q_plan.json
```

Submit the real IQM job:

```bash
PYTHONPATH=clean/src python -m emerald_witness submit \
  --target-size 15 \
  --shots 1000 \
  --plan-path measurement_data/clean_emerald_15q_plan.json
```

Retrieve the fresh hardware counts:

```bash
PYTHONPATH=clean/src python -m emerald_witness retrieve \
  --plan-path measurement_data/clean_emerald_15q_plan.json \
  --results-path measurement_data/clean_emerald_15q_results.json
```

Evaluate the stabilizer witness:

```bash
PYTHONPATH=clean/src python -m emerald_witness evaluate \
  --results-path measurement_data/clean_emerald_15q_results.json \
  --output-path measurement_data/clean_emerald_15q_summary.json
```

Render the exact subgraph used in that plan:

```bash
PYTHONPATH=clean/src python -m emerald_witness plot \
  --plan-path measurement_data/clean_emerald_15q_plan.json \
  --output plots/clean_emerald_15q_subgraph.png
```

## What the code measures

The witness evaluation uses the grouped graph-state stabilizer generators

```text
K_i = X_i ∏_{j in N(i)} Z_j
```

and computes

```text
W = (n - 1) I - Σ_i K_i
```

from measured stabilizer expectation values. A negative witness value certifies entanglement according to this witness.

## Notes

- `measurement.py` handles circuit construction, plan creation, IQM submission, and result retrieval.
- `witness.py` converts bitstring counts into stabilizer expectations and the final witness value.
- `plotting.py` renders either a stored plan subgraph or a newly fetched one.
- `characterization.py` isolates the Resonance REST API handling from the circuit and witness logic.
# qec2026
