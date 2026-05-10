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

Run the live alpha-1 GG projector-witness workflow from the repository root:

```bash
IQM_TOKEN='your token here' PYTHONPATH=clean/src /Users/moritzkahrweg/.pyenv/versions/3.13.7/bin/poetry run python -m emerald_witness.alpha_projector_witness \
  --target-size 9 \
  --shots 10 \
  --circuits-per-job 100 \
  --alpha 1.0 \
  --method-label alpha_1_gg_projector_witness \
  --plan-path clean/measurement_data/emerald_12q_alpha1_gg_plan.json \
  --matrix-path clean/measurement_data/emerald_12q_alpha1_gg_matrix.npy \
  --results-path clean/measurement_data/emerald_12q_alpha1_gg_results.json \
  --results-csv-path clean/measurement_data/emerald_12q_alpha1_gg_results.csv \
  --summary-path clean/measurement_data/emerald_12q_alpha1_gg_summary.json \
  --terms-csv-path clean/measurement_data/emerald_12q_alpha1_gg_terms.csv
```

This command executes the full workflow end to end:

1. Fetch the current IQM Emerald calibration and topology data.
2. Select a connected hardware subgraph with `--target-size` qubits.
3. Build the graph-state preparation circuit for that subgraph.
4. Construct the projector witness

   ```text
   W_G = alpha I - |G><G|
   ```

   with `--alpha 1.0` for this alpha-1 GG run.
5. Decompose the witness into Pauli terms.
6. Build one measurement circuit per Pauli term.
7. Submit the circuits to IQM in batches of `--circuits-per-job`.
8. Retrieve hardware counts for every submitted batch.
9. Recombine the measured Pauli expectations into a projector expectation and witness value.
10. Write the plan, matrix, raw results, CSV exports, and final summary files.

The most important options are:

| Option | Meaning |
| --- | --- |
| `IQM_TOKEN='your token here'` | Authenticates with IQM Resonance. You can also export `IQM_TOKEN` once instead of prefixing the command. |
| `PYTHONPATH=clean/src` | Lets Python import the local `emerald_witness` package without installing it. |
| `python -m emerald_witness.alpha_projector_witness` | Runs the alpha-projector pipeline module. |
| `--target-size 9` | Selects a 9-qubit connected Emerald subgraph. This controls the actual qubit count. |
| `--shots 10` | Uses 10 hardware shots per Pauli measurement circuit. |
| `--circuits-per-job 100` | Batches up to 100 circuits into each submitted IQM job. |
| `--alpha 1.0` | Uses the alpha-1 projector witness. Omit this flag for the automatic biseparable threshold, usually about `0.5` for graph states. |
| `--method-label alpha_1_gg_projector_witness` | Stores a descriptive label in the output JSON files. |

The output files are:

| File | Contents |
| --- | --- |
| `emerald_12q_alpha1_gg_plan.json` | Selected qubits, graph, Pauli terms, batching, submitted IQM job IDs, and run metadata. |
| `emerald_12q_alpha1_gg_matrix.npy` | Dense NumPy matrix for `alpha I - |G><G|`. |
| `emerald_12q_alpha1_gg_results.json` | Retrieved hardware counts and measured expectation value for every Pauli term. |
| `emerald_12q_alpha1_gg_results.csv` | Flat CSV version of the raw per-term measurement results. |
| `emerald_12q_alpha1_gg_summary.json` | Final fidelity estimate, witness expectation, uncertainty, and diagnostic metrics. |
| `emerald_12q_alpha1_gg_terms.csv` | Per-term comparison between ideal and measured Pauli contributions. |

Note: the filenames above use the historical `12q` prefix from the hackathon run. The actual number of qubits is controlled by `--target-size` and is recorded inside the plan and summary JSON files.

## What the code measures

The alpha-projector workflow measures the graph-state projector expansion

```text
|G><G| = 1 / 2^n * Σ_{s in S_G} s
```

and evaluates

```text
W_G = alpha I - |G><G|
```

from hardware-measured Pauli expectation values. For the alpha-1 GG run above, `alpha = 1.0`.
For genuine multipartite entanglement certification with the graph-state biseparable threshold, omit
`--alpha 1.0` so the code computes the threshold automatically.

The package also still contains the grouped stabilizer-generator workflow, which uses

```text
K_i = X_i ∏_{j in N(i)} Z_j
W = (n - 1) I - Σ_i K_i
```

from grouped stabilizer measurements.

## Notes

- `measurement.py` handles circuit construction, plan creation, IQM submission, and result retrieval.
- `witness.py` converts bitstring counts into stabilizer expectations and the final witness value.
- `plotting.py` renders either a stored plan subgraph or a newly fetched one.
- `characterization.py` isolates the Resonance REST API handling from the circuit and witness logic.
