from __future__ import annotations

import argparse

from .config import DEFAULT_QUANTUM_COMPUTER, DEFAULT_SERVER_URL, ResonanceConfig
from .measurement import (
    DEFAULT_PLAN_PATH,
    DEFAULT_RESULTS_PATH,
    prepare_measurement_plan,
    retrieve_measurement_results,
    submit_measurement_job,
)
from .paths import plot_path
from .plotting import draw_emerald_subgraph, draw_subgraph_from_plan
from .separate_stabilizers import (
    DEFAULT_SEPARATE_PLAN_PATH,
    DEFAULT_SEPARATE_RESULTS_PATH,
    run_separate_stabilizer_pipeline,
)
from .witness import DEFAULT_SUMMARY_PATH, evaluate_results_file


def _build_config(args: argparse.Namespace) -> ResonanceConfig:
    return ResonanceConfig(
        server_url=args.server_url,
        quantum_computer=args.quantum_computer,
        calibration_set_id=args.calibration_set_id,
        timeout=args.timeout,
    )


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--quantum-computer", default=DEFAULT_QUANTUM_COMPUTER)
    parser.add_argument("--calibration-set-id")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--token")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean modular workflow for IQM Emerald graph-state stabilizer witnesses."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build and save a grouped stabilizer measurement plan.")
    prepare_parser.add_argument("--target-size", type=int, default=15)
    prepare_parser.add_argument("--shots", type=int, default=1000, dest="shots_per_setting")
    prepare_parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    _add_backend_args(prepare_parser)

    submit_parser = subparsers.add_parser("submit", help="Prepare and submit a grouped stabilizer measurement job.")
    submit_parser.add_argument("--target-size", type=int, default=15)
    submit_parser.add_argument("--shots", type=int, default=1000, dest="shots_per_setting")
    submit_parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    _add_backend_args(submit_parser)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve grouped stabilizer counts from IQM.")
    retrieve_parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    retrieve_parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    retrieve_parser.add_argument("--job-id")
    retrieve_parser.add_argument("--result-timeout", type=float, default=10800.0)
    _add_backend_args(retrieve_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute the stabilizer witness from a results JSON file.")
    evaluate_parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    evaluate_parser.add_argument("--output-path", default=DEFAULT_SUMMARY_PATH)

    run_separate_parser = subparsers.add_parser(
        "run-separate-stabilizers",
        help="Run one separate measurement circuit per stabilizer and evaluate the witness.",
    )
    run_separate_parser.add_argument("--target-size", type=int, default=5)
    run_separate_parser.add_argument("--shots", type=int, default=10, dest="shots_per_stabilizer")
    run_separate_parser.add_argument("--circuits-per-job", type=int, default=100)
    run_separate_parser.add_argument("--plan-path", default=DEFAULT_SEPARATE_PLAN_PATH)
    run_separate_parser.add_argument("--results-path", default=DEFAULT_SEPARATE_RESULTS_PATH)
    run_separate_parser.add_argument("--output-path", default=DEFAULT_SUMMARY_PATH)
    run_separate_parser.add_argument("--result-timeout", type=float, default=10800.0)
    _add_backend_args(run_separate_parser)

    plot_parser = subparsers.add_parser("plot", help="Render the Emerald subgraph used in a plan or fetch a new one.")
    plot_parser.add_argument("--plan-path")
    plot_parser.add_argument("--target-size", type=int, default=15)
    plot_parser.add_argument("--output", default=plot_path("emerald_subgraph.png"))
    _add_backend_args(plot_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        summary, output = evaluate_results_file(args.results_path, args.output_path)
        print(f"Saved witness summary to: {output}")
        print(f"Stabilizer sum: {summary['stabilizer_expectation_sum']:.6f}")
        print(f"Witness value: {summary['stabilizer_witness_value']:.6f}")
        print(f"Entangled by witness: {summary['is_entangled_by_witness']}")
        return

    config = _build_config(args)

    if args.command == "prepare":
        plan, _ = prepare_measurement_plan(
            target_size=args.target_size,
            shots_per_setting=args.shots_per_setting,
            config=config,
            token=args.token,
            plan_path=args.plan_path,
        )
        print(f"Saved measurement plan to: {args.plan_path}")
        print(f"Qubits: {plan['num_qubits']}")
        print(f"Measurement settings: {plan['num_measurement_settings']}")
        return

    if args.command == "submit":
        job, plan, _, _ = submit_measurement_job(
            target_size=args.target_size,
            shots_per_setting=args.shots_per_setting,
            config=config,
            token=args.token,
            plan_path=args.plan_path,
        )
        print(f"Submitted IQM job: {job.job_id()}")
        print(f"Saved measurement plan to: {args.plan_path}")
        print(f"Measurement settings: {plan['num_measurement_settings']}")
        return

    if args.command == "retrieve":
        payload = retrieve_measurement_results(
            config=config,
            token=args.token,
            plan_path=args.plan_path,
            results_path=args.results_path,
            job_id=args.job_id,
            timeout=args.result_timeout,
        )
        print(f"Saved measurement results to: {args.results_path}")
        print(f"Job id: {payload['job_id']}")
        print(f"Status: {payload['status']}")
        return

    if args.command == "run-separate-stabilizers":
        plan, results_payload, summary = run_separate_stabilizer_pipeline(
            target_size=args.target_size,
            shots_per_stabilizer=args.shots_per_stabilizer,
            circuits_per_job=args.circuits_per_job,
            config=config,
            token=args.token,
            plan_path=args.plan_path,
            results_path=args.results_path,
            summary_path=args.output_path,
            result_timeout=args.result_timeout,
        )
        print(f"Saved measurement plan to: {args.plan_path}")
        print(f"Saved measurement results to: {args.results_path}")
        print(f"Saved witness summary to: {args.output_path}")
        print(f"Qubits: {plan['num_qubits']}")
        print(f"Separate stabilizer circuits: {plan['num_measurement_settings']}")
        print(f"Jobs submitted: {plan['num_jobs']}")
        print(f"Stabilizer sum: {summary['stabilizer_expectation_sum']:.6f}")
        print(f"Witness value: {summary['stabilizer_witness_value']:.6f}")
        print(f"Entangled by witness: {summary['is_entangled_by_witness']}")
        print(f"Job statuses: {', '.join(status['status'] for status in results_payload['job_statuses'])}")
        return

    if args.command == "plot":
        if args.plan_path:
            output = draw_subgraph_from_plan(args.plan_path, output_path=args.output)
        else:
            output = draw_emerald_subgraph(
                target_size=args.target_size,
                config=config,
                output_path=args.output,
                token=args.token,
            )
        print(output)
        return

    raise RuntimeError(f"Unsupported command {args.command!r}.")
