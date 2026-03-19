#!/usr/bin/env python
"""Night-system benchmark for bottleneck monitoring.

This script runs a single, reproducible benchmark workload and stores
structured metrics for nightly performance tracking. It supports choosing
either DA or PTDA from the CLI.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from ._benchmark_utils import add_beta_and_seed_args, ensure_project_root_on_path
else:
    from _benchmark_utils import add_beta_and_seed_args, ensure_project_root_on_path

ROOT_DIR = ensure_project_root_on_path(__file__)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "benchmarks" / "outputs"

from benchmarks.problem_generators import generate_dense_problem_dict
from digitalannealer import DASampler, PTDASampler


def _run_warmup(sampler_type: str) -> None:
    """Run a tiny problem to warm up GPU before timed benchmark."""
    try:
        warmup_h = {0: 0.0, 1: 0.0}
        warmup_J = {(0, 1): -1.0}
        if sampler_type == "da":
            DASampler().sample_ising(warmup_h, warmup_J, num_sweeps=10, num_reads=1, sparse=False)
        else:
            PTDASampler().sample_ising(
                warmup_h, warmup_J, num_sweeps=10, num_reads=1, num_replicas=2, sparse=False
            )
    except Exception:
        # Warmup is non-critical; ignore any failures so that main benchmark still runs.
        pass


def _run_da(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    args: argparse.Namespace,
) -> tuple[float, float]:
    sampler = DASampler(seed=args.seed)
    start = time.perf_counter()
    result = sampler.sample_ising(
        h,
        J,
        num_sweeps=args.num_sweeps,
        num_reads=args.num_reads,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        offset_delta=args.offset_delta,
        sparse=False,
    )
    elapsed = time.perf_counter() - start
    return elapsed, float(result.first.energy)


def _run_ptda(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    args: argparse.Namespace,
) -> tuple[float, float]:
    sampler = PTDASampler(seed=args.seed)
    start = time.perf_counter()
    result = sampler.sample_ising(
        h,
        J,
        num_sweeps=args.num_sweeps,
        num_reads=args.num_reads,
        num_replicas=args.num_replicas,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        exchange_interval=args.exchange_interval,
        adapt_betas=not args.disable_beta_adaptation,
        adapt_rounds=args.adapt_rounds,
        adapt_sweeps=args.adapt_sweeps,
        log_adaptation=args.log_adaptation,
        offset_delta=args.offset_delta,
        sparse=False,
    )
    elapsed = time.perf_counter() - start
    return elapsed, float(result.first.energy)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    h, J = generate_dense_problem_dict(args.problem_size, seed=args.seed)

    if not args.no_warmup:
        _run_warmup(args.sampler)

    ts = datetime.now(tz=timezone.utc)
    run_id = ts.strftime("%Y%m%dT%H%M%S.%fZ")

    elapsed: float
    best_energy: float
    if args.sampler == "da":
        elapsed, best_energy = _run_da(h, J, args)
    elif args.sampler == "ptda":
        elapsed, best_energy = _run_ptda(h, J, args)
    else:
        raise ValueError(f"Unsupported sampler: {args.sampler}")

    sweeps_per_sec = args.num_sweeps * args.num_reads / elapsed

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": ts.isoformat(),
        "hostname": platform.node(),
        "sampler": args.sampler,
        "problem_size": args.problem_size,
        "num_sweeps": args.num_sweeps,
        "num_reads": args.num_reads,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "offset_delta": args.offset_delta,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "sweeps_per_sec": sweeps_per_sec,
        "best_energy": best_energy,
        "status": "success",
    }

    if args.sampler == "ptda":
        record.update(
            {
                "num_replicas": args.num_replicas,
                "exchange_interval": args.exchange_interval,
                "adapt_betas": not args.disable_beta_adaptation,
                "adapt_rounds": args.adapt_rounds,
                "adapt_sweeps": args.adapt_sweeps,
                "log_adaptation": args.log_adaptation,
            }
        )

    return record


def print_summary(record: dict[str, Any]) -> None:
    print("Digital Annealer - Night Bottleneck Monitor")
    print("=" * 80)
    print(
        f"Sampler: {record['sampler']} | N={record['problem_size']} | "
        f"Sweeps={record['num_sweeps']} | Reads={record['num_reads']}"
    )
    print(f"Elapsed: {record['elapsed_sec']:.6f}s")
    print(f"Sweeps/s: {record['sweeps_per_sec']:.2f}")
    print(f"Best energy: {record['best_energy']:.6f}")
    print(f"Run ID: {record['run_id']}")


def save_jsonl(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nAppended monitoring record to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Night-system bottleneck monitoring benchmark for DA/PTDA"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["da", "ptda"],
        required=True,
        help="Sampler type to benchmark: da or ptda",
    )
    parser.add_argument(
        "--problem-size",
        type=int,
        required=True,
        help="Number of spins (N)",
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        required=True,
        help="Number of Monte Carlo sweeps",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        required=True,
        help="Number of independent reads",
    )
    add_beta_and_seed_args(parser)
    parser.add_argument(
        "--offset-delta",
        type=float,
        default=0.1,
        help="Dynamic offset increment on rejection (default: 0.1)",
    )

    parser.add_argument(
        "--num-replicas",
        type=int,
        default=10,
        help="PTDA only: number of replicas (default: 10)",
    )
    parser.add_argument(
        "--exchange-interval",
        type=int,
        default=10,
        help="PTDA only: exchange attempt interval in sweeps (default: 10)",
    )
    parser.add_argument(
        "--disable-beta-adaptation",
        action="store_true",
        help="PTDA only: disable beta adaptation",
    )
    parser.add_argument(
        "--adapt-rounds",
        type=int,
        default=3,
        help="PTDA only: adaptation rounds when enabled (default: 3)",
    )
    parser.add_argument(
        "--adapt-sweeps",
        type=int,
        default=None,
        help="PTDA only: sweeps per adaptation round (default: auto)",
    )
    parser.add_argument(
        "--log-adaptation",
        action="store_true",
        help="PTDA only: include adaptation history in result info",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "night_monitor.jsonl",
        help="JSONL output path for monitoring records",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup iteration before timing",
    )

    args = parser.parse_args()

    if args.problem_size < 2:
        parser.error("--problem-size must be >= 2")
    if args.num_sweeps < 1:
        parser.error("--num-sweeps must be >= 1")
    if args.num_reads < 1:
        parser.error("--num-reads must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    try:
        record = run_once(args)
        print_summary(record)
        save_jsonl(record, args.output)
    except Exception as e:
        ts = datetime.now(tz=timezone.utc).isoformat()
        error_record = {
            "timestamp_utc": ts,
            "sampler": args.sampler,
            "problem_size": args.problem_size,
            "num_sweeps": args.num_sweeps,
            "num_reads": args.num_reads,
            "seed": args.seed,
            "status": f"error: {e}",
        }
        print(f"Benchmark failed: {e}")
        save_jsonl(error_record, args.output)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
