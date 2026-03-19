#!/usr/bin/env python
"""Benchmark: Dense coupling execution time measurement.

This script measures execution time for fully-connected (dense) Ising problems
across various problem sizes (N). Use this to evaluate GPU acceleration
effectiveness for dense interaction graphs.

Usage:
    python benchmarks/bench_dense.py
    python benchmarks/bench_dense.py --sizes 100 500 1000 2000 --num-reads 10
    python benchmarks/bench_dense.py --output results_dense.csv
"""

import argparse
import sys
import time
from typing import Any

import numpy as np

if __package__:
    from ._benchmark_utils import (
        add_standard_sampling_args,
        ensure_project_root_on_path,
        save_rows_to_csv,
    )
else:
    from _benchmark_utils import (
        add_standard_sampling_args,
        ensure_project_root_on_path,
        save_rows_to_csv,
    )

ensure_project_root_on_path(__file__)

from benchmarks.problem_generators import generate_dense_problem_array
from digitalannealer import DASampler


def run_benchmark_dense(
    sizes: list[int],
    num_sweeps: int = 1000,
    num_reads: int = 1,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    seed: int = 42,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Run dense mode benchmarks across multiple problem sizes.

    Args:
        sizes: List of problem sizes (N) to benchmark.
        num_sweeps: Number of Monte Carlo sweeps per run.
        num_reads: Number of independent samples per run.
        beta_min: Minimum inverse temperature.
        beta_max: Maximum inverse temperature.
        seed: Random seed for reproducibility.
        warmup: Whether to run a warmup iteration before timing.

    Returns:
        List of result dictionaries containing benchmark data.
    """
    results = []
    sampler = DASampler(seed=seed)

    for n in sizes:
        print(f"Benchmarking N={n} (dense)...", end=" ", flush=True)

        # Generate problem
        h, J = generate_dense_problem_array(n, seed=seed)

        # Warmup run to ensure GPU is initialized
        if warmup and n == sizes[0]:
            try:
                _ = sampler.sample_ising(h, J, num_sweeps=10, num_reads=1, sparse=False)
            except Exception:
                # Ignore warmup failures: any real issues will surface in the timed run below.
                pass

        # Timed run
        try:
            start_time = time.perf_counter()
            result = sampler.sample_ising(
                h,
                J,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                beta_min=beta_min,
                beta_max=beta_max,
                sparse=False,
            )
            elapsed = time.perf_counter() - start_time

            best_energy = float(result.first.energy)
            avg_energy = float(np.mean(result.record.energy))

            results.append(
                {
                    "n_spins": n,
                    "mode": "dense",
                    "num_sweeps": num_sweeps,
                    "num_reads": num_reads,
                    "elapsed_sec": elapsed,
                    "sweeps_per_sec": num_sweeps * num_reads / elapsed,
                    "best_energy": best_energy,
                    "avg_energy": avg_energy,
                    "status": "success",
                }
            )
            print(f"done in {elapsed:.3f}s")

        except Exception as e:
            results.append(
                {
                    "n_spins": n,
                    "mode": "dense",
                    "num_sweeps": num_sweeps,
                    "num_reads": num_reads,
                    "elapsed_sec": None,
                    "sweeps_per_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"error: {e}",
                }
            )
            print(f"failed: {e}")

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("Dense Mode Benchmark Results")
    print("=" * 80)
    print(
        f"{'N':>8} {'Sweeps':>10} {'Reads':>8} {'Time (s)':>12} "
        f"{'Sweeps/s':>14} {'Best E':>12} {'Status':>10}"
    )
    print("-" * 80)

    for r in results:
        elapsed = f"{r['elapsed_sec']:.4f}" if r["elapsed_sec"] else "N/A"
        sps = f"{r['sweeps_per_sec']:.1f}" if r["sweeps_per_sec"] else "N/A"
        best_e = f"{r['best_energy']:.4f}" if r["best_energy"] is not None else "N/A"
        status = "OK" if r["status"] == "success" else "FAIL"

        print(
            f"{r['n_spins']:>8} {r['num_sweeps']:>10} {r['num_reads']:>8} "
            f"{elapsed:>12} {sps:>14} {best_e:>12} {status:>10}"
        )


def save_results_csv(results: list[dict[str, Any]], filepath: str) -> None:
    """Save benchmark results to a CSV file."""
    save_rows_to_csv(results, filepath)
    print(f"\nResults saved to {filepath}")


def main():
    """Run the dense benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark Digital Annealer dense mode performance"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 2000, 5000],
        help="Problem sizes (N) to benchmark (default: 100 200 500 1000 2000 5000)",
    )
    add_standard_sampling_args(parser)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup iteration",
    )

    args = parser.parse_args()

    print("Digital Annealer - Dense Mode Benchmark")
    print("=" * 80)
    print(f"Problem sizes: {args.sizes}")
    print(f"Sweeps: {args.num_sweeps}, Reads: {args.num_reads}")
    print(f"Beta range: [{args.beta_min}, {args.beta_max}]")
    print(f"Seed: {args.seed}")
    print()

    results = run_benchmark_dense(
        sizes=args.sizes,
        num_sweeps=args.num_sweeps,
        num_reads=args.num_reads,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        seed=args.seed,
        warmup=not args.no_warmup,
    )

    print_results(results)

    if args.output:
        save_results_csv(results, args.output)

    # Return exit code based on whether any tests failed
    has_failures = any(r["status"] != "success" for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
