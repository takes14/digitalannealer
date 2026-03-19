#!/usr/bin/env python
"""Benchmark: Sparse coupling execution time measurement.

This script measures execution time for sparse Ising problems across various
problem sizes (N). Use this to evaluate GPU acceleration effectiveness for
sparse interaction graphs (e.g., lattice structures, random sparse graphs).

Usage:
    python benchmarks/bench_sparse.py
    python benchmarks/bench_sparse.py --sizes 100 500 1000 2000 --connectivity 4
    python benchmarks/bench_sparse.py --topology ring --output results_sparse.csv
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

from benchmarks.problem_generators import (
    generate_sparse_problem_2d_lattice,
    generate_sparse_problem_random,
    generate_sparse_problem_ring,
)
from digitalannealer import DASampler


def get_problem_generator(topology: str):
    """Get the appropriate problem generator for the topology."""
    generators = {
        "ring": generate_sparse_problem_ring,
        "lattice": generate_sparse_problem_2d_lattice,
        "random": generate_sparse_problem_random,
    }
    if topology not in generators:
        raise ValueError(f"Unknown topology: {topology}. Choose from {list(generators.keys())}")
    return generators[topology]


def run_benchmark_sparse(
    sizes: list[int],
    topology: str = "ring",
    connectivity: int = 4,
    num_sweeps: int = 1000,
    num_reads: int = 1,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    seed: int = 42,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Run sparse mode benchmarks across multiple problem sizes.

    Args:
        sizes: List of problem sizes (N) to benchmark.
        topology: Graph topology ("ring", "lattice", "random").
        connectivity: Average connectivity for random topology.
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
        print(f"Benchmarking N={n} (sparse, {topology})...", end=" ", flush=True)

        # Initialize variables to avoid potential NameError in exception handler
        n_edges = None

        # Generate problem based on topology
        try:
            if topology == "lattice":
                # For lattice, n is the side length
                n_side = int(np.sqrt(n))
                actual_n = n_side * n_side
                h, J = generate_sparse_problem_2d_lattice(n_side, seed=seed)
            elif topology == "random":
                actual_n = n
                h, J = generate_sparse_problem_random(n, connectivity=connectivity, seed=seed)
            else:  # ring
                actual_n = n
                h, J = generate_sparse_problem_ring(n, seed=seed)

            n_edges = len(J)
        except Exception as e:
            results.append(
                {
                    "n_spins": n,
                    "mode": "sparse",
                    "topology": topology,
                    "n_edges": None,
                    "num_sweeps": num_sweeps,
                    "num_reads": num_reads,
                    "elapsed_sec": None,
                    "sweeps_per_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"generation error: {e}",
                }
            )
            print(f"failed to generate: {e}")
            continue

        # Warmup run to ensure GPU is initialized
        if warmup and n == sizes[0]:
            try:
                small_h = {0: 0.0, 1: 0.0}
                small_J = {(0, 1): -1.0}
                _ = sampler.sample_ising(small_h, small_J, num_sweeps=10, num_reads=1, sparse=True)
            except Exception:
                # Warmup is non-critical; ignore any failures so that main benchmarks still run.
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
                sparse=True,
            )
            elapsed = time.perf_counter() - start_time

            best_energy = float(result.first.energy)
            avg_energy = float(np.mean(result.record.energy))

            results.append(
                {
                    "n_spins": actual_n,
                    "mode": "sparse",
                    "topology": topology,
                    "n_edges": n_edges,
                    "num_sweeps": num_sweeps,
                    "num_reads": num_reads,
                    "elapsed_sec": elapsed,
                    "sweeps_per_sec": num_sweeps * num_reads / elapsed,
                    "best_energy": best_energy,
                    "avg_energy": avg_energy,
                    "status": "success",
                }
            )
            print(f"done in {elapsed:.3f}s (edges={n_edges})")

        except Exception as e:
            results.append(
                {
                    "n_spins": actual_n,
                    "mode": "sparse",
                    "topology": topology,
                    "n_edges": n_edges,
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
    print("\n" + "=" * 90)
    print("Sparse Mode Benchmark Results")
    print("=" * 90)
    print(
        f"{'N':>8} {'Edges':>8} {'Topology':>10} {'Sweeps':>8} "
        f"{'Time (s)':>12} {'Sweeps/s':>12} {'Best E':>12} {'Status':>8}"
    )
    print("-" * 90)

    for r in results:
        elapsed = f"{r['elapsed_sec']:.4f}" if r["elapsed_sec"] else "N/A"
        sps = f"{r['sweeps_per_sec']:.1f}" if r["sweeps_per_sec"] else "N/A"
        best_e = f"{r['best_energy']:.4f}" if r["best_energy"] is not None else "N/A"
        edges = f"{r['n_edges']}" if r["n_edges"] is not None else "N/A"
        status = "OK" if r["status"] == "success" else "FAIL"

        print(
            f"{r['n_spins']:>8} {edges:>8} {r['topology']:>10} {r['num_sweeps']:>8} "
            f"{elapsed:>12} {sps:>12} {best_e:>12} {status:>8}"
        )


def save_results_csv(results: list[dict[str, Any]], filepath: str) -> None:
    """Save benchmark results to a CSV file."""
    save_rows_to_csv(results, filepath)
    print(f"\nResults saved to {filepath}")


def main():
    """Run the sparse benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark Digital Annealer sparse mode performance"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 400, 900, 1600, 2500, 10000],
        help="Problem sizes (N) to benchmark (default: 100 400 900 1600 2500 10000)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["ring", "lattice", "random"],
        default="ring",
        help="Graph topology (default: ring)",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=4,
        help="Average connectivity for random topology (default: 4)",
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

    print("Digital Annealer - Sparse Mode Benchmark")
    print("=" * 90)
    print(f"Problem sizes: {args.sizes}")
    print(f"Topology: {args.topology}")
    if args.topology == "random":
        print(f"Connectivity: {args.connectivity}")
    print(f"Sweeps: {args.num_sweeps}, Reads: {args.num_reads}")
    print(f"Beta range: [{args.beta_min}, {args.beta_max}]")
    print(f"Seed: {args.seed}")
    print()

    results = run_benchmark_sparse(
        sizes=args.sizes,
        topology=args.topology,
        connectivity=args.connectivity,
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
