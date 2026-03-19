#!/usr/bin/env python
"""Benchmark: float32 vs float64 comparison for DA and PTDA (dense & sparse).

Compares execution time and solution quality across:
  - DASampler  (dense / sparse) × (float32 / float64)
  - PTDASampler (dense / sparse) × (float32 / float64)

Energy verification uses OpenJij SASampler as a reference baseline.

Problem sizes and sweep counts (paired 1:1):
  N=100  → 1,000 sweeps
  N=200  → 5,000 sweeps
  N=500  → 10,000 sweeps
  N=1000 → 500,000 sweeps
  N=2000 → 1,000,000 sweeps

num_reads = 100 for all configurations.
OpenJij reference: num_sweeps=1000, num_reads=100.

Usage:
    python benchmarks/bench_dtype_comparison.py
    python benchmarks/bench_dtype_comparison.py --output results.csv
    python benchmarks/bench_dtype_comparison.py --skip-sparse
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

if __package__:
    from ._benchmark_utils import ensure_project_root_on_path, save_rows_to_csv
else:
    from _benchmark_utils import ensure_project_root_on_path, save_rows_to_csv

ROOT_DIR = ensure_project_root_on_path(__file__)

from benchmarks.problem_generators import generate_dense_problem_array
from digitalannealer import DASampler, PTDASampler

try:
    import openjij as oj

    OPENJIJ_AVAILABLE = True
except ImportError:
    OPENJIJ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEM_CONFIGS: list[tuple[int, int]] = [
    # (n_spins, num_sweeps)
    (100, 1_000),
    (200, 5_000),
    (500, 10_000),
    (1_000, 500_000),
    (2_000, 1_000_000),
]

NUM_READS = 100
OPENJIJ_NUM_SWEEPS = 1_000
OPENJIJ_NUM_READS = 100
SEED = 42
BETA_MIN = 0.1
BETA_MAX = 10.0

# PTDA-specific
NUM_REPLICAS = 10
EXCHANGE_INTERVAL = 10


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Single benchmark result row."""

    sampler: str  # "DA" | "PTDA" | "OpenJij"
    mode: str  # "dense" | "sparse"
    dtype: str  # "float32" | "float64" | "n/a"
    n_spins: int
    num_sweeps: int
    num_reads: int
    elapsed_sec: float
    best_energy: float
    mean_energy: float
    std_energy: float
    error: str = ""


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------


def generate_problem(
    n_spins: int, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, dict[int, float], dict[tuple[int, int], float]]:
    """Return (h_array, J_array, h_dict, J_dict) for a dense random Ising problem."""
    h_arr, J_arr = generate_dense_problem_array(n_spins, seed=seed)
    # Dict versions for OpenJij
    h_dict = {i: float(h_arr[i]) for i in range(n_spins)}
    J_dict = {}
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            v = float(J_arr[i, j])
            if v != 0.0:
                J_dict[(i, j)] = v
    return h_arr, J_arr, h_dict, J_dict


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_da(
    h: np.ndarray,
    J: np.ndarray,
    num_sweeps: int,
    num_reads: int,
    sparse: bool,
    dtype: type,
    seed: int = SEED,
) -> BenchResult:
    sampler = DASampler(seed=seed)
    mode = "sparse" if sparse else "dense"
    dtype_name = np.dtype(dtype).name

    start = time.perf_counter()
    result = sampler.sample_ising(
        h=h,
        J=J,
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        sparse=sparse,
        dtype=dtype,
    )
    elapsed = time.perf_counter() - start

    energies = result.record.energy
    return BenchResult(
        sampler="DA",
        mode=mode,
        dtype=dtype_name,
        n_spins=h.shape[0],
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        elapsed_sec=elapsed,
        best_energy=float(result.first.energy),
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
    )


def _run_ptda(
    h: np.ndarray,
    J: np.ndarray,
    num_sweeps: int,
    num_reads: int,
    sparse: bool,
    dtype: type,
    seed: int = SEED,
) -> BenchResult:
    sampler = PTDASampler(seed=seed)
    mode = "sparse" if sparse else "dense"
    dtype_name = np.dtype(dtype).name

    start = time.perf_counter()
    result = sampler.sample_ising(
        h=h,
        J=J,
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        num_replicas=NUM_REPLICAS,
        exchange_interval=EXCHANGE_INTERVAL,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        sparse=sparse,
        dtype=dtype,
    )
    elapsed = time.perf_counter() - start

    energies = result.record.energy
    return BenchResult(
        sampler="PTDA",
        mode=mode,
        dtype=dtype_name,
        n_spins=h.shape[0],
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        elapsed_sec=elapsed,
        best_energy=float(result.first.energy),
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
    )


def _run_openjij(
    h_dict: dict[int, float],
    J_dict: dict[tuple[int, int], float],
    n_spins: int,
    seed: int = SEED,
) -> BenchResult:
    sampler = oj.SASampler()
    start = time.perf_counter()
    result = sampler.sample_ising(
        h_dict,
        J_dict,
        num_sweeps=OPENJIJ_NUM_SWEEPS,
        num_reads=OPENJIJ_NUM_READS,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        seed=seed,
    )
    elapsed = time.perf_counter() - start

    energies = result.record.energy
    return BenchResult(
        sampler="OpenJij",
        mode="n/a",
        dtype="n/a",
        n_spins=n_spins,
        num_sweeps=OPENJIJ_NUM_SWEEPS,
        num_reads=OPENJIJ_NUM_READS,
        elapsed_sec=elapsed,
        best_energy=float(result.first.energy),
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
    )


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def _header() -> str:
    return (
        f"{'Sampler':<8} {'Mode':<7} {'dtype':<8} {'N':>5} {'Sweeps':>9} "
        f"{'Time(s)':>9} {'BestE':>12} {'MeanE':>12} {'StdE':>10} {'Note'}"
    )


def _row(r: BenchResult) -> str:
    note = r.error if r.error else ""
    return (
        f"{r.sampler:<8} {r.mode:<7} {r.dtype:<8} {r.n_spins:>5} {r.num_sweeps:>9} "
        f"{r.elapsed_sec:>9.3f} {r.best_energy:>12.2f} {r.mean_energy:>12.2f} "
        f"{r.std_energy:>10.2f} {note}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks(
    skip_sparse: bool = False,
    output_csv: str | None = None,
) -> list[BenchResult]:
    """Run all benchmark configurations and return results."""
    all_results: list[BenchResult] = []

    # Print header
    separator = "=" * 110
    print(separator)
    print("Float32 vs Float64 Benchmark — DA / PTDA × Dense / Sparse")
    print(separator)
    print(f"  num_reads      = {NUM_READS}")
    print(f"  beta_min/max   = {BETA_MIN} / {BETA_MAX}")
    print(f"  PTDA replicas  = {NUM_REPLICAS}, exchange_interval = {EXCHANGE_INTERVAL}")
    print(f"  OpenJij sweeps = {OPENJIJ_NUM_SWEEPS}, reads = {OPENJIJ_NUM_READS}")
    print(f"  seed           = {SEED}")
    print(separator)
    print()

    # Warmup GPU (small problem)
    print("Warming up GPU...", flush=True)
    _warmup_sampler = DASampler(seed=0)
    _h_warm, _J_warm = generate_dense_problem_array(10, seed=0)
    try:
        _warmup_sampler.sample_ising(_h_warm, _J_warm, num_sweeps=10, num_reads=1)
    except Exception:
        pass
    print()

    for n_spins, num_sweeps in PROBLEM_CONFIGS:
        print(f"{'─' * 110}")
        print(f"  Problem: N={n_spins}, num_sweeps={num_sweeps:,}")
        print(f"{'─' * 110}")
        print(_header())

        # Generate problem once per size
        h_arr, J_arr, h_dict, J_dict = generate_problem(n_spins)

        # --- OpenJij reference ---
        if OPENJIJ_AVAILABLE:
            try:
                res = _run_openjij(h_dict, J_dict, n_spins)
                all_results.append(res)
                print(_row(res))
            except Exception as e:
                res = BenchResult(
                    sampler="OpenJij", mode="n/a", dtype="n/a",
                    n_spins=n_spins, num_sweeps=OPENJIJ_NUM_SWEEPS,
                    num_reads=OPENJIJ_NUM_READS, elapsed_sec=0,
                    best_energy=float("nan"), mean_energy=float("nan"),
                    std_energy=float("nan"), error=str(e),
                )
                all_results.append(res)
                print(_row(res))
        else:
            print("  [OpenJij not installed — skipping reference]")

        # --- DA configurations ---
        for dtype in [np.float32, np.float64]:
            # Dense
            try:
                res = _run_da(h_arr, J_arr, num_sweeps, NUM_READS, sparse=False, dtype=dtype)
                all_results.append(res)
                print(_row(res))
            except Exception as e:
                res = BenchResult(
                    sampler="DA", mode="dense", dtype=np.dtype(dtype).name,
                    n_spins=n_spins, num_sweeps=num_sweeps, num_reads=NUM_READS,
                    elapsed_sec=0, best_energy=float("nan"),
                    mean_energy=float("nan"), std_energy=float("nan"),
                    error=str(e),
                )
                all_results.append(res)
                print(_row(res))

            # Sparse
            if not skip_sparse:
                try:
                    res = _run_da(h_arr, J_arr, num_sweeps, NUM_READS, sparse=True, dtype=dtype)
                    all_results.append(res)
                    print(_row(res))
                except Exception as e:
                    res = BenchResult(
                        sampler="DA", mode="sparse", dtype=np.dtype(dtype).name,
                        n_spins=n_spins, num_sweeps=num_sweeps, num_reads=NUM_READS,
                        elapsed_sec=0, best_energy=float("nan"),
                        mean_energy=float("nan"), std_energy=float("nan"),
                        error=str(e),
                    )
                    all_results.append(res)
                    print(_row(res))

        # --- PTDA configurations ---
        for dtype in [np.float32, np.float64]:
            # Dense
            try:
                res = _run_ptda(h_arr, J_arr, num_sweeps, NUM_READS, sparse=False, dtype=dtype)
                all_results.append(res)
                print(_row(res))
            except Exception as e:
                res = BenchResult(
                    sampler="PTDA", mode="dense", dtype=np.dtype(dtype).name,
                    n_spins=n_spins, num_sweeps=num_sweeps, num_reads=NUM_READS,
                    elapsed_sec=0, best_energy=float("nan"),
                    mean_energy=float("nan"), std_energy=float("nan"),
                    error=str(e),
                )
                all_results.append(res)
                print(_row(res))

            # Sparse
            if not skip_sparse:
                try:
                    res = _run_ptda(h_arr, J_arr, num_sweeps, NUM_READS, sparse=True, dtype=dtype)
                    all_results.append(res)
                    print(_row(res))
                except Exception as e:
                    res = BenchResult(
                        sampler="PTDA", mode="sparse", dtype=np.dtype(dtype).name,
                        n_spins=n_spins, num_sweeps=num_sweeps, num_reads=NUM_READS,
                        elapsed_sec=0, best_energy=float("nan"),
                        mean_energy=float("nan"), std_energy=float("nan"),
                        error=str(e),
                    )
                    all_results.append(res)
                    print(_row(res))

        print()

    # --- Summary table ---
    print(separator)
    print("SUMMARY")
    print(separator)
    print(_header())
    for r in all_results:
        print(_row(r))
    print(separator)

    # --- CSV output ---
    if output_csv:
        _write_csv(all_results, output_csv)
        print(f"\nResults written to {output_csv}")

    return all_results


def _write_csv(results: list[BenchResult], path: str) -> None:
    """Write results to CSV."""
    fieldnames = [
        "sampler", "mode", "dtype", "n_spins", "num_sweeps", "num_reads",
        "elapsed_sec", "best_energy", "mean_energy", "std_energy", "error",
    ]
    rows = []
    for r in results:
        rows.append(
            {
                "sampler": r.sampler,
                "mode": r.mode,
                "dtype": r.dtype,
                "n_spins": r.n_spins,
                "num_sweeps": r.num_sweeps,
                "num_reads": r.num_reads,
                "elapsed_sec": f"{r.elapsed_sec:.4f}",
                "best_energy": f"{r.best_energy:.6f}",
                "mean_energy": f"{r.mean_energy:.6f}",
                "std_energy": f"{r.std_energy:.6f}",
                "error": r.error,
            }
        )
    save_rows_to_csv(rows, path, fieldnames=fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Float32 vs Float64 benchmark for DA and PTDA (dense & sparse).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write CSV results (default: print only).",
    )
    parser.add_argument(
        "--skip-sparse",
        action="store_true",
        help="Skip sparse mode benchmarks (useful for large N where sparse SMR may fail).",
    )
    args = parser.parse_args()

    run_benchmarks(
        skip_sparse=args.skip_sparse,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
