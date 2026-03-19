#!/usr/bin/env python
"""Benchmark: Speed comparison with OpenJij SASampler.

This script compares the performance of Digital Annealer with OpenJij's
SASampler (CPU-based Simulated Annealing) to measure GPU acceleration
effectiveness.

Requirements:
    pip install openjij

Usage:
    python benchmarks/bench_compare_openjij.py
    python benchmarks/bench_compare_openjij.py --sizes 100 500 1000 2000
    python benchmarks/bench_compare_openjij.py --output comparison.csv
"""

import argparse
import sys
import time
from pathlib import Path
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

ROOT_DIR = ensure_project_root_on_path(__file__)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "benchmarks" / "outputs"

from benchmarks.problem_generators import (
    generate_dense_problem_dict,
    generate_sparse_problem_ring,
)
from digitalannealer import DASampler, PTDASampler

# Try to import OpenJij
try:
    import openjij as oj

    OPENJIJ_AVAILABLE = True
except ImportError:
    OPENJIJ_AVAILABLE = False


def benchmark_digitalannealer(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    num_sweeps: int,
    num_reads: int,
    beta_min: float,
    beta_max: float,
    seed: int,
    sparse: bool = False,
) -> tuple[float, float, str]:
    """Run Digital Annealer benchmark.

    Returns:
        Tuple of (elapsed_time, best_energy, status)
    """
    try:
        sampler = DASampler(seed=seed)
        start_time = time.perf_counter()
        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            sparse=sparse,
        )
        elapsed = time.perf_counter() - start_time
        best_energy = float(result.first.energy)
        return elapsed, best_energy, "success"
    except Exception as e:
        return None, None, f"error: {e}"


def benchmark_ptda(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    num_sweeps: int,
    num_reads: int,
    beta_min: float,
    beta_max: float,
    seed: int,
    num_replicas: int,
    exchange_interval: int,
) -> tuple[float, float, str]:
    """Run Parallel Tempering Digital Annealer benchmark.

    Returns:
        Tuple of (elapsed_time, best_energy, status)
    """
    try:
        sampler = PTDASampler(seed=seed)
        start_time = time.perf_counter()
        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            exchange_interval=exchange_interval,
        )
        elapsed = time.perf_counter() - start_time
        best_energy = float(result.first.energy)
        return elapsed, best_energy, "success"
    except Exception as e:
        return None, None, f"error: {e}"


def benchmark_openjij(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    num_sweeps: int,
    num_reads: int,
    beta_min: float,
    beta_max: float,
    seed: int,
) -> tuple[float, float, str]:
    """Run OpenJij SASampler benchmark.

    Returns:
        Tuple of (elapsed_time, best_energy, status)
    """
    if not OPENJIJ_AVAILABLE:
        return None, None, "openjij not installed"

    try:
        sampler = oj.SASampler()
        start_time = time.perf_counter()
        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
        )
        elapsed = time.perf_counter() - start_time
        best_energy = float(result.first.energy)
        return elapsed, best_energy, "success"
    except Exception as e:
        return None, None, f"error: {e}"


def run_comparison_benchmark(
    sizes: list[int],
    problem_type: str = "dense",
    num_sweeps: int = 1000,
    num_reads: int = 1,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    seed: int = 42,
    warmup: bool = True,
    num_replicas: int = 10,
    exchange_interval: int = 10,
) -> list[dict[str, Any]]:
    """Run comparison benchmarks between Digital Annealer and OpenJij.

    Args:
        sizes: List of problem sizes (N) to benchmark.
        problem_type: Either "dense" or "sparse".
        num_sweeps: Number of Monte Carlo sweeps per run.
        num_reads: Number of independent samples per run.
        beta_min: Minimum inverse temperature.
        beta_max: Maximum inverse temperature.
        seed: Random seed for reproducibility.
        warmup: Whether to run warmup iterations.

    Returns:
        List of result dictionaries containing benchmark data.
    """
    results = []

    # Select problem generator
    generate_problem = (
        generate_dense_problem_dict if problem_type == "dense" else generate_sparse_problem_ring
    )

    # Warmup runs
    if warmup:
        print("Running warmup...")
        h_warmup, J_warmup = generate_problem(10, seed=seed)

        # Warmup Digital Annealer
        try:
            benchmark_digitalannealer(
                h_warmup,
                J_warmup,
                num_sweeps=10,
                num_reads=1,
                beta_min=beta_min,
                beta_max=beta_max,
                seed=seed,
                sparse=(problem_type == "sparse"),
            )
        except Exception:
            # Ignore warmup failures: any real issues will surface in the timed run below.
            pass

        # Warmup PT Digital Annealer (dense only)
        if problem_type == "dense":
            try:
                benchmark_ptda(
                    h_warmup,
                    J_warmup,
                    num_sweeps=10,
                    num_reads=1,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    seed=seed,
                    num_replicas=num_replicas,
                    exchange_interval=exchange_interval,
                )
            except Exception:
                # Ignore warmup failures: any real issues will surface in the timed run below.
                pass

        # Warmup OpenJij
        if OPENJIJ_AVAILABLE:
            try:
                benchmark_openjij(
                    h_warmup,
                    J_warmup,
                    num_sweeps=10,
                    num_reads=1,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    seed=seed,
                )
            except Exception:
                # Ignore warmup failures: any real issues will surface in the timed run below.
                pass

    for n in sizes:
        print(f"\nBenchmarking N={n} ({problem_type})...")

        # Generate problem
        h, J = generate_problem(n, seed=seed)

        # Benchmark Digital Annealer
        print("  Digital Annealer...", end=" ", flush=True)
        da_time, da_energy, da_status = benchmark_digitalannealer(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
            sparse=(problem_type == "sparse"),
        )
        if da_status == "success":
            print(f"{da_time:.4f}s, E={da_energy:.4f}")
        else:
            print(f"failed: {da_status}")

        # Benchmark PT Digital Annealer (dense only)
        if problem_type == "dense":
            print("  PT Digital Annealer...", end=" ", flush=True)
            ptda_time, ptda_energy, ptda_status = benchmark_ptda(
                h,
                J,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                beta_min=beta_min,
                beta_max=beta_max,
                seed=seed,
                num_replicas=num_replicas,
                exchange_interval=exchange_interval,
            )
            if ptda_status == "success":
                print(f"{ptda_time:.4f}s, E={ptda_energy:.4f}")
            else:
                print(f"failed: {ptda_status}")
        else:
            ptda_time, ptda_energy, ptda_status = None, None, "ptda dense-only"
            print("  PT Digital Annealer... skipped (dense only)")

        # Benchmark OpenJij
        print("  OpenJij SASampler...", end=" ", flush=True)
        oj_time, oj_energy, oj_status = benchmark_openjij(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
        )
        if oj_status == "success":
            print(f"{oj_time:.4f}s, E={oj_energy:.4f}")
        else:
            print(f"failed: {oj_status}")

        # Calculate speedup
        speedup_da = None
        speedup_ptda = None
        if (
            da_time is not None
            and oj_time is not None
            and da_time > 0
            and oj_time > 0
        ):
            speedup_da = oj_time / da_time
        if (
            ptda_time is not None
            and oj_time is not None
            and ptda_time > 0
            and oj_time > 0
        ):
            speedup_ptda = oj_time / ptda_time

        results.append(
            {
                "n_spins": n,
                "problem_type": problem_type,
                "num_sweeps": num_sweeps,
                "num_reads": num_reads,
                "da_time_sec": da_time,
                "da_energy": da_energy,
                "da_status": da_status,
                "ptda_time_sec": ptda_time,
                "ptda_energy": ptda_energy,
                "ptda_status": ptda_status,
                "oj_time_sec": oj_time,
                "oj_energy": oj_energy,
                "oj_status": oj_status,
                "speedup_da": speedup_da,
                "speedup_ptda": speedup_ptda,
                "num_replicas": num_replicas,
                "exchange_interval": exchange_interval,
            }
        )

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 100)
    print("Digital Annealer / PTDA vs OpenJij SASampler Comparison")
    print("=" * 100)
    print(
        f"{'N':>8} {'Type':>8} {'DA Time':>12} {'PTDA Time':>12} {'OJ Time':>12} "
        f"{'DA Spd':>10} {'PT Spd':>10} {'DA Energy':>12} {'PTDA Energy':>12} "
        f"{'OJ Energy':>12}"
    )
    print("-" * 100)

    for r in results:
        da_time_val = r.get("da_time_sec")
        da_time = f"{da_time_val:.4f}" if da_time_val is not None else "N/A"
        ptda_time_val = r.get("ptda_time_sec")
        ptda_time = f"{ptda_time_val:.4f}" if ptda_time_val is not None else "N/A"
        oj_time_val = r.get("oj_time_sec")
        oj_time = f"{oj_time_val:.4f}" if oj_time_val is not None else "N/A"
        speedup_da = f"{r['speedup_da']:.2f}x" if r.get("speedup_da") else "N/A"
        speedup_ptda = f"{r['speedup_ptda']:.2f}x" if r.get("speedup_ptda") else "N/A"
        da_energy = f"{r['da_energy']:.4f}" if r["da_energy"] is not None else "N/A"
        ptda_energy = (
            f"{r['ptda_energy']:.4f}" if r.get("ptda_energy") is not None else "N/A"
        )
        oj_energy = f"{r['oj_energy']:.4f}" if r["oj_energy"] is not None else "N/A"

        print(
            f"{r['n_spins']:>8} {r['problem_type']:>8} {da_time:>12} {ptda_time:>12} "
            f"{oj_time:>12} {speedup_da:>10} {speedup_ptda:>10} {da_energy:>12} "
            f"{ptda_energy:>12} {oj_energy:>12}"
        )

    # Summary statistics
    speedups_da = [r["speedup_da"] for r in results if r.get("speedup_da") is not None]
    speedups_pt = [
        r["speedup_ptda"] for r in results if r.get("speedup_ptda") is not None
    ]
    if speedups_da:
        print("-" * 100)
        print(f"Average speedup (DA): {np.mean(speedups_da):.2f}x")
        print(f"Max speedup (DA): {np.max(speedups_da):.2f}x")
        print(f"Min speedup (DA): {np.min(speedups_da):.2f}x")
    if speedups_pt:
        print("-" * 100)
        print(f"Average speedup (PTDA): {np.mean(speedups_pt):.2f}x")
        print(f"Max speedup (PTDA): {np.max(speedups_pt):.2f}x")
        print(f"Min speedup (PTDA): {np.min(speedups_pt):.2f}x")


def save_reads_plot(results: list[dict[str, Any]], filepath: Path) -> None:
    """Plot num_reads vs elapsed time for DA/PTDA/OpenJij."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    try:
        reads = [r["num_reads"] for r in results]
        series = [
            ("da_time_sec", "DA (DASampler)", "C0", "o"),
            ("ptda_time_sec", "PTDA (PTDASampler)", "C1", "s"),
            ("oj_time_sec", "OpenJij (SASampler)", "C2", "^"),
        ]

        for key, label, color, marker in series:
            times = [r.get(key) for r in results]
            if not any(t is not None for t in times):
                continue
            times_plottable = [t if t is not None else np.nan for t in times]
            plt.plot(reads, times_plottable, marker=marker, color=color, label=label)

        plt.xlabel("Number of samples (num_reads)")
        plt.ylabel("Time (sec)")
        plt.title("Sampling time vs num_reads")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved to {filepath}")
    except (OSError, RuntimeError) as exc:
        print(f"Failed to generate plot at {filepath}: {exc}")
        return


def run_reads_scaling_benchmark(
    reads: list[int],
    n_spins: int,
    num_sweeps: int = 1000,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    seed: int = 42,
    warmup: bool = True,
    num_replicas: int = 10,
    exchange_interval: int = 10,
) -> list[dict[str, Any]]:
    """Benchmark time scaling versus ``num_reads`` for DA/PTDA/OpenJij.

    This function measures how sampling time scales with the number of
    requested samples for:

    * Digital Annealer (``DASampler``)
    * Parallel Tempering Digital Annealer (``PTDASampler``)
    * OpenJij's CPU-based ``SASampler`` (if installed)

    A single dense Ising/QUBO instance of size ``n_spins`` is generated once
    and reused for all measurements to ensure comparability.

    Parameters
    ----------
    reads
        List of values for ``num_reads`` to benchmark. Each value is the number
        of samples requested from each sampler in a single run.
    n_spins
        Number of spins/variables in the dense Ising problem.
    num_sweeps
        Number of Monte Carlo sweeps per read for all samplers.
    beta_min
        Minimum inverse temperature (start of the annealing schedule).
    beta_max
        Maximum inverse temperature (end of the annealing schedule).
    seed
        Random seed used for problem generation and passed through to the
        samplers for reproducibility where supported.
    warmup
        If ``True``, run short warmup calls for each sampler before the main
        measurements to amortize one-time initialization costs (e.g., GPU
        context creation).
    num_replicas
        Number of replicas used for the PTDA benchmark
        (passed to ``benchmark_ptda``).
    exchange_interval
        Replica-exchange interval (in sweeps) used for the PTDA benchmark
        (passed to ``benchmark_ptda``).

    Returns
    -------
    list[dict[str, Any]]
        A list of result dictionaries, one per ``num_reads`` value. Each
        dictionary contains at least:

        * ``"num_reads"``: the corresponding number of requested samples.
        * ``"da_time_sec"``: elapsed wall-clock time for the DA run in seconds,
          or ``None`` if the DA benchmark failed.
        * ``"ptda_time_sec"``: elapsed wall-clock time for the PTDA run in
          seconds, or ``None`` if the PTDA benchmark failed.
        * ``"oj_time_sec"``: elapsed wall-clock time for the OpenJij SASampler
          run in seconds, or ``None`` if OpenJij is unavailable or the run
          failed.

        Additional metadata may be included for convenience.
    """
    results: list[dict[str, Any]] = []

    # Generate single dense problem instance to isolate num_reads scaling effects
    h, J = generate_dense_problem_dict(n_spins, seed=seed)

    if warmup:
        print("Running warmup...")
        try:
            benchmark_digitalannealer(
                h,
                J,
                num_sweeps=10,
                num_reads=1,
                beta_min=beta_min,
                beta_max=beta_max,
                seed=seed,
                sparse=False,
            )
        except Exception:
            # Ignore warmup failures: any real issues will surface in the timed run below.
            pass
        try:
            benchmark_ptda(
                h,
                J,
                num_sweeps=10,
                num_reads=1,
                beta_min=beta_min,
                beta_max=beta_max,
                seed=seed,
                num_replicas=num_replicas,
                exchange_interval=exchange_interval,
            )
        except Exception:
            # Ignore warmup failures: any real issues will surface in the timed run below.
            pass
        if OPENJIJ_AVAILABLE:
            try:
                benchmark_openjij(
                    h,
                    J,
                    num_sweeps=10,
                    num_reads=1,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    seed=seed,
                )
            except Exception:
                # Ignore warmup failures: any real issues will surface in the timed run below.
                pass

    for num_reads in reads:
        print(f"\nBenchmarking num_reads={num_reads} (dense)...")

        print("  Digital Annealer...", end=" ", flush=True)
        da_time, da_energy, da_status = benchmark_digitalannealer(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
            sparse=False,
        )
        if da_status == "success":
            print(f"{da_time:.4f}s, E={da_energy:.4f}")
        else:
            print(f"failed: {da_status}")

        print("  PT Digital Annealer...", end=" ", flush=True)
        ptda_time, ptda_energy, ptda_status = benchmark_ptda(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
            num_replicas=num_replicas,
            exchange_interval=exchange_interval,
        )
        if ptda_status == "success":
            print(f"{ptda_time:.4f}s, E={ptda_energy:.4f}")
        else:
            print(f"failed: {ptda_status}")

        print("  OpenJij SASampler...", end=" ", flush=True)
        oj_time, oj_energy, oj_status = benchmark_openjij(
            h,
            J,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=seed,
        )
        if oj_status == "success":
            print(f"{oj_time:.4f}s, E={oj_energy:.4f}")
        else:
            print(f"failed: {oj_status}")

        results.append(
            {
                "num_reads": num_reads,
                "n_spins": n_spins,
                "num_sweeps": num_sweeps,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "num_replicas": num_replicas,
                "exchange_interval": exchange_interval,
                "da_time_sec": da_time,
                "da_energy": da_energy,
                "da_status": da_status,
                "ptda_time_sec": ptda_time,
                "ptda_energy": ptda_energy,
                "ptda_status": ptda_status,
                "oj_time_sec": oj_time,
                "oj_energy": oj_energy,
                "oj_status": oj_status,
            }
        )

    return results


def print_reads_results(results: list[dict[str, Any]]) -> None:
    """Pretty print num_reads scaling results."""
    print("\n" + "=" * 100)
    print("Digital Annealer vs PTDA vs OpenJij (num_reads scaling)")
    print("=" * 100)
    print(
        f"{'Reads':>8} {'DA Time':>12} {'PTDA Time':>12} {'OJ Time':>12} "
        f"{'DA Energy':>12} {'PTDA Energy':>12} {'OJ Energy':>12}"
    )
    print("-" * 100)

    for r in results:
        reads_value = r.get("num_reads")
        reads_display = f"{reads_value}" if reads_value is not None else "N/A"
        da_time_val = r.get("da_time_sec")
        da_time = f"{da_time_val:.4f}" if da_time_val is not None else "N/A"
        ptda_time_val = r.get("ptda_time_sec")
        ptda_time = f"{ptda_time_val:.4f}" if ptda_time_val is not None else "N/A"
        oj_time_val = r.get("oj_time_sec")
        oj_time = f"{oj_time_val:.4f}" if oj_time_val is not None else "N/A"
        da_energy = f"{r['da_energy']:.4f}" if r["da_energy"] is not None else "N/A"
        ptda_energy = (
            f"{r['ptda_energy']:.4f}" if r["ptda_energy"] is not None else "N/A"
        )
        oj_energy = f"{r['oj_energy']:.4f}" if r["oj_energy"] is not None else "N/A"

        print(
            f"{reads_display:>8} {da_time:>12} {ptda_time:>12} {oj_time:>12} "
            f"{da_energy:>12} {ptda_energy:>12} {oj_energy:>12}"
        )


def save_results_csv(results: list[dict[str, Any]], filepath: str) -> None:
    """Save comparison results to a CSV file."""
    save_rows_to_csv(results, filepath)
    print(f"\nResults saved to {filepath}")


def main():
    """Run the comparison benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Compare Digital Annealer with OpenJij SASampler performance"
    )
    parser.add_argument(
        "--mode",
        choices=["sizes", "reads"],
        default="sizes",
        help="Benchmark by varying problem sizes or num_reads (default: sizes)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 2000],
        help="Problem sizes (N) to benchmark (default: 100 200 500 1000 2000)",
    )
    parser.add_argument(
        "--problem-size",
        type=int,
        default=200,
        help="Problem size (N) to use when --mode reads (default: 200)",
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["dense", "sparse"],
        default="dense",
        help="Problem type (default: dense)",
    )
    parser.add_argument(
        "--reads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="List of num_reads values when --mode reads (default: 1 2 4 8 16 32)",
    )
    add_standard_sampling_args(parser)
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=10,
        help="Number of PT replicas for PTDA (default: 10)",
    )
    parser.add_argument(
        "--exchange-interval",
        type=int,
        default=10,
        help="Replica exchange interval for PTDA (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output plot file path for --mode reads (optional)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup iteration",
    )

    args = parser.parse_args()

    print("Digital Annealer / PTDA vs OpenJij Comparison Benchmark")
    print("=" * 100)

    if not OPENJIJ_AVAILABLE:
        print("\nWARNING: OpenJij is not installed. Install with: pip install openjij")
        print("Only Digital Annealer benchmarks will be run.\n")

    if args.mode == "reads":
        print(f"Problem size: {args.problem_size}")
        print(f"Reads list: {args.reads}")
        print(f"Sweeps: {args.num_sweeps}")
        print(f"Beta range: [{args.beta_min}, {args.beta_max}]")
        print(f"Seed: {args.seed}")
        print(f"PT replicas: {args.num_replicas}, exchange_interval: {args.exchange_interval}")

        results = run_reads_scaling_benchmark(
            reads=args.reads,
            n_spins=args.problem_size,
            num_sweeps=args.num_sweeps,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            seed=args.seed,
            warmup=not args.no_warmup,
            num_replicas=args.num_replicas,
            exchange_interval=args.exchange_interval,
        )
        print_reads_results(results)

        outputs_dir = DEFAULT_OUTPUT_DIR
        csv_path = Path(args.output) if args.output else outputs_dir / "ptda_reads_scaling.csv"
        plot_path = Path(args.plot) if args.plot else outputs_dir / "ptda_reads_scaling.png"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(results, str(csv_path))
        save_reads_plot(results, plot_path)
    else:
        print(f"Problem sizes: {args.sizes}")
        print(f"Problem type: {args.problem_type}")
        print(f"Sweeps: {args.num_sweeps}, Reads: {args.num_reads}")
        print(f"Beta range: [{args.beta_min}, {args.beta_max}]")
        print(f"Seed: {args.seed}")
        print(f"PT replicas: {args.num_replicas}, exchange_interval: {args.exchange_interval}")

        results = run_comparison_benchmark(
            sizes=args.sizes,
            problem_type=args.problem_type,
            num_sweeps=args.num_sweeps,
            num_reads=args.num_reads,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            seed=args.seed,
            warmup=not args.no_warmup,
            num_replicas=args.num_replicas,
            exchange_interval=args.exchange_interval,
        )

        print_results(results)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            save_results_csv(results, args.output)

    sys.exit(0)


if __name__ == "__main__":
    main()
