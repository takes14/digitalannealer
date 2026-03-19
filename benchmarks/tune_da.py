#!/usr/bin/env python
"""DA parameter tuning: find optimal num_sweeps per problem size.

For each problem size, runs DA with a geometric sweep sequence and plots
energy + elapsed time vs num_sweeps.  A grey horizontal line marks the
OpenJij SASampler baseline energy for reference.

Outputs (per size):
    benchmarks/outputs/tuning/da_sweep_n{N}.csv
    benchmarks/outputs/tuning/da_sweep_n{N}.png

Usage:
    python benchmarks/tune_da.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from ._benchmark_utils import (
        build_geometric_sweeps,
        ensure_project_root_on_path,
        save_rows_to_csv,
    )
else:
    from _benchmark_utils import (
        build_geometric_sweeps,
        ensure_project_root_on_path,
        save_rows_to_csv,
    )

ROOT_DIR = ensure_project_root_on_path(__file__)

from benchmarks.problem_generators import generate_dense_problem_dict
from digitalannealer import DASampler

# Try to import OpenJij
try:
    import openjij as oj

    OPENJIJ_AVAILABLE = True
except ImportError:
    OPENJIJ_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
# Configuration – edit HERE to change all experiments at once
# ──────────────────────────────────────────────────────────────────────

PROBLEM_SIZES: list[int] = [100, 200, 500, 1000, 2000, 8000]

SWEEPS_START: int = 100  # lower bound (same for every size)
SWEEPS_STOPS: dict[int, int] = {  # upper bound per size
    100: 5_000,
    200: 10_000,
    500: 20_000,
    1000: 50_000,
    2000: 100_000,
    8000: 200_000,
}
SWEEPS_MULTIPLIER: float = 1.5

NUM_READS: int = 100
BETA_MIN: float = 0.1
BETA_MAX: float = 10.0
OFFSET_DELTA: float = 0.1
SEED: int = 42

OJ_NUM_SWEEPS: int = 1000  # fixed sweep count for OpenJij baseline

OUTPUT_DIR: str = "benchmarks/outputs/tuning"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def warmup_da() -> None:
    """Best-effort GPU context init."""
    try:
        sampler = DASampler(seed=SEED)
        sampler.sample_ising(
            h={0: 0.0, 1: 0.0},
            J={(0, 1): -1.0},
            num_sweeps=10,
            num_reads=1,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
        )
    except Exception:
        pass


def get_openjij_baseline(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> tuple[float | None, float | None, float | None]:
    """Return (best_energy, avg_energy, elapsed) from OpenJij SASampler."""
    if not OPENJIJ_AVAILABLE:
        print("  [WARN] openjij not installed – skipping baseline")
        return None, None, None
    try:
        sampler = oj.SASampler()
        start = time.perf_counter()
        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=OJ_NUM_SWEEPS,
            num_reads=NUM_READS,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            seed=SEED,
        )
        elapsed = time.perf_counter() - start
        best_energy = float(result.first.energy)
        avg_energy = float(np.mean(result.record.energy))
        return best_energy, avg_energy, elapsed
    except Exception as e:
        print(f"  [WARN] openjij failed: {e}")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────


def run_sweep_study(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    sweeps_list: list[int],
) -> list[dict[str, Any]]:
    """Run DA across *sweeps_list* and collect results."""
    sampler = DASampler(seed=SEED)
    results: list[dict[str, Any]] = []

    for num_sweeps in sweeps_list:
        try:
            start = time.perf_counter()
            result = sampler.sample_ising(
                h=h,
                J=J,
                num_sweeps=num_sweeps,
                num_reads=NUM_READS,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX,
                offset_delta=OFFSET_DELTA,
            )
            elapsed = time.perf_counter() - start

            best_energy = float(result.first.energy)
            avg_energy = float(np.mean(result.record.energy))

            results.append(
                {
                    "num_sweeps": num_sweeps,
                    "num_reads": NUM_READS,
                    "elapsed_sec": elapsed,
                    "best_energy": best_energy,
                    "avg_energy": avg_energy,
                    "status": "success",
                }
            )
            print(
                f"  sweeps={num_sweeps:>8}  E_best={best_energy:>12.4f}  "
                f"E_avg={avg_energy:>12.4f}  time={elapsed:.3f}s"
            )
        except Exception as e:
            results.append(
                {
                    "num_sweeps": num_sweeps,
                    "num_reads": NUM_READS,
                    "elapsed_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"error: {e}",
                }
            )
            print(f"  sweeps={num_sweeps:>8}  FAILED: {e}")

    return results


def plot_sweep_study(
    n_spins: int,
    results: list[dict[str, Any]],
    oj_best_energy: float | None,
    oj_elapsed_sec: float | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        return

    sweeps = [r["num_sweeps"] for r in ok]
    best_e = [r["best_energy"] for r in ok]
    elapsed = [r["elapsed_sec"] for r in ok]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Energy (left axis)
    ax.plot(sweeps, best_e, marker="o", color="#1f77b4", label="Best Energy")
    ax.set_xlabel("num_sweeps")
    ax.set_ylabel("Energy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, linestyle="--", alpha=0.4)

    # OpenJij baseline
    if oj_best_energy is not None:
        if oj_elapsed_sec is not None:
            oj_label = f"OpenJij baseline (E={oj_best_energy:.2f}, t={oj_elapsed_sec:.3f}s)"
        else:
            oj_label = f"OpenJij baseline (E={oj_best_energy:.2f})"
        ax.axhline(
            y=oj_best_energy,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=oj_label,
        )

    # Time (right axis)
    ax_time = ax.twinx()
    ax_time.plot(sweeps, elapsed, marker="s", color="#ff7f0e", label="Elapsed (s)")
    ax_time.set_ylabel("Elapsed time (s)", color="#ff7f0e")
    ax_time.tick_params(axis="y", labelcolor="#ff7f0e")

    # Combined legend
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_time.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax.set_title(f"DA sweep tuning: N={n_spins}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DA Sweep Tuning Benchmark")
    print("=" * 80)
    print(f"Problem sizes : {PROBLEM_SIZES}")
    print(f"Sweep range   : {SWEEPS_START} → (size-dependent stop) × {SWEEPS_MULTIPLIER}")
    print(f"num_reads={NUM_READS}  beta=[{BETA_MIN}, {BETA_MAX}]  seed={SEED}")
    print(f"OpenJij sweeps: {OJ_NUM_SWEEPS}")
    print(f"Output dir    : {output_dir}")
    print()

    # GPU warmup
    print("Warming up GPU …")
    warmup_da()

    for n in PROBLEM_SIZES:
        stop = SWEEPS_STOPS.get(n, 20_000)
        sweeps_list = build_geometric_sweeps(SWEEPS_START, stop, SWEEPS_MULTIPLIER)

        print(f"\n{'─'*60}")
        print(f"N={n}  sweeps={sweeps_list}")
        print(f"{'─'*60}")

        h, J = generate_dense_problem_dict(n, seed=SEED)

        # OpenJij baseline (once per size)
        print("  OpenJij baseline …")
        oj_best, oj_avg, oj_time = get_openjij_baseline(h, J)
        if oj_best is not None:
            print(
                f"  OpenJij  E_best={oj_best:>12.4f}  E_avg={oj_avg:>12.4f}  "
                f"time={oj_time:.3f}s"
            )

        # DA sweep study
        results = run_sweep_study(h, J, sweeps_list)

        # Save CSV
        csv_path = output_dir / f"da_sweep_n{n}.csv"
        save_rows_to_csv(results, csv_path, create_parent=True)
        print(f"  → {csv_path}")

        # Save plot
        png_path = output_dir / f"da_sweep_n{n}.png"
        plot_sweep_study(n, results, oj_best, oj_time, png_path)
        print(f"  → {png_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
