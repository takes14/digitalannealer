#!/usr/bin/env python
"""PTDA parameter tuning: three studies per problem size.

Study 1 – num_sweeps   : geometric sweep sequence (same grid as DA tuning)
Study 2 – num_replicas : compare 8 / 16 / 32 replicas
Study 3 – exchange_interval : compare 10 / 50 / 100 / 1000

For every study × size, outputs a CSV and a dual-axis PNG (energy + time).
A grey dashed line marks the OpenJij SASampler baseline energy.

Outputs go to  benchmarks/outputs/tuning/ptda_*.{csv,png}

Usage:
    python benchmarks/tune_ptda.py
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
from digitalannealer import PTDASampler

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

# Study 1: num_sweeps tuning (geometric sequence)
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
STUDY1_NUM_REPLICAS: int = 10  # default
STUDY1_EXCHANGE_INTERVAL: int = 10  # default

# Study 2: num_replicas tuning
NUM_REPLICAS_LIST: list[int] = [8, 16, 32]
STUDY2_EXCHANGE_INTERVAL: int = 10  # default
# num_sweeps per size = SWEEPS_STOPS (large value to let replicas matter)

# Study 3: exchange_interval tuning
EXCHANGE_INTERVALS: list[int] = [10, 50, 100, 1000]
STUDY3_NUM_REPLICAS: int = 10  # default
# num_sweeps per size = SWEEPS_STOPS

# Common parameters
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


def warmup_ptda() -> None:
    """Best-effort GPU context init."""
    try:
        sampler = PTDASampler(seed=SEED)
        sampler.sample_ising(
            h={0: 0.0, 1: 0.0},
            J={(0, 1): -1.0},
            num_sweeps=10,
            num_reads=1,
            num_replicas=2,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            exchange_interval=5,
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
# Dual-axis plotting helper
# ──────────────────────────────────────────────────────────────────────


def _plot_dual_axis(
    *,
    title: str,
    xlabel: str,
    x_values: list[int | float],
    energies: list[float],
    times: list[float],
    oj_best_energy: float | None,
    oj_elapsed_sec: float | None,
    output_path: Path,
) -> None:
    """Save a dual-axis (energy + time) chart with an optional OJ baseline."""
    import matplotlib.pyplot as plt

    if not x_values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Energy (left axis)
    ax.plot(x_values, energies, marker="o", color="#1f77b4", label="Best Energy")
    ax.set_xlabel(xlabel)
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
    ax_time.plot(x_values, times, marker="s", color="#ff7f0e", label="Elapsed (s)")
    ax_time.set_ylabel("Elapsed time (s)", color="#ff7f0e")
    ax_time.tick_params(axis="y", labelcolor="#ff7f0e")

    # Combined legend
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_time.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Study 1 – num_sweeps
# ──────────────────────────────────────────────────────────────────────


def run_study_sweeps(
    n: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    oj_best: float | None,
    oj_time: float | None,
    output_dir: Path,
) -> None:
    stop = SWEEPS_STOPS.get(n, 20_000)
    sweeps_list = build_geometric_sweeps(SWEEPS_START, stop, SWEEPS_MULTIPLIER)
    print(f"  [Study 1] sweeps={sweeps_list}")

    sampler = PTDASampler(seed=SEED)
    rows: list[dict[str, Any]] = []

    for ns in sweeps_list:
        try:
            start = time.perf_counter()
            result = sampler.sample_ising(
                h=h,
                J=J,
                num_sweeps=ns,
                num_reads=NUM_READS,
                num_replicas=STUDY1_NUM_REPLICAS,
                exchange_interval=STUDY1_EXCHANGE_INTERVAL,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX,
                offset_delta=OFFSET_DELTA,
            )
            elapsed = time.perf_counter() - start
            best_e = float(result.first.energy)
            avg_e = float(np.mean(result.record.energy))
            rows.append(
                {
                    "num_sweeps": ns,
                    "num_replicas": STUDY1_NUM_REPLICAS,
                    "exchange_interval": STUDY1_EXCHANGE_INTERVAL,
                    "num_reads": NUM_READS,
                    "elapsed_sec": elapsed,
                    "best_energy": best_e,
                    "avg_energy": avg_e,
                    "status": "success",
                }
            )
            print(
                f"    sweeps={ns:>8}  E_best={best_e:>12.4f}  "
                f"E_avg={avg_e:>12.4f}  time={elapsed:.3f}s"
            )
        except Exception as e:
            rows.append(
                {
                    "num_sweeps": ns,
                    "num_replicas": STUDY1_NUM_REPLICAS,
                    "exchange_interval": STUDY1_EXCHANGE_INTERVAL,
                    "num_reads": NUM_READS,
                    "elapsed_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"error: {e}",
                }
            )
            print(f"    sweeps={ns:>8}  FAILED: {e}")

    # CSV
    csv_path = output_dir / f"ptda_sweep_n{n}.csv"
    save_rows_to_csv(rows, csv_path, create_parent=True)
    print(f"  → {csv_path}")

    # Plot
    ok = [r for r in rows if r["status"] == "success"]
    _plot_dual_axis(
        title=f"PTDA sweep tuning: N={n}",
        xlabel="num_sweeps",
        x_values=[r["num_sweeps"] for r in ok],
        energies=[r["best_energy"] for r in ok],
        times=[r["elapsed_sec"] for r in ok],
        oj_best_energy=oj_best,
        oj_elapsed_sec=oj_time,
        output_path=output_dir / f"ptda_sweep_n{n}.png",
    )
    print(f"  → {output_dir / f'ptda_sweep_n{n}.png'}")


# ──────────────────────────────────────────────────────────────────────
# Study 2 – num_replicas
# ──────────────────────────────────────────────────────────────────────


def run_study_replicas(
    n: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    oj_best: float | None,
    oj_time: float | None,
    output_dir: Path,
) -> None:
    ns = SWEEPS_STOPS.get(n, 20_000)
    print(f"  [Study 2] num_replicas={NUM_REPLICAS_LIST}  num_sweeps={ns}")

    sampler = PTDASampler(seed=SEED)
    rows: list[dict[str, Any]] = []

    for nr in NUM_REPLICAS_LIST:
        try:
            start = time.perf_counter()
            result = sampler.sample_ising(
                h=h,
                J=J,
                num_sweeps=ns,
                num_reads=NUM_READS,
                num_replicas=nr,
                exchange_interval=STUDY2_EXCHANGE_INTERVAL,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX,
                offset_delta=OFFSET_DELTA,
            )
            elapsed = time.perf_counter() - start
            best_e = float(result.first.energy)
            avg_e = float(np.mean(result.record.energy))
            rows.append(
                {
                    "num_replicas": nr,
                    "num_sweeps": ns,
                    "exchange_interval": STUDY2_EXCHANGE_INTERVAL,
                    "num_reads": NUM_READS,
                    "elapsed_sec": elapsed,
                    "best_energy": best_e,
                    "avg_energy": avg_e,
                    "status": "success",
                }
            )
            print(
                f"    replicas={nr:>4}  E_best={best_e:>12.4f}  "
                f"E_avg={avg_e:>12.4f}  time={elapsed:.3f}s"
            )
        except Exception as e:
            rows.append(
                {
                    "num_replicas": nr,
                    "num_sweeps": ns,
                    "exchange_interval": STUDY2_EXCHANGE_INTERVAL,
                    "num_reads": NUM_READS,
                    "elapsed_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"error: {e}",
                }
            )
            print(f"    replicas={nr:>4}  FAILED: {e}")

    # CSV
    csv_path = output_dir / f"ptda_replicas_n{n}.csv"
    save_rows_to_csv(rows, csv_path, create_parent=True)
    print(f"  → {csv_path}")

    # Plot
    ok = [r for r in rows if r["status"] == "success"]
    _plot_dual_axis(
        title=f"PTDA replicas tuning: N={n}  (sweeps={ns})",
        xlabel="num_replicas",
        x_values=[r["num_replicas"] for r in ok],
        energies=[r["best_energy"] for r in ok],
        times=[r["elapsed_sec"] for r in ok],
        oj_best_energy=oj_best,
        oj_elapsed_sec=oj_time,
        output_path=output_dir / f"ptda_replicas_n{n}.png",
    )
    print(f"  → {output_dir / f'ptda_replicas_n{n}.png'}")


# ──────────────────────────────────────────────────────────────────────
# Study 3 – exchange_interval
# ──────────────────────────────────────────────────────────────────────


def run_study_exchange(
    n: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    oj_best: float | None,
    oj_time: float | None,
    output_dir: Path,
) -> None:
    ns = SWEEPS_STOPS.get(n, 20_000)
    print(f"  [Study 3] exchange_intervals={EXCHANGE_INTERVALS}  num_sweeps={ns}")

    sampler = PTDASampler(seed=SEED)
    rows: list[dict[str, Any]] = []

    for ei in EXCHANGE_INTERVALS:
        try:
            start = time.perf_counter()
            result = sampler.sample_ising(
                h=h,
                J=J,
                num_sweeps=ns,
                num_reads=NUM_READS,
                num_replicas=STUDY3_NUM_REPLICAS,
                exchange_interval=ei,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX,
                offset_delta=OFFSET_DELTA,
            )
            elapsed = time.perf_counter() - start
            best_e = float(result.first.energy)
            avg_e = float(np.mean(result.record.energy))
            rows.append(
                {
                    "exchange_interval": ei,
                    "num_sweeps": ns,
                    "num_replicas": STUDY3_NUM_REPLICAS,
                    "num_reads": NUM_READS,
                    "elapsed_sec": elapsed,
                    "best_energy": best_e,
                    "avg_energy": avg_e,
                    "status": "success",
                }
            )
            print(
                f"    exchange_interval={ei:>6}  E_best={best_e:>12.4f}  "
                f"E_avg={avg_e:>12.4f}  time={elapsed:.3f}s"
            )
        except Exception as e:
            rows.append(
                {
                    "exchange_interval": ei,
                    "num_sweeps": ns,
                    "num_replicas": STUDY3_NUM_REPLICAS,
                    "num_reads": NUM_READS,
                    "elapsed_sec": None,
                    "best_energy": None,
                    "avg_energy": None,
                    "status": f"error: {e}",
                }
            )
            print(f"    exchange_interval={ei:>6}  FAILED: {e}")

    # CSV
    csv_path = output_dir / f"ptda_exchange_n{n}.csv"
    save_rows_to_csv(rows, csv_path, create_parent=True)
    print(f"  → {csv_path}")

    # Plot
    ok = [r for r in rows if r["status"] == "success"]
    _plot_dual_axis(
        title=f"PTDA exchange_interval tuning: N={n}  (sweeps={ns})",
        xlabel="exchange_interval",
        x_values=[r["exchange_interval"] for r in ok],
        energies=[r["best_energy"] for r in ok],
        times=[r["elapsed_sec"] for r in ok],
        oj_best_energy=oj_best,
        oj_elapsed_sec=oj_time,
        output_path=output_dir / f"ptda_exchange_n{n}.png",
    )
    print(f"  → {output_dir / f'ptda_exchange_n{n}.png'}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PTDA Parameter Tuning Benchmark")
    print("=" * 80)
    print(f"Problem sizes  : {PROBLEM_SIZES}")
    print(f"Study 1 sweeps : {SWEEPS_START} → (size-dep stop) × {SWEEPS_MULTIPLIER}")
    print(f"Study 2 replicas: {NUM_REPLICAS_LIST}")
    print(f"Study 3 exchange: {EXCHANGE_INTERVALS}")
    print(f"num_reads={NUM_READS}  beta=[{BETA_MIN}, {BETA_MAX}]  seed={SEED}")
    print(f"OpenJij sweeps : {OJ_NUM_SWEEPS}")
    print(f"Output dir     : {output_dir}")
    print()

    # GPU warmup
    print("Warming up GPU …")
    warmup_ptda()

    for n in PROBLEM_SIZES:
        print(f"\n{'━'*60}")
        print(f"N = {n}")
        print(f"{'━'*60}")

        h, J = generate_dense_problem_dict(n, seed=SEED)

        # OpenJij baseline (once per size, shared across all three studies)
        print("  OpenJij baseline …")
        oj_best, oj_avg, oj_time = get_openjij_baseline(h, J)
        if oj_best is not None:
            print(
                f"  OpenJij  E_best={oj_best:>12.4f}  E_avg={oj_avg:>12.4f}  "
                f"time={oj_time:.3f}s"
            )

        # Study 1
        run_study_sweeps(n, h, J, oj_best, oj_time, output_dir)

        # Study 2
        run_study_replicas(n, h, J, oj_best, oj_time, output_dir)

        # Study 3
        run_study_exchange(n, h, J, oj_best, oj_time, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
