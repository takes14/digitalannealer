#!/usr/bin/env python
"""PTDA beta-adaptation tuning benchmark for adapt_rounds/adapt_sweeps.

This script evaluates all requested combinations for intermediate temperature
adaptation settings:

- n_spin: [1000, 2000]
- exchange_interval: [10, 1000]
- adapt_rounds: [3, 5, 8, 10]
- adapt_sweeps: [10, 100, 1000, exchange_interval, 5 * n_spin]

For each case, it records energy/time metrics and writes:

- master CSV/JSON (all runs)
- summary CSV/JSON by adapt_rounds and by adapt_sweeps
- dual-axis PNG plots (Energy + Elapsed time)

Usage:
    python benchmarks/tune_ptda_adaptation.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from ._benchmark_utils import ensure_project_root_on_path, save_rows_to_csv
else:
    from _benchmark_utils import ensure_project_root_on_path, save_rows_to_csv

ROOT_DIR = ensure_project_root_on_path(__file__)

from benchmarks.problem_generators import generate_dense_problem_dict  # noqa: E402
from digitalannealer import PTDASampler  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEM_SIZES: list[int] = [1000, 2000]
NUM_SWEEPS_BY_SIZE: dict[int, int] = {
    1000: 50_000,
    2000: 100_000,
}
EXCHANGE_INTERVALS: list[int] = [10, 1000]
ADAPT_ROUNDS_LIST: list[int] = [3, 5, 8, 10]

NUM_READS: int = 100
NUM_REPLICAS: int = 10
BETA_MIN: float = 0.1
BETA_MAX: float = 10.0
OFFSET_DELTA: float = 0.1
SEED: int = 42

OUTPUT_DIR: str = "benchmarks/outputs/tuning"

MASTER_ROWS_FIELDNAMES: list[str] = [
    "case_id",
    "n_spin",
    "num_sweeps",
    "num_reads",
    "num_replicas",
    "exchange_interval",
    "adapt_rounds",
    "adapt_sweeps_label",
    "adapt_sweeps",
    "effective_adapt_sweeps",
    "sweeps_clamped",
    "elapsed_sec",
    "best_energy",
    "avg_energy",
    "status",
]

SUMMARY_ROUNDS_FIELDNAMES: list[str] = [
    "n_spin",
    "exchange_interval",
    "adapt_rounds",
    "num_success",
    "mean_best_energy",
    "min_best_energy",
    "mean_elapsed_sec",
    "min_elapsed_sec",
]

SUMMARY_SWEEPS_FIELDNAMES: list[str] = [
    "n_spin",
    "exchange_interval",
    "adapt_sweeps_label",
    "adapt_sweeps",
    "num_success",
    "mean_best_energy",
    "min_best_energy",
    "mean_elapsed_sec",
    "min_elapsed_sec",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def warmup_ptda() -> None:
    """Best-effort GPU context initialization."""
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
            adapt_betas=True,
            adapt_rounds=1,
            adapt_sweeps=10,
        )
    except Exception:
        pass


def build_adapt_sweeps_candidates(
    n_spin: int,
    exchange_interval: int,
) -> list[tuple[str, int]]:
    """Return ordered adapt_sweeps candidates with semantic labels.

    Note:
        We intentionally keep duplicates (e.g., 1000 and exchange_interval=1000)
        to respect the requested candidate list as-is.
    """
    return [
        ("10", 10),
        ("100", 100),
        ("1000", 1000),
        ("exchange_interval", exchange_interval),
        ("5*n_spin", 5 * n_spin),
    ]


def _safe_token(raw: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_")
    return token if token else "value"


def _adapt_sweeps_xtick(label: str, value: int) -> str:
    if label in {"10", "100", "1000"}:
        return str(value)
    return f"{label} ({value})"


def _plot_dual_axis(
    *,
    title: str,
    xlabel: str,
    x_values: list[int | float | str],
    energies: list[float],
    times: list[float],
    output_path: Path,
) -> None:
    """Save a dual-axis plot (energy + elapsed time)."""
    import matplotlib.pyplot as plt

    if not x_values:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Energy on left axis
    ax.plot(x_values, energies, marker="o", color="#1f77b4", label="Best Energy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Elapsed on right axis
    ax_time = ax.twinx()
    ax_time.plot(x_values, times, marker="s", color="#ff7f0e", label="Elapsed (s)")
    ax_time.set_ylabel("Elapsed time (s)", color="#ff7f0e")
    ax_time.tick_params(axis="y", labelcolor="#ff7f0e")

    if x_values and isinstance(x_values[0], str):
        ax.tick_params(axis="x", rotation=20)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_time.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rows_to_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _case_order(
    row: dict[str, Any],
    candidates: list[tuple[str, int]],
) -> tuple[int, int]:
    """Stable sorting key for adapt_rounds/adapt_sweeps plots."""
    adapt_round_idx = ADAPT_ROUNDS_LIST.index(row["adapt_rounds"])
    candidate_idx = candidates.index((row["adapt_sweeps_label"], row["adapt_sweeps"]))
    return adapt_round_idx, candidate_idx


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_single_case(
    *,
    case_id: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    n_spin: int,
    num_sweeps: int,
    exchange_interval: int,
    adapt_rounds: int,
    adapt_sweeps_label: str,
    adapt_sweeps: int,
) -> dict[str, Any]:
    """Run one PTDA adaptation configuration and return a result row."""
    sampler = PTDASampler(seed=SEED)
    effective_adapt_sweeps = min(num_sweeps, adapt_sweeps)

    try:
        start = time.perf_counter()
        result = sampler.sample_ising(
            h=h,
            J=J,
            num_sweeps=num_sweeps,
            num_reads=NUM_READS,
            num_replicas=NUM_REPLICAS,
            exchange_interval=exchange_interval,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            adapt_betas=True,
            adapt_rounds=adapt_rounds,
            adapt_sweeps=adapt_sweeps,
            offset_delta=OFFSET_DELTA,
        )
        elapsed = time.perf_counter() - start
        best_energy = float(result.first.energy)
        avg_energy = float(np.mean(result.record.energy))
        status = "success"
    except Exception as e:
        elapsed = None
        best_energy = None
        avg_energy = None
        status = f"error: {e}"

    return {
        "case_id": case_id,
        "n_spin": n_spin,
        "num_sweeps": num_sweeps,
        "num_reads": NUM_READS,
        "num_replicas": NUM_REPLICAS,
        "exchange_interval": exchange_interval,
        "adapt_rounds": adapt_rounds,
        "adapt_sweeps_label": adapt_sweeps_label,
        "adapt_sweeps": adapt_sweeps,
        "effective_adapt_sweeps": effective_adapt_sweeps,
        "sweeps_clamped": adapt_sweeps > num_sweeps,
        "elapsed_sec": elapsed,
        "best_energy": best_energy,
        "avg_energy": avg_energy,
        "status": status,
    }


def run_all_cases() -> list[dict[str, Any]]:
    """Run all requested Cartesian combinations and collect rows."""
    rows: list[dict[str, Any]] = []
    case_id = 0

    for n_spin in PROBLEM_SIZES:
        num_sweeps = NUM_SWEEPS_BY_SIZE.get(n_spin, max(NUM_SWEEPS_BY_SIZE.values()))
        print(f"\n{'━' * 72}")
        print(f"N={n_spin} | num_sweeps={num_sweeps} | num_reads={NUM_READS}")
        print(f"{'━' * 72}")

        h, J = generate_dense_problem_dict(n_spin, seed=SEED)

        for exchange_interval in EXCHANGE_INTERVALS:
            candidates = build_adapt_sweeps_candidates(n_spin, exchange_interval)
            print(f"  exchange_interval={exchange_interval}")

            for adapt_rounds in ADAPT_ROUNDS_LIST:
                for adapt_sweeps_label, adapt_sweeps in candidates:
                    case_id += 1
                    row = run_single_case(
                        case_id=case_id,
                        h=h,
                        J=J,
                        n_spin=n_spin,
                        num_sweeps=num_sweeps,
                        exchange_interval=exchange_interval,
                        adapt_rounds=adapt_rounds,
                        adapt_sweeps_label=adapt_sweeps_label,
                        adapt_sweeps=adapt_sweeps,
                    )
                    rows.append(row)

                    if row["status"] == "success":
                        print(
                            "    "
                            f"case={case_id:>2} "
                            f"adapt_rounds={adapt_rounds:>2} "
                            f"adapt_sweeps={adapt_sweeps_label}({adapt_sweeps:>5}) "
                            f"E_best={row['best_energy']:>12.4f} "
                            f"time={row['elapsed_sec']:.3f}s"
                        )
                    else:
                        print(
                            "    "
                            f"case={case_id:>2} "
                            f"adapt_rounds={adapt_rounds:>2} "
                            f"adapt_sweeps={adapt_sweeps_label}({adapt_sweeps:>5}) "
                            f"FAILED: {row['status']}"
                        )

    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def build_rounds_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful runs by (n_spin, exchange_interval, adapt_rounds)."""
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}

    for row in rows:
        if row["status"] != "success":
            continue
        key = (row["n_spin"], row["exchange_interval"], row["adapt_rounds"])
        bucket = grouped.setdefault(
            key,
            {
                "n_spin": row["n_spin"],
                "exchange_interval": row["exchange_interval"],
                "adapt_rounds": row["adapt_rounds"],
                "energies": [],
                "times": [],
            },
        )
        bucket["energies"].append(float(row["best_energy"]))
        bucket["times"].append(float(row["elapsed_sec"]))

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        bucket = grouped[key]
        energies = bucket["energies"]
        times = bucket["times"]
        summary_rows.append(
            {
                "n_spin": bucket["n_spin"],
                "exchange_interval": bucket["exchange_interval"],
                "adapt_rounds": bucket["adapt_rounds"],
                "num_success": len(energies),
                "mean_best_energy": float(np.mean(energies)),
                "min_best_energy": float(np.min(energies)),
                "mean_elapsed_sec": float(np.mean(times)),
                "min_elapsed_sec": float(np.min(times)),
            }
        )

    return summary_rows


def build_sweeps_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful runs by (n_spin, exchange_interval, adapt_sweeps)."""
    grouped: dict[tuple[int, int, str, int], dict[str, Any]] = {}

    for row in rows:
        if row["status"] != "success":
            continue
        key = (
            row["n_spin"],
            row["exchange_interval"],
            row["adapt_sweeps_label"],
            row["adapt_sweeps"],
        )
        bucket = grouped.setdefault(
            key,
            {
                "n_spin": row["n_spin"],
                "exchange_interval": row["exchange_interval"],
                "adapt_sweeps_label": row["adapt_sweeps_label"],
                "adapt_sweeps": row["adapt_sweeps"],
                "energies": [],
                "times": [],
            },
        )
        bucket["energies"].append(float(row["best_energy"]))
        bucket["times"].append(float(row["elapsed_sec"]))

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[3], item[2])):
        bucket = grouped[key]
        energies = bucket["energies"]
        times = bucket["times"]
        summary_rows.append(
            {
                "n_spin": bucket["n_spin"],
                "exchange_interval": bucket["exchange_interval"],
                "adapt_sweeps_label": bucket["adapt_sweeps_label"],
                "adapt_sweeps": bucket["adapt_sweeps"],
                "num_success": len(energies),
                "mean_best_energy": float(np.mean(energies)),
                "min_best_energy": float(np.min(energies)),
                "mean_elapsed_sec": float(np.mean(times)),
                "min_elapsed_sec": float(np.min(times)),
            }
        )

    return summary_rows


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def create_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate dual-axis plots for adapt_rounds and adapt_sweeps viewpoints."""
    success_rows = [row for row in rows if row["status"] == "success"]

    for n_spin in PROBLEM_SIZES:
        for exchange_interval in EXCHANGE_INTERVALS:
            candidates = build_adapt_sweeps_candidates(n_spin, exchange_interval)

            # Plot A: x = adapt_rounds (fixed adapt_sweeps candidate)
            for adapt_sweeps_label, adapt_sweeps in candidates:
                selected = [
                    row
                    for row in success_rows
                    if row["n_spin"] == n_spin
                    and row["exchange_interval"] == exchange_interval
                    and row["adapt_sweeps_label"] == adapt_sweeps_label
                    and row["adapt_sweeps"] == adapt_sweeps
                ]
                if not selected:
                    continue

                selected.sort(key=lambda row: _case_order(row, candidates))
                x_values = [row["adapt_rounds"] for row in selected]
                energies = [float(row["best_energy"]) for row in selected]
                times = [float(row["elapsed_sec"]) for row in selected]

                safe_label = _safe_token(adapt_sweeps_label)
                output_path = (
                    output_dir
                    / "plots"
                    / f"ptda_adapt_rounds_n{n_spin}_ei{exchange_interval}"
                    f"_as_{safe_label}_{adapt_sweeps}.png"
                )
                _plot_dual_axis(
                    title=(
                        "PTDA adaptation tuning (x=adapt_rounds): "
                        f"N={n_spin}, exchange_interval={exchange_interval}, "
                        f"adapt_sweeps={adapt_sweeps_label}({adapt_sweeps})"
                    ),
                    xlabel="adapt_rounds",
                    x_values=x_values,
                    energies=energies,
                    times=times,
                    output_path=output_path,
                )

            # Plot B: x = adapt_sweeps (fixed adapt_rounds)
            for adapt_rounds in ADAPT_ROUNDS_LIST:
                selected = [
                    row
                    for row in success_rows
                    if row["n_spin"] == n_spin
                    and row["exchange_interval"] == exchange_interval
                    and row["adapt_rounds"] == adapt_rounds
                ]
                if not selected:
                    continue

                selected.sort(key=lambda row: _case_order(row, candidates))
                x_values = [
                    _adapt_sweeps_xtick(row["adapt_sweeps_label"], row["adapt_sweeps"])
                    for row in selected
                ]
                energies = [float(row["best_energy"]) for row in selected]
                times = [float(row["elapsed_sec"]) for row in selected]

                output_path = (
                    output_dir
                    / "plots"
                    / f"ptda_adapt_sweeps_n{n_spin}_ei{exchange_interval}"
                    f"_ar_{adapt_rounds}.png"
                )
                _plot_dual_axis(
                    title=(
                        "PTDA adaptation tuning (x=adapt_sweeps): "
                        f"N={n_spin}, exchange_interval={exchange_interval}, "
                        f"adapt_rounds={adapt_rounds}"
                    ),
                    xlabel="adapt_sweeps",
                    x_values=x_values,
                    energies=energies,
                    times=times,
                    output_path=output_path,
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PTDA Beta-Adaptation Tuning Benchmark")
    print("=" * 80)
    print(f"Problem sizes      : {PROBLEM_SIZES}")
    print(f"num_sweeps by size : {NUM_SWEEPS_BY_SIZE}")
    print(f"exchange_interval  : {EXCHANGE_INTERVALS}")
    print(f"adapt_rounds       : {ADAPT_ROUNDS_LIST}")
    print("adapt_sweeps       : [10, 100, 1000, exchange_interval, 5*n_spin]")
    print(
        f"num_reads={NUM_READS}  num_replicas={NUM_REPLICAS}  "
        f"beta=[{BETA_MIN}, {BETA_MAX}]  seed={SEED}"
    )
    print(f"Output dir         : {output_dir}")
    print()

    print("Warming up GPU …")
    warmup_ptda()

    rows = run_all_cases()

    master_csv_path = output_dir / "ptda_adaptation_grid.csv"
    save_rows_to_csv(
        rows,
        master_csv_path,
        create_parent=True,
        fieldnames=MASTER_ROWS_FIELDNAMES,
    )
    print(f"\n→ {master_csv_path}")

    master_json_path = output_dir / "ptda_adaptation_grid.json"
    save_rows_to_json(rows, master_json_path)
    print(f"→ {master_json_path}")

    rounds_summary = build_rounds_summary(rows)
    rounds_csv_path = output_dir / "ptda_adaptation_summary_adapt_rounds.csv"
    save_rows_to_csv(
        rounds_summary,
        rounds_csv_path,
        create_parent=True,
        fieldnames=SUMMARY_ROUNDS_FIELDNAMES,
    )
    print(f"→ {rounds_csv_path}")

    rounds_json_path = output_dir / "ptda_adaptation_summary_adapt_rounds.json"
    save_rows_to_json(rounds_summary, rounds_json_path)
    print(f"→ {rounds_json_path}")

    sweeps_summary = build_sweeps_summary(rows)
    sweeps_csv_path = output_dir / "ptda_adaptation_summary_adapt_sweeps.csv"
    save_rows_to_csv(
        sweeps_summary,
        sweeps_csv_path,
        create_parent=True,
        fieldnames=SUMMARY_SWEEPS_FIELDNAMES,
    )
    print(f"→ {sweeps_csv_path}")

    sweeps_json_path = output_dir / "ptda_adaptation_summary_adapt_sweeps.json"
    save_rows_to_json(sweeps_summary, sweeps_json_path)
    print(f"→ {sweeps_json_path}")

    create_plots(rows, output_dir)
    print(f"→ {output_dir / 'plots'}")

    total_cases = len(rows)
    success_cases = sum(1 for row in rows if row["status"] == "success")
    print(f"\nDone. Success: {success_cases}/{total_cases}")


if __name__ == "__main__":
    main()
