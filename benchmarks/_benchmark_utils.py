"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def ensure_project_root_on_path(current_file: str | Path) -> Path:
    """Ensure the repository root for benchmark scripts is importable."""
    root_dir = Path(current_file).resolve().parent.parent
    root_dir_str = str(root_dir)
    if root_dir_str not in sys.path:
        sys.path.insert(0, root_dir_str)
    return root_dir


def save_rows_to_csv(
    rows: Sequence[Mapping[str, Any]],
    filepath: str | Path,
    *,
    create_parent: bool = False,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write mapping rows to CSV using the first row keys as fieldnames."""
    if not rows and fieldnames is None:
        return

    output_path = Path(filepath)
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_fieldnames = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def build_geometric_sweeps(start: int, stop: int, multiplier: float) -> list[int]:
    """Return sorted, deduplicated geometric sweep counts including ``stop``."""
    values: list[int] = []
    current = float(start)
    while current <= stop + 1e-9:
        values.append(int(round(current)))
        current *= multiplier
    values.append(stop)
    return sorted(set(values))


def add_standard_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    num_sweeps_default: int = 1000,
    num_reads_default: int = 1,
    beta_min_default: float = 0.1,
    beta_max_default: float = 10.0,
    seed_default: int = 42,
) -> None:
    """Add shared optional sampling CLI arguments to ``parser``."""
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=num_sweeps_default,
        help=f"Number of Monte Carlo sweeps (default: {num_sweeps_default})",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=num_reads_default,
        help=f"Number of independent samples (default: {num_reads_default})",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=beta_min_default,
        help=f"Minimum inverse temperature (default: {beta_min_default})",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=beta_max_default,
        help=f"Maximum inverse temperature (default: {beta_max_default})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help=f"Random seed (default: {seed_default})",
    )


def add_beta_and_seed_args(
    parser: argparse.ArgumentParser,
    *,
    beta_min_default: float = 0.1,
    beta_max_default: float = 10.0,
    seed_default: int = 42,
) -> None:
    """Add shared beta-range and seed CLI arguments to ``parser``."""
    parser.add_argument(
        "--beta-min",
        type=float,
        default=beta_min_default,
        help=f"Minimum inverse temperature (default: {beta_min_default})",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=beta_max_default,
        help=f"Maximum inverse temperature (default: {beta_max_default})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help=f"Random seed (default: {seed_default})",
    )
