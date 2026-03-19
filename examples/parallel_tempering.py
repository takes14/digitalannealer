#!/usr/bin/env python
"""Parallel Tempering example: when PTDASampler is useful.

We solve a moderate random Ising spin-glass instance several times and compare:
  - DASampler (single-temperature annealing)
  - PTDASampler (replica exchange across temperatures)
"""

import numpy as np

from digitalannealer import DASampler, PTDASampler


def generate_spin_glass(n_spins: int = 40, density: float = 0.3, seed: int = 12345):
    """Generate a random Ising spin-glass instance."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(n_spins).astype(np.float64)
    J = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if rng.random() < density:
                coupling = rng.standard_normal()
                J[i, j] = coupling
                J[j, i] = coupling
    return h, J


def run_trials(
    sampler_cls, h: np.ndarray, J: np.ndarray, trial_seeds, **sample_kwargs
) -> np.ndarray:
    """Run independent attempts and return best energy from each run."""
    best_energies = []
    for seed in trial_seeds:
        sampler = sampler_cls(seed=seed)
        response = sampler.sample_ising(h, J, **sample_kwargs)
        best_energies.append(float(response.first.energy))
    return np.asarray(best_energies, dtype=np.float64)


def format_summary(name: str, energies: np.ndarray) -> str:
    """Summarize energies from repeated runs."""
    return f"{name:<10} best={energies.min():8.3f}  median={np.median(energies):8.3f}"


def main():
    """Compare DA vs PTDA on a rugged random Ising instance."""
    h, J = generate_spin_glass(n_spins=40, density=0.3, seed=12345)
    edge_count = int(np.count_nonzero(np.triu(J, k=1)))

    print("Parallel Tempering (PTDASampler) demo")
    print(f"Problem: random Ising spin glass (40 spins, {edge_count} couplings)")
    print("Metric: best energy from each independent run (lower is better)\n")

    trial_seeds = range(8)
    common_kwargs = {
        "num_sweeps": 1500,
        "num_reads": 8,
        "beta_min": 0.1,
        "beta_max": 5.0,
    }

    da_energies = run_trials(DASampler, h, J, trial_seeds, **common_kwargs)
    pt_energies = run_trials(
        PTDASampler,
        h,
        J,
        trial_seeds,
        num_replicas=12,
        exchange_interval=10,
        **common_kwargs,
    )

    print(format_summary("DASampler", da_energies))
    print(format_summary("PTDASampler", pt_energies))

    median_gain = float(np.median(da_energies) - np.median(pt_energies))
    if median_gain > 0:
        print(
            f"\nPTDASampler found lower energies more consistently (median gain: {median_gain:.3f})."
        )
    else:
        print("\nBoth samplers performed similarly on this random instance.")
    print("PT is often more helpful when the energy landscape has many local minima.")


if __name__ == "__main__":
    main()
