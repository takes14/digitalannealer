#!/usr/bin/env python
"""Basic Ising example with DASampler and PTDASampler."""

from digitalannealer import DASampler, PTDASampler


def print_best(label, sampleset):
    """Print the best sample and energy."""
    best = sampleset.first
    print(f"{label:12} energy={best.energy:6.2f} sample={dict(best.sample)}")
    return best


def main():
    print("Digital Annealer: simple Ising example")

    # 1) Define a small Ising problem.
    h = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    J = {(0, 1): -1.0, (1, 2): 1.5, (2, 3): -1.0}
    print(f"h = {h}")
    print(f"J = {J}")

    # 2) Run DASampler.
    da_result = DASampler(seed=42).sample_ising(
        h, J, num_sweeps=1000, num_reads=10, beta_min=0.1, beta_max=10.0
    )

    # 3) Run PTDASampler.
    pt_result = PTDASampler(seed=42).sample_ising(
        h,
        J,
        num_sweeps=1000,
        num_reads=10,
        num_replicas=8,
        beta_min=0.1,
        beta_max=10.0,
        exchange_interval=10,
    )

    # 4) Print a concise comparison.
    print("\nBest results")
    da_best = print_best("DASampler", da_result)
    pt_best = print_best("PTDASampler", pt_result)

    if da_best.energy < pt_best.energy:
        winner = "DASampler"
    elif pt_best.energy < da_best.energy:
        winner = "PTDASampler"
    else:
        winner = "Tie"
    print(f"Comparison: {winner}")


if __name__ == "__main__":
    main()
