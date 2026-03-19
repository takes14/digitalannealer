#!/usr/bin/env python
"""Tiny QUBO example with DASampler."""

from digitalannealer import DASampler


def main() -> None:
    # Minimize: -3*x0 -2*x1 -1*x2 + 4*(x0*x1 + x0*x2 + x1*x2)
    qubo = {
        (0, 0): -3.0,
        (1, 1): -2.0,
        (2, 2): -1.0,
        (0, 1): 4.0,
        (0, 2): 4.0,
        (1, 2): 4.0,
    }

    sampler = DASampler(seed=42)
    sampleset = sampler.sample_qubo(
        qubo,
        num_reads=20,
        num_sweeps=1000,
        beta_min=0.1,
        beta_max=10.0,
    )

    best = sampleset.first
    print("Best solution:")
    print(f"  sample={dict(best.sample)} energy={best.energy:.3f}")

    print("\nTop samples:")
    for i, (sample, energy, count) in enumerate(
        sampleset.data(["sample", "energy", "num_occurrences"])
    ):
        if i >= 5:
            break
        bits = "".join(str(sample[v]) for v in sorted(sample))
        print(f"  {i + 1}. x={bits} energy={energy:.3f} count={count}")


if __name__ == "__main__":
    main()
