"""CPU fallback helpers for PTDASampler.

This module provides pure-Python/NumPy implementations of the Parallel
Tempering loop and replica-exchange logic.  They are used when the
CUDA backend (``digitalannealer._core``) is not available, e.g. during
development and testing on machines without a GPU.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from digitalannealer.utils.energy import compute_all_delta_e, compute_energy

SingleSweepFn = Callable[
    [
        NDArray[np.int8],
        NDArray[np.float64],
        NDArray[np.floating[Any]],
        float,
        float,
        float,
        np.random.Generator,
    ],
    float,
]


def attempt_exchanges_cpu(
    spins: NDArray[np.int8],
    betas: NDArray[np.floating[Any]],
    replica_map: NDArray[np.int32],
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
    rng: np.random.Generator,
    exchange_stats: dict[str, NDArray[np.int64]] | None = None,
) -> None:
    """Attempt zero-copy replica exchanges between adjacent temperatures.

    Even-indexed and odd-indexed pairs are processed in separate sweeps
    (two-phase parity scheme) so that every pair is attempted once per call.

    Args:
        spins: Spin configurations for all replicas, shape ``(num_replicas, n_spins)``.
        betas: Sorted inverse temperatures, shape ``(num_replicas,)``.
        replica_map: Index map from temperature slot to spin buffer.  Swaps
            modify this array in-place rather than copying spin data.
        h: Linear biases, shape ``(n_spins,)``.
        J: Quadratic couplings, shape ``(n_spins, n_spins)``.
        rng: NumPy random number generator.
        exchange_stats: Optional dict with ``"attempts"`` and ``"accepted"``
            int64 arrays for tracking exchange acceptance rates.
    """
    num_replicas = len(betas)

    for exchange_parity in (0, 1):
        k = exchange_parity
        while k + 1 < num_replicas:
            r1 = replica_map[k]
            r2 = replica_map[k + 1]

            E1 = compute_energy(spins[r1], h, J)
            E2 = compute_energy(spins[r2], h, J)

            beta1 = betas[k]
            beta2 = betas[k + 1]

            delta = (beta2 - beta1) * (E1 - E2)
            if exchange_stats is not None:
                exchange_stats["attempts"][k] += 1

            accept = delta <= 0
            if not accept:
                # Numerically stable form of: rng.random() <= exp(-delta)
                u = max(float(rng.random()), np.finfo(np.float64).tiny)
                accept = np.log(u) <= -float(delta)

            if accept:
                replica_map[k], replica_map[k + 1] = replica_map[k + 1], replica_map[k]
                if exchange_stats is not None:
                    exchange_stats["accepted"][k] += 1

            k += 2


def run_pt_cpu(
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
    betas: NDArray[np.floating[Any]],
    num_sweeps: int,
    num_reads: int,
    exchange_interval: int,
    offset_delta: float,
    seed: int | None,
    single_sweep_fn: SingleSweepFn,
    exchange_stats: dict[str, NDArray[np.int64]] | None = None,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Run CPU fallback Parallel Tempering sampling.

    For each of *num_reads* independent runs the function initialises
    *num_replicas* random spin configurations, performs *num_sweeps*
    annealing sweeps with periodic replica exchange, and returns the
    final cold-replica sample.

    Args:
        h: Linear biases, shape ``(n_spins,)``.
        J: Quadratic couplings, shape ``(n_spins, n_spins)``.
        betas: Sorted inverse temperatures for each replica.
        num_sweeps: Total number of Monte Carlo sweeps per run.
        num_reads: Number of independent runs.
        exchange_interval: Sweeps between replica-exchange attempts.
        offset_delta: Dynamic offset increment on rejection.
        seed: Random seed (``None`` for non-deterministic).
        single_sweep_fn: Callable that performs a single DA sweep and
            returns the updated offset value.
        exchange_stats: Optional dict for tracking exchange statistics.

    Returns:
        Tuple ``(samples, energies)`` where *samples* has shape
        ``(num_reads, n_spins)`` (int8) and *energies* has shape
        ``(num_reads,)`` (float64).
    """
    rng = np.random.default_rng(seed)
    n_spins = len(h)
    num_replicas = len(betas)

    all_samples = np.zeros((num_reads, n_spins), dtype=np.int8)
    all_energies = np.zeros(num_reads, dtype=np.float64)

    for read_idx in range(num_reads):
        spins = rng.choice([-1, 1], size=(num_replicas, n_spins)).astype(np.int8)

        delta_e = np.zeros((num_replicas, n_spins), dtype=np.float64)
        for r in range(num_replicas):
            delta_e[r] = compute_all_delta_e(spins[r], h, J)

        offsets = np.zeros(num_replicas, dtype=np.float64)
        replica_map = np.arange(num_replicas, dtype=np.int32)

        for sweep_idx in range(num_sweeps):
            for temp_idx in range(num_replicas):
                r = replica_map[temp_idx]
                offsets[r] = single_sweep_fn(
                    spins[r],
                    delta_e[r],
                    J,
                    float(betas[temp_idx]),
                    offsets[r],
                    offset_delta,
                    rng,
                )

            if (sweep_idx + 1) % exchange_interval == 0:
                attempt_exchanges_cpu(
                    spins, betas, replica_map, h, J, rng, exchange_stats=exchange_stats
                )

        coldest_replica = replica_map[num_replicas - 1]
        all_samples[read_idx] = spins[coldest_replica]
        all_energies[read_idx] = compute_energy(spins[coldest_replica], h, J)

    return all_samples.astype(np.int8, copy=False), np.asarray(
        all_energies, dtype=np.float64
    )
