"""Type stubs for _core C++ extension module."""

import numpy as np
from numpy.typing import NDArray

# Float arrays can be either float32 or float64; the C++ bindings dispatch
# to the appropriate template instantiation based on the dtype of `h`.
_FloatArray = NDArray[np.float32] | NDArray[np.float64]

def da_sample_ising(
    h: _FloatArray,
    J: _FloatArray,
    beta_schedule: _FloatArray,
    num_reads: int,
    offset_delta: float,
    seed: int,
) -> tuple[NDArray[np.int8], _FloatArray]:
    """Run Digital Annealer sampling on Ising model.

    Args:
        h: Linear biases [n_spins]. dtype determines precision (float32 or float64).
        J: Interaction matrix [n_spins, n_spins].
        beta_schedule: Inverse temperature schedule [num_sweeps].
        num_reads: Number of independent samples.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Tuple of (samples, energies) where:
            samples: Spin configurations [num_reads, n_spins].
            energies: Energies of each sample [num_reads], same dtype as h.
    """
    ...

def pt_sample_ising(
    h: _FloatArray,
    J: _FloatArray,
    betas: _FloatArray,
    num_sweeps: int,
    num_reads: int,
    exchange_interval: int,
    offset_delta: float,
    seed: int,
) -> tuple[NDArray[np.int8], _FloatArray]:
    """Run Parallel Tempering Digital Annealer sampling.

    Args:
        h: Linear biases [n_spins]. dtype determines precision (float32 or float64).
        J: Interaction matrix [n_spins, n_spins].
        betas: Inverse temperatures for replicas [num_replicas].
        num_sweeps: Total number of Monte Carlo sweeps.
        num_reads: Number of independent samples to return.
        exchange_interval: Sweeps between replica exchange attempts.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Tuple of (samples, energies) where:
            samples: Best spin configurations [num_reads, n_spins].
            energies: Energies of each sample [num_reads], same dtype as h.
    """
    ...

def da_sample_ising_sparse(
    h: _FloatArray,
    row_offsets: NDArray[np.int32],
    col_indices: NDArray[np.int32],
    values: _FloatArray,
    beta_schedule: _FloatArray,
    num_reads: int,
    offset_delta: float,
    seed: int,
) -> tuple[NDArray[np.int8], _FloatArray]:
    """Run Digital Annealer sampling on Ising model with sparse CSR input.

    Args:
        h: Linear biases [n_spins]. dtype determines precision (float32 or float64).
        row_offsets: CSR row pointers [n_spins + 1].
        col_indices: CSR column indices [nnz].
        values: CSR non-zero interaction values [nnz].
        beta_schedule: Inverse temperature schedule [num_sweeps].
        num_reads: Number of independent samples.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Tuple of (samples, energies).

    Raises:
        RuntimeError: If problem size exceeds shared memory capacity.
    """
    ...

def pt_sample_ising_sparse(
    h: _FloatArray,
    row_offsets: NDArray[np.int32],
    col_indices: NDArray[np.int32],
    values: _FloatArray,
    betas: _FloatArray,
    num_sweeps: int,
    num_reads: int,
    exchange_interval: int,
    offset_delta: float,
    seed: int,
) -> tuple[NDArray[np.int8], _FloatArray]:
    """Run Parallel Tempering Digital Annealer sampling with sparse CSR input.

    Args:
        h: Linear biases [n_spins]. dtype determines precision (float32 or float64).
        row_offsets: CSR row pointers [n_spins + 1].
        col_indices: CSR column indices [nnz].
        values: CSR non-zero interaction values [nnz].
        betas: Inverse temperatures for replicas [num_replicas].
        num_sweeps: Total number of Monte Carlo sweeps.
        num_reads: Number of independent samples to return.
        exchange_interval: Sweeps between replica exchange attempts.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Tuple of (samples, energies).

    Raises:
        RuntimeError: If problem size exceeds shared memory capacity.
    """
    ...

def check_vram_dense_da(
    n_spins: int,
    n_replicas: int,
    num_sweeps: int,
    dtype: str = "float32",
) -> None:
    """Check if sufficient VRAM is available for dense DA mode.

    Estimates the VRAM required for a dense Digital Annealer run and raises
    an error if the current GPU cannot accommodate it.

    Args:
        n_spins: Number of spin variables in the problem.
        n_replicas: Number of independent reads (parallel runs).
        num_sweeps: Number of annealing sweeps per run.
        dtype: Floating-point dtype ("float32" or "float64").

    Raises:
        RuntimeError: If estimated VRAM requirement exceeds available GPU memory
            (after applying the 90% safety margin).
        ValueError: If an unsupported dtype string is provided.
    """
    ...

def pt_exchange_stats(
    h: _FloatArray,
    J: _FloatArray,
    betas: _FloatArray,
    num_sweeps: int,
    num_reads: int,
    exchange_interval: int,
    offset_delta: float,
    seed: int,
) -> _FloatArray:
    """Collect per-pair exchange acceptance rates from short PT runs on GPU (dense).

    Runs the full PT pipeline on GPU for the specified sweeps/reads to collect
    exchange statistics, then returns per-pair acceptance rates.

    Args:
        h: Linear biases [n_spins]. dtype determines precision.
        J: Interaction matrix [n_spins, n_spins].
        betas: Inverse temperatures for replicas [num_replicas].
        num_sweeps: Number of sweeps per read.
        num_reads: Number of reads to average over.
        exchange_interval: Sweeps between exchange attempts.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Per-pair acceptance rates [num_replicas - 1], same dtype as h.
    """
    ...

def pt_exchange_stats_sparse(
    h: _FloatArray,
    row_offsets: NDArray[np.int32],
    col_indices: NDArray[np.int32],
    values: _FloatArray,
    betas: _FloatArray,
    num_sweeps: int,
    num_reads: int,
    exchange_interval: int,
    offset_delta: float,
    seed: int,
) -> _FloatArray:
    """Collect per-pair exchange acceptance rates from short PT runs on GPU (sparse CSR).

    Args:
        h: Linear biases [n_spins]. dtype determines precision.
        row_offsets: CSR row pointers [n_spins + 1].
        col_indices: CSR column indices [nnz].
        values: CSR non-zero interaction values [nnz].
        betas: Inverse temperatures for replicas [num_replicas].
        num_sweeps: Number of sweeps per read.
        num_reads: Number of reads to average over.
        exchange_interval: Sweeps between exchange attempts.
        offset_delta: Dynamic offset increment.
        seed: Random seed (-1 for random).

    Returns:
        Per-pair acceptance rates [num_replicas - 1], same dtype as h.
    """
    ...
