"""Digital Annealer Sampler implementation."""

import warnings
from typing import Any

import dimod
import numpy as np
from numpy.typing import DTypeLike, NDArray

from digitalannealer.samplers.base_sampler import BaseSampler
from digitalannealer.utils.energy import (
    compute_all_delta_e,
    compute_energy,
)
from digitalannealer.utils.validation import (
    normalize_ising_to_dense,
    normalize_ising_to_sparse,
)


class DASampler(BaseSampler):
    """Digital Annealer Sampler for solving Ising/QUBO problems on GPU.

    This sampler implements the Digital Annealer algorithm with parallel trial
    updates and dynamic offset mechanism. The entire annealing process runs
    on the GPU for maximum performance.

    Supports two modes:
        - Dense mode (default): Uses dense interaction matrix. Best for fully-connected
          or near-fully-connected graphs.
        - Sparse mode: Uses CSR (Compressed Sparse Row) format with Shared Memory Resident
          (SMR) architecture. Best for sparse graphs. Problem size is limited by GPU
          shared memory capacity (varies by GPU model).

    Example:
        >>> sampler = DASampler()
        >>> h = {0: -1.0, 1: 1.0}
        >>> J = {(0, 1): -1.0}
        >>> result = sampler.sample_ising(h, J, num_sweeps=1000)
        >>> print(result.first.sample)

        # Using sparse mode for sparse graphs
        >>> result = sampler.sample_ising(h, J, num_sweeps=1000, sparse=True)
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the DASampler.

        Args:
            seed: Random seed for reproducibility. If None, a random seed is used.
        """
        super().__init__(seed=seed)

    def sample_qubo(
        self,
        Q: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        *,
        num_sweeps: int = 1000,
        num_reads: int = 1,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        offset_delta: float = 0.1,
        seed: int | None = None,
        sparse: bool = False,
        dtype: DTypeLike = np.float32,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from a QUBO model by converting to Ising and running DA.

        The underlying DA algorithm remains unchanged: QUBO is converted to
        Ising, solved via :meth:`sample_ising`, then converted back to binary
        variables with the appropriate energy offset.

        Args:
            Q: QUBO matrix. Dict ``{(i, j): value}`` or 2D square array.
            num_sweeps: Number of Monte Carlo sweeps (default: 1000).
            num_reads: Number of independent runs/samples (default: 1).
            beta_min: Minimum inverse temperature (default: 0.1).
            beta_max: Maximum inverse temperature (default: 10.0).
            offset_delta: Dynamic offset increment on rejection (default: 0.1).
            seed: Random seed. Overrides sampler seed when provided.
            sparse: If True, run sparse CSR mode; otherwise dense mode.
            dtype: Computation dtype forwarded to :meth:`sample_ising`.
            **kwargs: Additional DA options forwarded to
                :meth:`sample_ising`.

        Returns:
            A ``dimod.SampleSet`` with ``BINARY`` vartype and energies in the
            original QUBO objective scale.

        Raises:
            TypeError: If ``Q`` has an unsupported type.
            ValueError: If ``Q`` or DA parameters are invalid.
            RuntimeError: If backend execution fails (e.g., memory limits).
        """
        return super().sample_qubo(
            Q,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_min=beta_min,
            beta_max=beta_max,
            offset_delta=offset_delta,
            seed=seed,
            sparse=sparse,
            dtype=dtype,
            **kwargs,
        )

    def sample_ising(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        *,
        num_sweeps: int = 1000,
        num_reads: int = 1,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        offset_delta: float = 0.1,
        seed: int | None = None,
        sparse: bool = False,
        dtype: DTypeLike = np.float32,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from an Ising model using the Digital Annealer algorithm.

        Args:
            h: Linear biases. Dict mapping variable index to bias, or 1D array.
            J: Quadratic interactions. Dict mapping (i, j) to coupling, or 2D array.
            num_sweeps: Number of Monte Carlo sweeps (default: 1000).
            num_reads: Number of independent runs/samples (default: 1).
            beta_min: Minimum inverse temperature (default: 0.1).
            beta_max: Maximum inverse temperature (default: 10.0).
            offset_delta: Dynamic offset increment on rejection (default: 0.1).
            seed: Random seed. Overrides the sampler's seed if provided.
            sparse: If True, use sparse CSR mode with Shared Memory Resident (SMR)
                architecture. Best for sparse graphs. Problem size is limited by GPU
                shared memory capacity (typically a few thousand spins). If False
                (default), use dense mode which is better for fully-connected graphs.
            dtype: Data type for computation. Currently only np.float32 is supported;
                specifying other dtypes raises ValueError.
            **kwargs: Additional parameters. Unknown keys are ignored with a warning
                for forward compatibility.

        Returns:
            A dimod.SampleSet containing the samples and energies.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If GPU memory is insufficient or problem exceeds
                shared memory capacity (sparse mode only).
        """
        # Validate input and prepare dtype/seed
        dtype, effective_seed = self._validate_and_prepare_inputs(h, J, dtype, seed)
        self._validate_sampling_params(
            num_sweeps, num_reads, beta_min, beta_max, offset_delta
        )
        self._warn_ignored_kwargs(kwargs, "DASampler.sample_ising")

        # Build annealing schedule (geometric cooling beta schedule)
        beta_schedule = np.geomspace(beta_min, beta_max, num_sweeps, dtype=dtype)

        if sparse:
            return self._sample_sparse(
                h, J, beta_schedule, num_reads, offset_delta, effective_seed, dtype
            )
        else:
            return self._sample_dense(
                h, J, beta_schedule, num_reads, offset_delta, effective_seed, dtype
            )

    def _sample_dense(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        beta_schedule: NDArray[np.float32],
        num_reads: int,
        offset_delta: float,
        seed: int | None,
        dtype: np.dtype,
    ) -> dimod.SampleSet:
        """Run dense mode sampling on GPU or CPU fallback.

        This method normalizes the input to dense arrays, calls the CUDA backend
        for GPU execution, and falls back to CPU if CUDA is unavailable.

        Args:
            h: Linear biases (dict or array).
            J: Quadratic couplings (dict or array).
            beta_schedule: Annealing schedule (array of inverse temperatures).
            num_reads: Number of independent samples to generate.
            offset_delta: Dynamic offset increment value.
            seed: Random seed (None means non-deterministic).
            dtype: Data type for computation (must be float32).

        Returns:
            dimod.SampleSet with samples and energies.

        Notes:
            Dense mode is best for fully-connected or near-fully-connected graphs.
            Memory usage: O(n²) where n is the number of spins.
        """
        h_dense, J_dense, var_labels = normalize_ising_to_dense(h, J)
        h_dense, J_dense = self._cast_arrays(dtype, h_dense, J_dense)

        # Call C++ backend
        try:
            from digitalannealer._core import da_sample_ising

            samples, energies = da_sample_ising(
                **self._build_backend_kwargs(
                    dtype=dtype,
                    seed=seed,
                    scalar_kwargs={"offset_delta": offset_delta},
                    h=h_dense,
                    J=J_dense,
                    beta_schedule=beta_schedule,
                    num_reads=num_reads,
                )
            )
        except ImportError:
            # Fallback to CPU simulation for development/testing
            samples, energies = self._sample_cpu(
                h_dense, J_dense, beta_schedule, num_reads, offset_delta, seed
            )

        return self._build_sample_set(samples, energies, var_labels, dtype)

    def _sample_sparse(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        beta_schedule: NDArray[np.float32],
        num_reads: int,
        offset_delta: float,
        seed: int | None,
        dtype: np.dtype,
    ) -> dimod.SampleSet:
        """Run sparse CSR mode sampling on GPU or CPU fallback.

        This method normalizes the input to CSR format, calls the CUDA sparse backend,
        and falls back to dense CPU implementation if CUDA is unavailable.

        Args:
            h: Linear biases (dict or array).
            J: Quadratic couplings (dict or array).
            beta_schedule: Annealing schedule (array of inverse temperatures).
            num_reads: Number of independent samples to generate.
            offset_delta: Dynamic offset increment value.
            seed: Random seed (None means non-deterministic).
            dtype: Data type for computation (must be float32).

        Returns:
            dimod.SampleSet with samples and energies.

        Notes:
            Sparse mode uses Shared Memory Resident (SMR) architecture.
            Problem size is limited by GPU shared memory capacity.
            Best for sparse graphs with few interactions.
            Memory usage: O(nnz) where nnz is the number of non-zero entries.
        """
        h_dense, row_offsets, col_indices, values, var_labels = (
            normalize_ising_to_sparse(h, J)
        )
        h_dense, values = self._cast_arrays(dtype, h_dense, values)
        row_offsets_int32 = row_offsets.astype(np.int32, copy=False)
        col_indices_int32 = col_indices.astype(np.int32, copy=False)

        # Call C++ backend
        try:
            from digitalannealer._core import da_sample_ising_sparse

            samples, energies = da_sample_ising_sparse(
                **self._build_backend_kwargs(
                    dtype=dtype,
                    seed=seed,
                    scalar_kwargs={"offset_delta": offset_delta},
                    h=h_dense,
                    row_offsets=row_offsets_int32,
                    col_indices=col_indices_int32,
                    values=values,
                    beta_schedule=beta_schedule,
                    num_reads=num_reads,
                )
            )
        except ImportError:
            # Fallback to CPU simulation using dense path
            warnings.warn(
                "CUDA backend not available. Sparse mode is falling back to dense CPU "
                "implementation, which may have different memory characteristics and "
                "be slower for large problems.",
                RuntimeWarning,
                stacklevel=2,
            )
            h_dense_full, J_dense, _ = normalize_ising_to_dense(h, J)
            h_dense_full, J_dense = self._cast_arrays(dtype, h_dense_full, J_dense)
            samples, energies = self._sample_cpu(
                h_dense_full,
                J_dense,
                beta_schedule,
                num_reads,
                offset_delta,
                seed,
            )

        return self._build_sample_set(samples, energies, var_labels, dtype)

    def _sample_cpu(
        self,
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        beta_schedule: NDArray[np.floating[Any]],
        num_reads: int,
        offset_delta: float,
        seed: int | None,
    ) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
        """CPU fallback implementation for development/testing.

        Implements the Digital Annealer algorithm with parallel trial scheme and
        Metropolis-Hastings acceptance using inverse function table lookup.
        This is slower but useful for testing without GPU.

        The acceptance criterion uses the inverse function approach:
        - For each spin i, generate random r_i ~ U(0, 1)
        - Compute threshold_i = -log(r_i) / beta
        - Accept spin i if delta_E_i - offset <= threshold_i

        This is mathematically equivalent to the standard Metropolis test:
        - If delta_E <= 0: Always accept (energy decreases)
        - If delta_E > 0: Accept with probability exp(-beta * delta_E)

        Args:
            h: Linear biases as a 1D dense array.
            J: Quadratic couplings as a 2D dense array.
            beta_schedule: Annealing schedule (array of inverse temperatures).
            num_reads: Number of independent samples to generate.
            offset_delta: Dynamic offset increment value.
            seed: Random seed (None means non-deterministic).

        Returns:
            Tuple of (samples, energies) where:
                - samples: Spin configurations array of shape (num_reads, n_spins),
                  dtype int8, values in {-1, +1}.
                - energies: Energy of each sample, shape (num_reads,), dtype float64.
        """
        rng = np.random.default_rng(seed)
        n_spins = len(h)
        num_sweeps = len(beta_schedule)

        all_samples = np.zeros((num_reads, n_spins), dtype=np.int8)
        all_energies = np.zeros(num_reads, dtype=np.float64)

        for read_idx in range(num_reads):
            # Initialize random spin configuration
            spins = rng.choice([-1, 1], size=n_spins).astype(np.int8)

            # Compute initial delta_E for all spins
            delta_e = compute_all_delta_e(spins, h, J)

            offset = 0.0

            for sweep_idx in range(num_sweeps):
                offset = self._da_single_sweep(
                    spins,
                    delta_e,
                    J,
                    beta_schedule[sweep_idx],
                    offset,
                    offset_delta,
                    rng,
                )

            # Store final configuration
            all_samples[read_idx] = spins
            all_energies[read_idx] = compute_energy(spins, h, J)

        return all_samples, all_energies
