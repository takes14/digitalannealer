"""Parallel Tempering Digital Annealer Sampler implementation."""

import warnings
from typing import Any

import dimod
import numpy as np
from numpy.typing import DTypeLike, NDArray

from digitalannealer.samplers.base_sampler import BaseSampler
from digitalannealer.samplers.ptda_cpu import attempt_exchanges_cpu, run_pt_cpu
from digitalannealer.utils.validation import (
    normalize_ising_to_dense,
    normalize_ising_to_sparse,
)

ACCEPTANCE_EPS = 1e-8
_ADAPT_NUM_READS = 4


class PTDASampler(BaseSampler):
    """Parallel Tempering Digital Annealer Sampler.

    This sampler runs multiple replicas at different temperatures and
    periodically exchanges configurations between adjacent temperatures.
    All operations (updates and exchanges) run on the GPU.

    Supports two modes:
        - Dense mode (default): Uses dense interaction matrix. Best for fully-connected
          or near-fully-connected graphs.
        - Sparse mode: Uses CSR (Compressed Sparse Row) format with Shared Memory Resident
          (SMR) architecture. Best for sparse graphs. Problem size is limited by GPU
          shared memory capacity (varies by GPU model).

    Example:
        >>> sampler = PTDASampler()
        >>> h = {0: -1.0, 1: 1.0}
        >>> J = {(0, 1): -1.0}
        >>> result = sampler.sample_ising(h, J, num_replicas=10, num_sweeps=1000)
        >>> print(result.first.sample)

        # Using sparse mode for sparse graphs
        >>> result = sampler.sample_ising(h, J, num_replicas=10, num_sweeps=1000, sparse=True)
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the PTDASampler.

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
        num_replicas: int = 10,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        exchange_interval: int = 10,
        adapt_betas: bool = True,
        adapt_rounds: int = 3,
        adapt_sweeps: int | None = None,
        log_adaptation: bool = False,
        offset_delta: float = 0.1,
        seed: int | None = None,
        sparse: bool = False,
        dtype: DTypeLike = np.float32,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from a QUBO model by converting to Ising and running PTDA.

        The underlying PTDA algorithm remains unchanged: QUBO is converted to
        Ising, solved via :meth:`sample_ising`, then converted back to binary
        variables with the appropriate energy offset.

        Args:
            Q: QUBO matrix. Dict ``{(i, j): value}`` or 2D square array.
            num_sweeps: Number of Monte Carlo sweeps (default: 1000).
            num_reads: Number of independent runs to return (default: 1).
            num_replicas: Number of parallel tempering replicas (default: 10).
            beta_min: Minimum inverse temperature (default: 0.1).
            beta_max: Maximum inverse temperature (default: 10.0).
                The initial replica ladder is linearly spaced in beta-space
                between ``beta_min`` and ``beta_max``.
            exchange_interval: Sweeps between exchange attempts (default: 10).
            adapt_betas: Whether to adapt intermediate betas before the main run.
            adapt_rounds: Number of adaptation rounds when adaptation is enabled.
            adapt_sweeps: Sweeps per adaptation round (None selects auto value).
            log_adaptation: Whether to include adaptation history in ``info``.
            offset_delta: Dynamic offset increment on rejection (default: 0.1).
            seed: Random seed. Overrides sampler seed when provided.
            sparse: If True, run sparse CSR mode; otherwise dense mode.
            dtype: Computation dtype forwarded to :meth:`sample_ising`.
            **kwargs: Additional PTDA options forwarded to
                :meth:`sample_ising`.

        Returns:
            A ``dimod.SampleSet`` with ``BINARY`` vartype and energies in the
            original QUBO objective scale.

        Raises:
            TypeError: If ``Q`` has an unsupported type.
            ValueError: If ``Q`` or PTDA parameters are invalid.
            RuntimeError: If backend execution fails (e.g., memory limits).
        """
        return super().sample_qubo(
            Q,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            exchange_interval=exchange_interval,
            adapt_betas=adapt_betas,
            adapt_rounds=adapt_rounds,
            adapt_sweeps=adapt_sweeps,
            log_adaptation=log_adaptation,
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
        num_replicas: int = 10,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        exchange_interval: int = 10,
        adapt_betas: bool = True,
        adapt_rounds: int = 3,
        adapt_sweeps: int | None = None,
        log_adaptation: bool = False,
        offset_delta: float = 0.1,
        seed: int | None = None,
        sparse: bool = False,
        dtype: DTypeLike = np.float32,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from an Ising model using Parallel Tempering Digital Annealer.

        Args:
            h: Linear biases. Dict mapping variable index to bias, or 1D array.
            J: Quadratic interactions. Dict mapping (i, j) to coupling, or 2D array.
            num_sweeps: Number of Monte Carlo sweeps (default: 1000).
            num_reads: Number of independent runs to return (default: 1).
            num_replicas: Number of parallel tempering replicas (default: 10).
            beta_min: Minimum inverse temperature (default: 0.1).
            beta_max: Maximum inverse temperature (default: 10.0).
            exchange_interval: Sweeps between replica exchange attempts (default: 10).
            adapt_betas: If True, run short pre-runs to adapt intermediate betas
                using Hukushima-style acceptance equalization in linear-beta space.
                The ``beta_min`` and ``beta_max`` endpoints are always preserved;
                only intermediate betas are adjusted (default: True).
            adapt_rounds: Number of adaptation rounds when adapt_betas is True
                (default: 3).
            adapt_sweeps: Sweeps per adaptation round. If None, uses
                max(1000, n_spins * 5) to ensure sufficient per-spin updates and
                exchange events for reliable statistics; still capped at num_sweeps
                (default: None).
            log_adaptation: If True, include adaptation history in SampleSet.info for
                debugging/inspection (default: False).
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
        self._warn_ignored_kwargs(kwargs, "PTDASampler.sample_ising")
        if num_replicas < 2:
            raise ValueError(f"num_replicas must be >= 2, got {num_replicas}")
        if exchange_interval < 1:
            raise ValueError(f"exchange_interval must be >= 1, got {exchange_interval}")
        if adapt_betas and num_replicas > 2 and adapt_rounds < 1:
            raise ValueError("adapt_rounds must be >= 1 when adapt_betas is enabled")

        # Build initial temperature schedule in linear beta-space.
        betas = np.linspace(beta_min, beta_max, num_replicas, dtype=dtype)

        if sparse:
            return self._sample_sparse(
                h,
                J,
                betas,
                num_sweeps,
                num_reads,
                exchange_interval,
                offset_delta,
                effective_seed,
                adapt_betas,
                adapt_rounds,
                adapt_sweeps,
                log_adaptation,
                dtype,
            )
        else:
            return self._sample_dense(
                h,
                J,
                betas,
                num_sweeps,
                num_reads,
                exchange_interval,
                offset_delta,
                effective_seed,
                adapt_betas,
                adapt_rounds,
                adapt_sweeps,
                log_adaptation,
                dtype,
            )

    def _sample_dense(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        num_sweeps: int,
        num_reads: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
        adapt_betas: bool,
        adapt_rounds: int,
        adapt_sweeps: int | None,
        log_adaptation: bool,
        dtype: np.dtype,
    ) -> dimod.SampleSet:
        """Run dense mode Parallel Tempering sampling with optional beta adaptation.

        This method normalizes input to dense arrays, optionally adapts the beta
        schedule using short pre-runs, then calls the CUDA PT backend.

        Args:
            h: Linear biases (dict or array).
            J: Quadratic couplings (dict or array).
            betas: Initial temperature schedule (array of inverse temperatures).
            num_sweeps: Total number of sweeps for main run.
            num_reads: Number of independent runs.
            exchange_interval: Sweeps between replica exchange attempts.
            offset_delta: Dynamic offset increment value.
            seed: Random seed (None means non-deterministic).
            adapt_betas: Whether to run beta adaptation rounds.
            adapt_rounds: Number of adaptation rounds.
            adapt_sweeps: Sweeps per adaptation round.
            log_adaptation: Whether to include adaptation history in results.
            dtype: Data type for computation (must be float32).

        Returns:
            dimod.SampleSet with samples, energies, and PT-specific info.

        Notes:
            Dense mode is best for fully-connected graphs.
            Beta adaptation uses Hukushima-style acceptance equalization.
            Memory usage: O(n² + n*num_replicas) for model and replica states.
        """
        h_dense, J_dense, var_labels = normalize_ising_to_dense(h, J)
        h_dense, J_dense = self._cast_arrays(dtype, h_dense, J_dense)
        num_replicas = len(betas)

        adaptation_history: list[dict[str, Any]] = []
        if adapt_betas and num_replicas > 2:
            effective_adapt_sweeps = self._effective_adapt_sweeps(
                adapt_sweeps=adapt_sweeps,
                num_sweeps=num_sweeps,
                n_spins=h_dense.shape[0],
            )
            betas, adaptation_history = self._adapt_beta_schedule(
                h_dense,
                J_dense,
                betas,
                adapt_rounds=max(adapt_rounds, 1),
                adapt_sweeps=effective_adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
            )

        # Call C++ backend
        try:
            from digitalannealer._core import pt_sample_ising

            samples, energies = pt_sample_ising(
                **self._build_backend_kwargs(
                    dtype=dtype,
                    seed=seed,
                    scalar_kwargs={"offset_delta": offset_delta},
                    h=h_dense,
                    J=J_dense,
                    betas=betas,
                    num_sweeps=num_sweeps,
                    num_reads=num_reads,
                    exchange_interval=exchange_interval,
                )
            )
        except ImportError:
            # Fallback to CPU simulation for development/testing
            samples, energies = self._sample_cpu(
                h_dense,
                J_dense,
                betas,
                num_sweeps,
                num_reads,
                exchange_interval,
                offset_delta,
                seed,
                exchange_stats=None,
            )

        return self._build_sample_set(
            samples,
            energies,
            var_labels,
            dtype,
            extra_info=self._pt_info(
                betas, adapt_betas, num_replicas, log_adaptation, adaptation_history
            ),
        )

    def _sample_sparse(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        num_sweeps: int,
        num_reads: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
        adapt_betas: bool,
        adapt_rounds: int,
        adapt_sweeps: int | None,
        log_adaptation: bool,
        dtype: np.dtype,
    ) -> dimod.SampleSet:
        """Run sparse CSR mode Parallel Tempering sampling with optional beta adaptation.

        This method normalizes input to CSR format, optionally adapts the beta schedule
        using short CPU pre-runs (dense fallback), then calls the CUDA sparse PT backend.
        All replica updates and exchange decisions are performed on the GPU.

        Args:
            h: Linear biases (dict or array).
            J: Quadratic couplings (dict or array).
            betas: Initial temperature schedule (array of inverse temperatures).
            num_sweeps: Total number of sweeps for main run.
            num_reads: Number of independent runs.
            exchange_interval: Sweeps between replica exchange attempts.
            offset_delta: Dynamic offset increment value.
            seed: Random seed (None means non-deterministic).
            adapt_betas: Whether to run beta adaptation rounds.
            adapt_rounds: Number of adaptation rounds.
            adapt_sweeps: Sweeps per adaptation round.
            log_adaptation: Whether to include adaptation history in results.
            dtype: Data type for computation (must be float32).

        Returns:
            dimod.SampleSet with samples, energies, and PT-specific info.

        Notes:
            Sparse mode uses Shared Memory Resident (SMR) architecture: spins and delta_E
            are loaded into GPU shared memory at kernel start and all updates are performed
            there for fast random access. This constrains the problem size to what fits in
            GPU shared memory (typically N <= 4000 spins; device-dependent).
            Beta adaptation uses dense CPU fallback (short rounds only).
            Memory usage: O(nnz + n*num_replicas) for model and replica states.
        """
        h_dense, row_offsets, col_indices, values, var_labels = (
            normalize_ising_to_sparse(h, J)
        )
        h_dense, values = self._cast_arrays(dtype, h_dense, values)
        row_offsets_int32 = row_offsets.astype(np.int32, copy=False)
        col_indices_int32 = col_indices.astype(np.int32, copy=False)
        num_replicas = len(betas)

        # Beta adaptation using sparse GPU path when available
        adaptation_history: list[dict[str, Any]] = []
        if adapt_betas and num_replicas > 2:
            effective_adapt_sweeps = self._effective_adapt_sweeps(
                adapt_sweeps=adapt_sweeps,
                num_sweeps=num_sweeps,
                n_spins=len(h_dense),
            )
            betas, adaptation_history = self._adapt_beta_schedule_sparse(
                h_dense,
                row_offsets_int32,
                col_indices_int32,
                values,
                betas,
                adapt_rounds=max(adapt_rounds, 1),
                adapt_sweeps=effective_adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
            )

        # Call C++ backend
        try:
            from digitalannealer._core import pt_sample_ising_sparse

            samples, energies = pt_sample_ising_sparse(
                **self._build_backend_kwargs(
                    dtype=dtype,
                    seed=seed,
                    scalar_kwargs={"offset_delta": offset_delta},
                    h=h_dense,
                    row_offsets=row_offsets_int32,
                    col_indices=col_indices_int32,
                    values=values,
                    betas=betas,
                    num_sweeps=num_sweeps,
                    num_reads=num_reads,
                    exchange_interval=exchange_interval,
                )
            )
        except ImportError:
            # Fallback to CPU simulation using dense path
            warnings.warn(
                "CUDA backend not available. Sparse PT mode is falling back to dense CPU "
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
                betas,
                num_sweeps,
                num_reads,
                exchange_interval,
                offset_delta,
                seed,
                exchange_stats=None,
            )

        return self._build_sample_set(
            samples,
            energies,
            var_labels,
            dtype,
            extra_info=self._pt_info(
                betas, adapt_betas, num_replicas, log_adaptation, adaptation_history
            ),
        )

    @staticmethod
    def _pt_info(
        betas: NDArray[np.floating[Any]],
        adapt_betas: bool,
        num_replicas: int,
        log_adaptation: bool,
        adaptation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build PT-specific metadata dict for SampleSet.info.

        Args:
            betas: Final inverse temperature schedule used for the PT run.
            adapt_betas: Whether beta adaptation was requested.
            num_replicas: Number of temperature replicas.
            log_adaptation: Whether adaptation history should be included.
            adaptation_history: List of per-round adaptation data collected during
                the pre-run adaptation phase.

        Returns:
            Dict containing at minimum:
                - ``"betas"``: List of final beta values.
                - ``"beta_adaptation_used"``: Boolean indicating whether adaptation ran.
            When ``log_adaptation`` is True and history is non-empty, also includes:
                - ``"beta_adaptation_history"``: The full adaptation history list.
        """
        info: dict[str, Any] = {
            "betas": betas.tolist(),
            "beta_adaptation_used": bool(adapt_betas and num_replicas > 2),
        }
        if log_adaptation and adaptation_history:
            info["beta_adaptation_history"] = adaptation_history
        return info

    @staticmethod
    def _effective_adapt_sweeps(
        *,
        adapt_sweeps: int | None,
        num_sweeps: int,
        n_spins: int,
    ) -> int:
        """Resolve the number of sweeps used during beta adaptation."""
        effective_adapt_sweeps = (
            adapt_sweeps if adapt_sweeps is not None else max(1000, n_spins * 5)
        )
        return min(num_sweeps, effective_adapt_sweeps)

    def _adapt_beta_schedule(
        self,
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        *,
        adapt_rounds: int,
        adapt_sweeps: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
    ) -> tuple[NDArray[np.floating[Any]], list[dict[str, Any]]]:
        """Run short pre-runs to adapt intermediate betas using acceptance rates.

        This implements Hukushima-style beta adaptation: runs short sampling rounds,
        measures replica exchange acceptance rates between adjacent temperatures,
        and adjusts linear beta gaps using acceptance-rate scaling.

        Uses the GPU backend for exchange statistics collection when available,
        falling back to CPU simulation otherwise. GPU-based collection produces
        accurate statistics from the actual CUDA kernels even for large N.

        Args:
            h: Linear biases (dense array).
            J: Quadratic couplings (dense array).
            betas: Initial beta schedule.
            adapt_rounds: Number of adaptation iterations.
            adapt_sweeps: Sweeps per adaptation round.
            exchange_interval: Sweeps between exchange attempts.
            offset_delta: Dynamic offset increment.
            seed: Random seed for adaptation runs.

        Returns:
            Tuple of (adapted_betas, adaptation_history) where:
                - adapted_betas: Adjusted beta schedule with the same ``beta_min``
                  and ``beta_max`` endpoints as the input (endpoints are preserved).
                - adaptation_history: List of dicts with acceptance rates and betas per round.

        Notes:
            - Higher acceptance rates → increase spacing; lower → decrease spacing.
            - The first and last betas (``beta_min`` / ``beta_max``) are always
              kept fixed; only the intermediate values are adjusted.
        """
        try:
            from digitalannealer._core import pt_exchange_stats

            return self._adapt_beta_schedule_gpu(
                h,
                J,
                betas,
                adapt_rounds=adapt_rounds,
                adapt_sweeps=adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
                stats_fn=pt_exchange_stats,
            )
        except ImportError:
            return self._adapt_beta_schedule_cpu(
                h,
                J,
                betas,
                adapt_rounds=adapt_rounds,
                adapt_sweeps=adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
            )

    def _adapt_beta_schedule_sparse(
        self,
        h: NDArray[np.floating[Any]],
        row_offsets: NDArray[np.int32],
        col_indices: NDArray[np.int32],
        values: NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        *,
        adapt_rounds: int,
        adapt_sweeps: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
    ) -> tuple[NDArray[np.floating[Any]], list[dict[str, Any]]]:
        """Run sparse-mode beta adaptation using GPU exchange stats."""
        try:
            from digitalannealer._core import pt_exchange_stats_sparse

            def sparse_stats_fn(
                h_arr: NDArray,
                _J_unused: NDArray,
                betas_arr: NDArray,
                num_sweeps: int,
                num_reads: int,
                exc_interval: int,
                od: float,
                s: int,
            ) -> NDArray:
                return pt_exchange_stats_sparse(
                    h=h_arr,
                    row_offsets=row_offsets,
                    col_indices=col_indices,
                    values=values,
                    betas=betas_arr,
                    num_sweeps=num_sweeps,
                    num_reads=num_reads,
                    exchange_interval=exc_interval,
                    offset_delta=od,
                    seed=s,
                )

            return self._adapt_beta_schedule_gpu(
                h,
                np.empty(0),  # J unused for sparse GPU path
                betas,
                adapt_rounds=adapt_rounds,
                adapt_sweeps=adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
                stats_fn=sparse_stats_fn,
            )
        except ImportError:
            # CPU fallback: reconstruct dense J from CSR for _adapt_beta_schedule_cpu
            n_spins = len(h)
            J_dense = np.zeros((n_spins, n_spins), dtype=betas.dtype)
            row_offsets_int = row_offsets.astype(np.int64, copy=False)
            nnz = int(row_offsets_int[-1])
            if nnz > 0:
                row_counts = np.diff(row_offsets_int)
                row_indices = np.repeat(np.arange(n_spins, dtype=np.int64), row_counts)
                col_idx = col_indices[:nnz].astype(np.int64, copy=False)
                vals = values[:nnz].astype(betas.dtype, copy=False)
                J_dense[row_indices, col_idx] = vals
            return self._adapt_beta_schedule_cpu(
                h.astype(betas.dtype, copy=False),
                J_dense,
                betas,
                adapt_rounds=adapt_rounds,
                adapt_sweeps=adapt_sweeps,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=seed,
            )

    def _adapt_beta_schedule_gpu(
        self,
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        *,
        adapt_rounds: int,
        adapt_sweeps: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
        stats_fn: Any,
    ) -> tuple[NDArray[np.floating[Any]], list[dict[str, Any]]]:
        """Run beta adaptation using GPU exchange statistics collection.

        Args:
            h: Linear biases (dense array).
            J: Quadratic couplings (dense array, unused for sparse path).
            betas: Initial beta schedule.
            adapt_rounds: Number of adaptation iterations.
            adapt_sweeps: Sweeps per adaptation round.
            exchange_interval: Sweeps between exchange attempts.
            offset_delta: Dynamic offset increment.
            seed: Random seed for adaptation runs.
            stats_fn: Callable that collects exchange stats on GPU.

        Returns:
            Tuple of (adapted_betas, adaptation_history).
        """
        adapted_betas = np.asarray(betas, dtype=betas.dtype, copy=True)
        history: list[dict[str, Any]] = []
        current_seed = seed

        for round_idx in range(adapt_rounds):
            acceptance = np.asarray(
                stats_fn(
                    h,
                    J,
                    adapted_betas,
                    adapt_sweeps,
                    _ADAPT_NUM_READS,
                    exchange_interval,
                    self._coerce_backend_scalar(offset_delta, betas.dtype),
                    self._backend_seed(current_seed),
                ),
                dtype=np.float64,
            )
            target = float(np.mean(acceptance)) if len(acceptance) > 0 else 0.0

            if target > 0.0:
                adapted_betas = self._update_betas_from_acceptance(
                    adapted_betas, acceptance.astype(np.float32), target
                )

            history.append(
                {
                    "round": round_idx,
                    "acceptance": acceptance.astype(betas.dtype),
                    "betas": adapted_betas.copy(),
                    "target": target,
                }
            )

            if current_seed is not None:
                current_seed += 1

        return adapted_betas, history

    def _adapt_beta_schedule_cpu(
        self,
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        *,
        adapt_rounds: int,
        adapt_sweeps: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
    ) -> tuple[NDArray[np.floating[Any]], list[dict[str, Any]]]:
        """CPU fallback for beta adaptation (used when CUDA backend is unavailable)."""
        adapted_betas = np.asarray(betas, dtype=betas.dtype, copy=True)
        history: list[dict[str, Any]] = []
        current_seed = seed

        for round_idx in range(adapt_rounds):
            stats = {
                "attempts": np.zeros(len(adapted_betas) - 1, dtype=np.int64),
                "accepted": np.zeros(len(adapted_betas) - 1, dtype=np.int64),
            }
            self._sample_cpu(
                h,
                J,
                adapted_betas,
                num_sweeps=adapt_sweeps,
                num_reads=_ADAPT_NUM_READS,
                exchange_interval=exchange_interval,
                offset_delta=offset_delta,
                seed=current_seed,
                exchange_stats=stats,
            )

            attempt_counts = stats["attempts"]
            acceptance = np.divide(
                stats["accepted"],
                attempt_counts,
                out=np.zeros_like(attempt_counts, dtype=np.float64),
                where=attempt_counts > 0,
            )
            target = float(np.mean(acceptance)) if np.any(attempt_counts) else 0.0

            if target > 0.0:
                adapted_betas = self._update_betas_from_acceptance(
                    adapted_betas, acceptance.astype(np.float32), target
                )

            history.append(
                {
                    "round": round_idx,
                    "acceptance": acceptance.astype(betas.dtype),
                    "betas": adapted_betas.copy(),
                    "target": target,
                }
            )

            if current_seed is not None:
                current_seed += 1

        return adapted_betas, history

    @staticmethod
    def _update_betas_from_acceptance(
        betas: NDArray[np.floating[Any]],
        acceptance: NDArray[np.floating[Any]],
        target: float,
    ) -> NDArray[np.floating[Any]]:
        """Hukushima-style update in linear-β space for balanced adjustments.

        Adjusts adjacent beta gaps based on exchange acceptance rates. Updates are
        applied directly in linear beta space.

        Args:
            betas: Current beta schedule (must be all positive).
            acceptance: Acceptance rates for each adjacent pair [len(betas)-1].
            target: Normalization constant c, typically mean(acceptance).

        Returns:
            Updated beta schedule with adjusted linear spacing.
        """
        if len(betas) < 3:
            return betas

        if target <= ACCEPTANCE_EPS:
            return betas

        linear_betas = betas.astype(np.float64)
        linear_deltas = np.diff(linear_betas)
        if np.any(linear_deltas < 0):
            return betas

        span = float(linear_betas[-1] - linear_betas[0])
        if span <= 0.0:
            return betas

        scaling = acceptance.astype(np.float64) / float(target)
        scaling = np.where(np.isfinite(scaling), scaling, 0.0)
        scaling = np.clip(scaling, 0.0, None)

        adjusted_linear_deltas = linear_deltas * scaling
        adjusted_linear_deltas = np.maximum(adjusted_linear_deltas, 0.0)

        denom = float(np.sum(adjusted_linear_deltas))
        if denom <= ACCEPTANCE_EPS:
            return betas

        # Renormalize to keep endpoints fixed (beta_min/beta_max).
        adjusted_linear_deltas *= span / denom

        new_betas = np.empty_like(linear_betas)
        new_betas[0] = linear_betas[0]
        new_betas[1:] = linear_betas[0] + np.cumsum(adjusted_linear_deltas)
        new_betas[-1] = linear_betas[-1]

        return new_betas.astype(betas.dtype)

    def _sample_cpu(
        self,
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        betas: NDArray[np.floating[Any]],
        num_sweeps: int,
        num_reads: int,
        exchange_interval: int,
        offset_delta: float,
        seed: int | None,
        exchange_stats: dict[str, NDArray[np.int64]] | None = None,
    ) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
        """CPU fallback implementation for development/testing."""
        return run_pt_cpu(
            h=h,
            J=J,
            betas=betas,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            exchange_interval=exchange_interval,
            offset_delta=offset_delta,
            seed=seed,
            single_sweep_fn=self._da_single_sweep,
            exchange_stats=exchange_stats,
        )

    def _attempt_exchanges(
        self,
        spins: NDArray[np.int8],
        betas: NDArray[np.floating[Any]],
        replica_map: NDArray[np.int32],
        h: NDArray[np.floating[Any]],
        J: NDArray[np.floating[Any]],
        rng: np.random.Generator,
        exchange_stats: dict[str, NDArray[np.int64]] | None = None,
    ) -> None:
        """Backward-compatible wrapper around extracted CPU exchange helper."""
        attempt_exchanges_cpu(
            spins=spins,
            betas=betas,
            replica_map=replica_map,
            h=h,
            J=J,
            rng=rng,
            exchange_stats=exchange_stats,
        )
