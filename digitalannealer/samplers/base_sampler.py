"""Abstract base class for samplers."""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import dimod
import numpy as np
from numpy.typing import DTypeLike, NDArray

from digitalannealer.utils.energy import _MIN_RANDOM_VALUE, update_delta_e


class BaseSampler(ABC):
    """Abstract base class for Digital Annealer samplers.

    All samplers inherit from this class and implement the sample_ising method.
    The API is designed to be compatible with dimod samplers.

    This base class provides common utilities used by all sampler implementations
    to reduce code duplication and maintain consistency.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the sampler with an optional default seed."""
        self._seed = seed

    @abstractmethod
    def sample_ising(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from an Ising model.

        Args:
            h: Linear biases. Either a dict mapping variable to bias,
               or a 1D numpy array.
            J: Quadratic interactions. Either a dict mapping (i, j) to coupling,
               or a 2D numpy array (dense interaction matrix).
            **kwargs: Additional sampler-specific parameters.

        Returns:
            A dimod.SampleSet containing the samples and their energies.
        """
        pass

    def sample_qubo(
        self,
        Q: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> dimod.SampleSet:
        """Sample from a QUBO model via internal QUBO->Ising conversion.

        This helper is shared by all concrete samplers. It converts ``Q`` to
        Ising parameters ``(h, J, offset)``, delegates optimization to
        :meth:`sample_ising`, then converts the returned SPIN samples to BINARY
        while applying ``offset`` so energies match the original QUBO objective.

        Args:
            Q: QUBO matrix. Either a dict mapping ``(i, j)`` to coefficient, or
                a 2D square numpy array.
            **kwargs: Sampler-specific keyword arguments forwarded directly to
                :meth:`sample_ising` on the concrete sampler.

        Returns:
            A ``dimod.SampleSet`` with ``BINARY`` vartype and energies for the
            original QUBO model.

        Raises:
            TypeError: If ``Q`` has an unsupported type.
            ValueError: If ``Q`` has invalid shape/contents.
            Exception: Any exception raised by the concrete
                :meth:`sample_ising` implementation.
        """
        from digitalannealer.models.converters import qubo_to_ising

        h, J, offset = qubo_to_ising(Q)
        sample_set = self.sample_ising(h, J, **kwargs)

        # Adjust samples from spin to binary and fix energies
        return sample_set.change_vartype(dimod.BINARY, energy_offset=offset)

    def _validate_and_prepare_inputs(
        self,
        h: dict[int, float] | NDArray[np.floating[Any]],
        J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
        dtype: DTypeLike,
        seed: int | None,
    ) -> tuple[np.dtype, int | None]:
        """Validate inputs and prepare dtype and seed.

        This is a common utility used by all samplers to reduce duplication
        of input validation and preparation logic.

        Args:
            h: Linear biases (will be validated).
            J: Quadratic interactions (will be validated).
            dtype: Requested data type (will be validated).
            seed: Seed provided to sample_ising call.

        Returns:
            A tuple of (validated_dtype, effective_seed) where effective_seed
            is the seed parameter if not None, otherwise the sampler seed.
        """
        from digitalannealer.utils.validation import (
            validate_dtype,
            validate_ising_input,
        )

        validate_ising_input(h, J)
        validated_dtype = validate_dtype(dtype)
        effective_seed = self._resolve_seed(seed)

        return validated_dtype, effective_seed

    def _resolve_seed(self, seed: int | None) -> int | None:
        """Resolve a per-call seed against the sampler's default seed."""
        return self._seed if seed is None else seed

    @staticmethod
    def _backend_seed(seed: int | None) -> int:
        """Convert an optional Python seed into the backend sentinel convention."""
        return -1 if seed is None else seed

    @staticmethod
    def _coerce_backend_scalar(value: Any, dtype: DTypeLike) -> Any:
        """Coerce a Python scalar to the backend dtype and unwrap it."""
        return np.asarray(value, dtype=dtype).item()

    @classmethod
    def _build_backend_kwargs(
        cls,
        *,
        dtype: DTypeLike,
        seed: int | None,
        scalar_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build backend kwargs with centralized scalar coercion and seed handling."""
        backend_kwargs = dict(kwargs)
        if scalar_kwargs:
            for key, value in scalar_kwargs.items():
                backend_kwargs[key] = cls._coerce_backend_scalar(value, dtype)
        backend_kwargs["seed"] = cls._backend_seed(seed)
        return backend_kwargs

    @staticmethod
    def _cast_arrays(
        dtype: DTypeLike,
        *arrays: NDArray[Any],
    ) -> tuple[NDArray[Any], ...]:
        """Cast arrays to a shared dtype without unnecessary copies."""
        return tuple(array.astype(dtype, copy=False) for array in arrays)

    @staticmethod
    def _validate_sampling_params(
        num_sweeps: int,
        num_reads: int,
        beta_min: float,
        beta_max: float,
        offset_delta: float,
    ) -> None:
        """Validate common sampling hyperparameters.

        Args:
            num_sweeps: Number of Monte Carlo sweeps.
            num_reads: Number of independent samples.
            beta_min: Minimum inverse temperature.
            beta_max: Maximum inverse temperature.
            offset_delta: Dynamic offset increment on rejection.

        Raises:
            ValueError: If any parameter is out of the valid range.
        """
        if num_sweeps < 1:
            raise ValueError(f"num_sweeps must be >= 1, got {num_sweeps}")
        if num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {num_reads}")
        if beta_min <= 0:
            raise ValueError(f"beta_min must be > 0, got {beta_min}")
        if beta_max < beta_min:
            raise ValueError(
                f"beta_max must be >= beta_min, got beta_max={beta_max}, beta_min={beta_min}"
            )
        if offset_delta < 0:
            raise ValueError(f"offset_delta must be >= 0, got {offset_delta}")

    @staticmethod
    def _build_sample_set(
        samples: NDArray[np.int8],
        energies: NDArray[np.floating[Any]],
        var_labels: list[int],
        dtype: np.dtype,
        extra_info: dict[str, Any] | None = None,
    ) -> dimod.SampleSet:
        """Build a dimod SampleSet from raw samples and energies.

        Uses dimod's array-based interface ``(samples_array, variable_labels)``
        which avoids constructing ``num_reads × n_spins`` Python dicts and is
        significantly faster for large problems.

        Args:
            samples: Array of samples with shape (num_reads, n_spins).
            energies: Array of energies for each sample.
            var_labels: Variable labels for mapping spin indices.
            dtype: Data type used for computation.
            extra_info: Optional extra metadata to include in SampleSet.info.

        Returns:
            A dimod.SampleSet containing the samples and energies.
        """
        info: dict[str, Any] = {"dtype": dtype.name}
        if extra_info:
            info.update(extra_info)
        return dimod.SampleSet.from_samples(
            (samples, var_labels),
            vartype=dimod.SPIN,
            energy=energies,
            info=info,
        )

    @staticmethod
    def _warn_ignored_kwargs(kwargs: dict[str, Any], caller: str) -> None:
        """Warn when unknown keyword arguments are ignored for compatibility."""
        if not kwargs:
            return
        ignored = ", ".join(sorted(kwargs))
        warnings.warn(
            f"{caller} received unsupported keyword argument(s): {ignored}. "
            "These arguments are ignored for forward compatibility.",
            UserWarning,
            stacklevel=3,
        )

    @staticmethod
    def _da_single_sweep(
        spins: NDArray[np.int8],
        delta_e: NDArray[np.float64],
        J: NDArray[np.floating[Any]],
        beta: float,
        offset: float,
        offset_delta: float,
        rng: np.random.Generator,
    ) -> float:
        """Execute one DA sweep on a single spin configuration.

        Implements parallel trial with Metropolis-Hastings acceptance using
        inverse function table lookup. One candidate among negative effective-ΔE
        spins is selected uniformly and flipped; on rejection the dynamic offset
        is increased.

        Args:
            spins: Spin configuration (1D, modified in place on flip).
            delta_e: Per-spin ΔE values (1D, updated differentially on flip).
            J: Quadratic couplings (2D dense matrix).
            beta: Inverse temperature for this sweep.
            offset: Current dynamic offset.
            offset_delta: Increment on rejection.
            rng: Random number generator.

        Returns:
            Updated offset (0.0 on successful flip, offset + offset_delta otherwise).
        """
        n_spins = len(spins)
        effective_delta_e = delta_e - offset
        r = rng.random(n_spins)
        r = np.maximum(r, _MIN_RANDOM_VALUE)
        thresholds = -np.log(r) / beta
        candidates = np.where(effective_delta_e <= thresholds)[0]

        if len(candidates) > 0:
            flip_idx = rng.choice(candidates)
            spins[flip_idx] *= -1
            update_delta_e(delta_e, spins, J, flip_idx)
            return 0.0
        else:
            return offset + offset_delta
