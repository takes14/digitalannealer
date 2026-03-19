"""Input validation helpers for Ising/QUBO models."""

from collections.abc import Mapping
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

_SUPPORTED_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}
_INDEX_TYPES = (int, np.integer)
_NUMERIC_TYPES = (int, float, np.number)


def _raise_first_nonfinite_mapping_value(
    name: str, mapping: Mapping[Any, Any]
) -> NoReturn:
    """Raise a ValueError describing the first non-finite mapping entry."""
    for key, value in mapping.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} contains non-finite value at {key}: {value}")
    raise RuntimeError("expected a non-finite mapping value")


def _raise_first_nonfinite_array_value(name: str, array: np.ndarray) -> NoReturn:
    """Raise a ValueError describing the first non-finite array entry."""
    non_finite_positions = np.argwhere(~np.isfinite(array))
    first_pos = tuple(non_finite_positions[0])
    first_val = array[first_pos]
    raise ValueError(f"{name} contains non-finite value at {first_pos}: {first_val}")


def validate_qubo_input(
    Q: dict[tuple[int, int], float] | np.ndarray,
) -> None:
    """Validate QUBO matrix input for non-finite values.

    Args:
        Q: QUBO matrix as a dict or 2D numpy array.

    Raises:
        ValueError: If Q contains NaN or Inf values.
    """
    if isinstance(Q, dict):
        q_dict = cast(dict[tuple[int, int], float], Q)
        _validate_qubo_dict(q_dict)
        return

    if isinstance(Q, np.ndarray):
        q_array = cast(NDArray[Any], Q)
        _validate_qubo_array(q_array)
        return

    raise TypeError(f"Q must be a dict or numpy array, got {type(Q)}")


def _validate_qubo_dict(Q: dict[tuple[int, int], float]) -> None:
    """Validate QUBO dictionary input.

    Args:
        Q: QUBO matrix as a dict mapping (i, j) to coefficient.

    Raises:
        ValueError: If Q contains NaN or Inf values.
    """
    for (i, j), value in Q.items():
        if not np.isfinite(value):
            raise ValueError(
                f"QUBO matrix contains non-finite value at ({i}, {j}): {value}"
            )


def _validate_qubo_array(Q: np.ndarray) -> None:
    """Validate QUBO array input.

    Args:
        Q: QUBO matrix as a 2D numpy array.

    Raises:
        ValueError: If Q contains NaN or Inf values.
    """
    if not np.all(np.isfinite(Q)):
        _raise_first_nonfinite_array_value("QUBO matrix", Q)


def validate_ising_input(
    h: dict[int, float] | NDArray[np.floating[Any]],
    J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
) -> None:
    """Validate Ising model input parameters.

    Ensures that h and J have compatible types and valid structure for use
    with Digital Annealer samplers. Both dict and numpy array formats are supported.

    Args:
        h: Linear biases. Either a dict mapping integer variable indices to float values,
           or a 1D numpy array.
        J: Quadratic couplings. Either a dict mapping tuples of (i, j) to float values,
           or a 2D numpy array.

    Raises:
        ValueError: If input structure is invalid (e.g., mismatched dimensions,
            non-finite values, empty inputs).
        TypeError: If input types are incorrect or incompatible (e.g., h is dict
            but J is array).

    Notes:
        - When using dict format, both h and J must be dicts.
        - When using array format, both h and J must be numpy arrays.
        - Inputs are checked for NaN and Inf values in both dict and array form.
        - J must be a square matrix when using array format.
        - Time complexity: O(n²) for array input, O(m) for dict input where
          m is the number of entries.

    Example:
        Dict format validation:
        >>> h = {0: -1.0, 1: 1.0}
        >>> J = {(0, 1): -0.5}
        >>> validate_ising_input(h, J)  # Passes

        Array format validation:
        >>> h = np.array([-1.0, 1.0])
        >>> J = np.array([[0, -0.5], [-0.5, 0]])
        >>> validate_ising_input(h, J)  # Passes

        Invalid mixed format:
        >>> h = {0: -1.0}
        >>> J = np.array([[0]])
        >>> validate_ising_input(h, J)  # Raises TypeError
    """
    if isinstance(h, dict) and isinstance(J, dict):
        h_dict = cast(dict[int, float], h)
        j_dict = cast(dict[tuple[int, int], float], J)
        _validate_ising_dict(h_dict, j_dict)
        return

    if isinstance(h, np.ndarray) and isinstance(J, np.ndarray):
        h_array = cast(NDArray[np.floating[Any]], h)
        j_array = cast(NDArray[np.floating[Any]], J)
        _validate_ising_array(h_array, j_array)
        return

    if isinstance(h, dict):
        raise TypeError("If h is a dict, J must also be a dict")

    if isinstance(h, np.ndarray):
        raise TypeError("If h is a numpy array, J must also be a numpy array")

    raise TypeError(f"h must be a dict or numpy array, got {type(h)}")


def validate_dtype(dtype: DTypeLike) -> np.dtype:
    """Validate and normalize dtype parameter.

    Args:
        dtype: NumPy dtype object or floating point type requested by the caller.

    Returns:
        Normalized ``np.dtype`` instance.

    Raises:
        ValueError: If the dtype is not supported by the backend.
        TypeError: If dtype cannot be interpreted as a NumPy dtype.
    """
    try:
        normalized = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError("dtype must be a valid NumPy dtype or type") from exc

    if normalized not in _SUPPORTED_DTYPES:
        supported = ", ".join(
            sorted(
                (dt.name if isinstance(dt, np.dtype) else str(dt))
                for dt in _SUPPORTED_DTYPES
            )
        )
        raise ValueError(
            f"Unsupported dtype {normalized}; supported dtypes: {supported}"
        )

    return normalized


def _validate_ising_dict(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> None:
    """Validate Ising dict input.

    Uses numpy batch operations for value validation (NaN/Inf checks) to avoid
    O(N²) Python-loop overhead on large dense problems.  Type checks are
    performed on a small sample first; the full type check is only triggered
    when the fast batch path detects an issue.

    Args:
        h: Linear biases as a dict mapping integer variable index to float value.
        J: Quadratic couplings as a dict mapping (i, j) integer tuple to float value.

    Raises:
        TypeError: If h keys are not integers, J keys are not 2-tuples of integers,
            or values are not numeric types.
        ValueError: If h or J contain non-finite values (NaN or Inf).
    """
    # --- Validate h ---
    if h:
        # Spot-check first key/value for type (fast path)
        first_key = next(iter(h))
        if not isinstance(first_key, _INDEX_TYPES):
            raise TypeError(f"h keys must be integers, got {type(first_key)}")
        first_val = h[first_key]
        if not isinstance(first_val, _NUMERIC_TYPES):
            raise TypeError(f"h values must be numeric, got {type(first_val)}")

        # Batch NaN/Inf check via numpy
        try:
            h_vals = np.fromiter(h.values(), dtype=np.float64, count=len(h))
        except (TypeError, ValueError):
            # Fallback: one of the values is non-numeric — find it
            for key, val in h.items():
                if not isinstance(key, _INDEX_TYPES):
                    raise TypeError(
                        f"h keys must be integers, got {type(key)}"
                    ) from None
                if not isinstance(val, _NUMERIC_TYPES):
                    raise TypeError(
                        f"h values must be numeric, got {type(val)}"
                    ) from None
            h_vals = np.fromiter(h.values(), dtype=np.float64, count=len(h))

        if not np.all(np.isfinite(h_vals)):
            _raise_first_nonfinite_mapping_value("h", h)

        # Vectorized key-type check: np.fromiter with dtype=np.intp will raise
        # if any key is not integer-compatible (e.g. strings, floats, tuples).
        # OverflowError can occur for very large Python ints outside platform pointer range.
        try:
            np.fromiter(h.keys(), dtype=np.intp, count=len(h))
        except (TypeError, ValueError, OverflowError):
            for key in h:
                if not isinstance(key, _INDEX_TYPES):
                    raise TypeError(
                        f"h keys must be integers, got {type(key)}"
                    ) from None
                try:
                    np.intp(key)
                except (OverflowError, ValueError):
                    raise TypeError(
                        f"h key {key} is out of range for platform index type"
                    ) from None

    # --- Validate J ---
    if J:
        # Spot-check first key/value for type
        first_key_j = next(iter(J))
        if not isinstance(first_key_j, tuple) or len(first_key_j) != 2:
            raise TypeError(f"J keys must be tuples of length 2, got {first_key_j}")
        if not all(isinstance(k, _INDEX_TYPES) for k in first_key_j):
            raise TypeError(f"J keys must be tuples of integers, got {first_key_j}")
        first_val_j = J[first_key_j]
        if not isinstance(first_val_j, _NUMERIC_TYPES):
            raise TypeError(f"J values must be numeric, got {type(first_val_j)}")

        # Batch NaN/Inf check
        try:
            j_vals = np.fromiter(J.values(), dtype=np.float64, count=len(J))
        except (TypeError, ValueError):
            for key, val in J.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise TypeError(
                        f"J keys must be tuples of length 2, got {key}"
                    ) from None
                if not all(isinstance(k, _INDEX_TYPES) for k in key):
                    raise TypeError(
                        f"J keys must be tuples of integers, got {key}"
                    ) from None
                if not isinstance(val, _NUMERIC_TYPES):
                    raise TypeError(
                        f"J values must be numeric, got {type(val)}"
                    ) from None
            j_vals = np.fromiter(J.values(), dtype=np.float64, count=len(J))

        if not np.all(np.isfinite(j_vals)):
            _raise_first_nonfinite_mapping_value("J", J)

        # Vectorized key-type check for J: attempt to extract integer elements
        # from all keys. This catches non-tuple keys and non-integer elements.
        # OverflowError can occur for very large Python ints outside platform pointer range.
        try:
            j_keys = J.keys()
            np.fromiter(
                (k for key in j_keys for k in key), dtype=np.intp, count=len(J) * 2
            )
        except (TypeError, ValueError, OverflowError):
            for key in J:
                if not isinstance(key, tuple) or len(key) != 2:
                    raise TypeError(
                        f"J keys must be tuples of length 2, got {key}"
                    ) from None
                if not all(isinstance(k, _INDEX_TYPES) for k in key):
                    raise TypeError(
                        f"J keys must be tuples of integers, got {key}"
                    ) from None
                for k in key:
                    try:
                        np.intp(k)
                    except (OverflowError, ValueError):
                        raise TypeError(
                            f"J key element {k} is out of range for platform index type"
                        ) from None


def _validate_ising_array(
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
) -> None:
    """Validate Ising array input.

    Args:
        h: Linear biases as a 1D numpy array.
        J: Quadratic couplings as a 2D numpy array.

    Raises:
        ValueError: If h is not 1D, J is not 2D square, J dimensions don't match
            h length, or h/J contain NaN or Inf values.
    """
    if h.ndim != 1:
        raise ValueError(f"h must be a 1D array, got shape {h.shape}")

    if J.ndim != 2:
        raise ValueError(f"J must be a 2D array, got shape {J.shape}")

    if J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be a square matrix, got shape {J.shape}")

    n = len(h)
    if J.shape[0] != n:
        raise ValueError(f"J dimensions ({J.shape[0]}) must match h length ({n})")

    # Check for NaN/Inf
    if np.any(~np.isfinite(h)):
        raise ValueError("h contains NaN or Inf values")
    if np.any(~np.isfinite(J)):
        raise ValueError("J contains NaN or Inf values")
