"""Converters between QUBO and Ising model representations."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from digitalannealer.utils.validation import (
    normalize_ising_to_dense,
    validate_ising_input,
    validate_qubo_input,
)


def qubo_to_ising(
    Q: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Convert a QUBO matrix to Ising model parameters.

    The conversion follows the standard transformation:
        x_i = (s_i + 1) / 2

    where x_i is a binary variable (0 or 1) and s_i is a spin variable (-1 or +1).

    Args:
        Q: QUBO matrix. Either a dict mapping (i, j) to coefficient,
           or a 2D numpy array. When using dict format, both (i, j) and (j, i)
           entries are treated as the same interaction and their values are summed
           (matching dimod convention).

    Returns:
        A tuple (h, J, offset) where:
            - h: 1D numpy array of linear biases for the Ising model.
            - J: 2D numpy array of quadratic couplings for the Ising model (symmetric).
            - offset: Energy offset to add to Ising energy to recover QUBO energy.

    Raises:
        ValueError: If Q is empty, has invalid format, or contains non-finite values (NaN or Inf).
        TypeError: If Q is not a dict or numpy array.

    Notes:
        - The returned J matrix is always symmetric: J[i, j] = J[j, i].
        - For dict input, variable indices are automatically mapped to a contiguous
          range [0, n-1] where n is the number of unique variables.
        - For array input, Q should be a square 2D matrix where Q[i, i] represents
          the linear term and Q[i, j] (i ≠ j) represents the interaction.
        - Time complexity: O(n²) for array input, O(m) for dict input where m is
          the number of entries in Q.
        - Space complexity: O(n²) output where n is the number of variables.

    Example:
        Dict input (sparse representation):
        >>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
        >>> h, J, offset = qubo_to_ising(Q)
        >>> print(h)  # Linear biases
        >>> print(J)  # Quadratic couplings (symmetric)

        Array input (dense representation):
        >>> Q = np.array([[-1, 2], [0, -1]])
        >>> h, J, offset = qubo_to_ising(Q)
    """
    if isinstance(Q, dict):
        Q_dict = cast(dict[tuple[int, int], float], Q)
        if len(Q_dict) == 0:
            raise ValueError("QUBO matrix Q cannot be empty")
        validate_qubo_input(Q_dict)
        return _qubo_dict_to_ising(Q_dict)

    if isinstance(Q, np.ndarray):
        Q_array = cast(NDArray[np.floating[Any]], Q)
        if Q_array.ndim != 2 or Q_array.shape[0] != Q_array.shape[1]:
            raise ValueError("QUBO matrix Q must be a 2D square array")
        if Q_array.shape[0] == 0:
            raise ValueError("QUBO matrix Q cannot be empty")
        validate_qubo_input(Q_array)
        return _qubo_array_to_ising(Q_array)

    raise TypeError(f"Q must be a dict or numpy array, got {type(Q)}")


def _qubo_dict_to_ising(
    Q: dict[tuple[int, int], float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Convert a QUBO dict to Ising parameters using the dimod convention.

    Uses the transformation x_i = (s_i + 1) / 2 and the dimod convention.

    QUBO (dimod): E = sum_{i<=j} Q_ij x_i x_j
    Ising (dimod): E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset

    Derivation (for i<j term, counted ONCE in the QUBO sum):
    - Diagonal: Q_ii x_i = Q_ii (s_i+1)/2 = (Q_ii/2) s_i + Q_ii/2
    - Off-diagonal: Q_ij x_i x_j
        = Q_ij ((s_i+1)/2)((s_j+1)/2)
        = (Q_ij/4) s_i s_j + (Q_ij/4) s_i + (Q_ij/4) s_j + Q_ij/4

    Therefore (dimod convention):
    - h_i = Q_ii/2 + sum_{j!=i} Q_ij/4
    - J_ij = Q_ij/4 (for off-diagonal, stored symmetrically)
    - offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4

    Args:
        Q: QUBO matrix as a dict mapping (i, j) to coefficient. Both (i, j) and (j, i)
           entries are canonicalized and summed before conversion (dimod convention).

    Returns:
        Tuple of (h, J, offset) as described above.

    Notes:
        - Both (i,j) and (j,i) entries are summed after canonicalization (i <= j).
        - Accumulation is streamed: no full Q matrix is built.
        - Time complexity: O(m) where m is the number of entries in Q.
        - Space complexity: O(n²) for the output J matrix.
    """
    # Find the number of variables
    # canonical_Q maps canonicalized (i, j) with i <= j to summed values.
    # If both (i,j) and (j,i) are provided, they are treated as duplicates and summed
    # (matching dimod's standard handling of repeated interactions).
    canonical_Q: dict[tuple[int, int], float] = {}
    variables = set()
    for (i, j), q_ij in Q.items():
        key = (i, j) if i <= j else (j, i)
        variables.add(key[0])
        variables.add(key[1])
        canonical_Q[key] = canonical_Q.get(key, 0.0) + float(q_ij)

    # Create variable index mapping
    var_list = sorted(variables)
    var_to_idx = {v: idx for idx, v in enumerate(var_list)}
    n = len(var_list)

    # Streamed accumulation of Ising parameters without building full Q matrix
    diag = np.zeros(n, dtype=np.float64)
    row_sum_offdiag = np.zeros(n, dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)
    sum_diag = 0.0
    sum_offdiag_upper = 0.0

    for (i, j), q_ij in canonical_Q.items():
        idx_i = var_to_idx[i]
        idx_j = var_to_idx[j]

        if idx_i == idx_j:
            # Diagonal contribution
            diag[idx_i] += q_ij
            sum_diag += q_ij
        else:
            # Off-diagonal contributions are symmetrized
            row_sum_offdiag[idx_i] += q_ij
            row_sum_offdiag[idx_j] += q_ij

            # J_ij = Q_ij/4 for off-diagonal (symmetric, dimod convention)
            quarter_q = q_ij / 4.0
            J[idx_i, idx_j] += quarter_q
            J[idx_j, idx_i] += quarter_q

            # For the offset we need sum_{i<j} Q_ij over the symmetrized off-diagonal
            sum_offdiag_upper += q_ij

    # Compute Ising parameters (dimod convention)
    # h_i = Q_ii/2 + sum_{j!=i} Q_ij/4
    h = diag / 2.0 + row_sum_offdiag / 4.0

    # offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4
    offset = float(sum_diag / 2.0 + sum_offdiag_upper / 4.0)

    return h, J, offset


def _qubo_array_to_ising(
    Q: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Convert a QUBO array to Ising parameters using the dimod convention.

    Uses the same transformation as _qubo_dict_to_ising. Both (i,j) and (j,i)
    entries are canonicalized by symmetrizing before conversion, matching dimod
    convention where Q[i,j] and Q[j,i] are treated as the same interaction.

    Args:
        Q: QUBO matrix as a 2D square numpy array (float64 or compatible type).

    Returns:
        Tuple of (h, J, offset) where:
            - h: 1D array of linear biases (float64).
            - J: 2D symmetric array of quadratic couplings (float64).
            - offset: Energy constant for recovering QUBO energy from Ising energy.

    Notes:
        - Q is first symmetrized: Q_canon = triu(Q) + tril(Q,-1).T to combine
          upper and lower off-diagonal entries into a canonical upper-triangular form.
        - Time complexity: O(n²) where n is the number of variables.
        - Space complexity: O(n²) for the output J matrix.
    """
    Q_arr = np.asarray(Q, dtype=np.float64)

    # Canonicalize like the dict form:
    # - treat (i,j) and (j,i) as duplicates and sum them into a single interaction
    # - keep diagonal as-is
    diag = np.diag(Q_arr)
    off_upper = np.triu(Q_arr, k=1) + np.tril(Q_arr, k=-1).T
    row_sum_offdiag = np.sum(off_upper, axis=1) + np.sum(off_upper, axis=0)

    # Compute Ising parameters (dimod convention)
    # h_i = Q_ii/2 + sum_{j!=i} Q_ij/4
    h = diag / 2.0 + row_sum_offdiag / 4.0

    # J_ij = Q_ij/4 for off-diagonal
    J = (off_upper + off_upper.T) / 4.0

    # offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4
    offset = float(np.sum(diag) / 2.0 + np.sum(off_upper) / 4.0)

    return h, J, offset


def ising_to_qubo(
    h: dict[int, float] | NDArray[np.floating[Any]],
    J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], float]:
    """Convert Ising model parameters to a QUBO matrix.

    The conversion follows the inverse transformation:
        s_i = 2 * x_i - 1

    Args:
        h: Linear biases. Dict mapping variable to bias, or 1D array.
        J: Quadratic couplings. Dict mapping (i, j) to coupling, or 2D array.
           The J matrix should be symmetric or will be symmetrized during conversion.

    Returns:
        A tuple (Q, offset) where:
            - Q: 2D numpy array representing the QUBO matrix.
            - offset: Energy offset to add to QUBO energy to recover Ising energy.

    Raises:
        ValueError: If h or J contain non-finite values, or have incompatible dimensions.
        TypeError: If h and J have incompatible types (e.g., one dict and one array).

    Notes:
        - The input J matrix is symmetrized before conversion: J_sym = (J + J.T) / 2.
        - For dict input, variables are automatically mapped to a contiguous range.
        - Time complexity: O(n²) where n is the number of variables.
        - Space complexity: O(n²) output.

    Example:
        Array input:
        >>> h = np.array([-1.0, 1.0])
        >>> J = np.array([[0, -0.5], [-0.5, 0]])
        >>> Q, offset = ising_to_qubo(h, J)

        Dict input:
        >>> h = {0: -1.0, 1: 1.0}
        >>> J = {(0, 1): -0.5}
        >>> Q, offset = ising_to_qubo(h, J)
    """
    validate_ising_input(h, J)

    # Convert dict to dense arrays, reusing the shared normalizer to avoid duplication
    h_arr, J_arr, _ = normalize_ising_to_dense(h, J)

    # normalize_ising_to_dense already returns a symmetric dense J_arr for both
    # array and dict input, so no additional symmetrization is needed.
    row_sum_offdiag = np.sum(J_arr, axis=1) - np.diag(J_arr)

    # Compute QUBO matrix (dimod convention)
    # Q_ii = 2*h_i - 2*sum_{j!=i} J_ij
    # Q_ij = 4*J_ij (for i != j)
    Q = 4.0 * J_arr
    np.fill_diagonal(Q, 2.0 * h_arr - 2.0 * row_sum_offdiag)

    # Compute offset
    # offset = -sum_i h_i + sum_{i<j} J_ij
    offset = float(-np.sum(h_arr) + np.sum(np.triu(J_arr, k=1)))

    return Q, offset
