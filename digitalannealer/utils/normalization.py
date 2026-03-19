"""Normalization helpers for Ising model dense/CSR conversions."""

from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray


class _IsingDictArrays(NamedTuple):
    """Vectorized views of dict-based Ising inputs."""

    h_keys: NDArray[np.intp]
    h_values: NDArray[np.float64]
    j_rows: NDArray[np.intp]
    j_cols: NDArray[np.intp]
    j_values: NDArray[np.float64]


def _as_float64_vector(values: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Convert a 1D array-like input to float64 without unnecessary copies."""
    return np.asarray(values, dtype=np.float64)


def _symmetrize_dense_matrix(matrix: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Return a float64 symmetric matrix using the standard average convention."""
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    return (matrix_arr + matrix_arr.T) / 2.0


def _extract_ising_dict_arrays(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> _IsingDictArrays:
    """Extract dense numpy views from dict-based Ising inputs once."""
    from operator import itemgetter

    get_row = itemgetter(0)
    get_col = itemgetter(1)

    h_count = len(h)
    h_keys = np.fromiter(h.keys(), dtype=np.intp, count=h_count)
    h_values = np.fromiter(h.values(), dtype=np.float64, count=h_count)

    j_count = len(J)
    if j_count == 0:
        empty = np.empty(0, dtype=np.intp)
        return _IsingDictArrays(
            h_keys=h_keys,
            h_values=h_values,
            j_rows=empty,
            j_cols=empty,
            j_values=np.empty(0, dtype=np.float64),
        )

    j_keys = J.keys()
    j_rows = np.fromiter(map(get_row, j_keys), dtype=np.intp, count=j_count)
    j_cols = np.fromiter(map(get_col, j_keys), dtype=np.intp, count=j_count)
    j_values = np.fromiter(J.values(), dtype=np.float64, count=j_count)
    return _IsingDictArrays(
        h_keys=h_keys,
        h_values=h_values,
        j_rows=j_rows,
        j_cols=j_cols,
        j_values=j_values,
    )


def _sorted_variable_labels(dict_arrays: _IsingDictArrays) -> NDArray[np.intp]:
    """Return sorted variable labels appearing in h or J."""
    if dict_arrays.j_rows.size == 0:
        return np.unique(dict_arrays.h_keys)
    return np.unique(
        np.concatenate((dict_arrays.h_keys, dict_arrays.j_rows, dict_arrays.j_cols))
    )


def _is_zero_based_contiguous(labels: NDArray[np.intp]) -> bool:
    """Check whether sorted labels already match dense array indices."""
    return bool(
        labels.size > 0 and int(labels[0]) == 0 and int(labels[-1]) == len(labels) - 1
    )


def _remap_indices(
    labels: NDArray[np.intp], var_to_idx: dict[int, int]
) -> NDArray[np.intp]:
    """Map original variable labels to dense zero-based indices."""
    return np.fromiter(
        (var_to_idx[int(label)] for label in labels),
        dtype=np.intp,
        count=len(labels),
    )


def normalize_ising_to_dense(
    h: dict[int, float] | NDArray[np.floating[Any]],
    J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Normalize Ising input to dense numpy arrays.

    Converts dict or array input to standardized dense array format suitable
    for use with the dense GPU kernels. Variable indices are mapped to a
    contiguous range for dict input.

    Args:
        h: Linear biases. Dict mapping variable indices to values, or 1D array.
        J: Quadratic couplings. Dict mapping (i, j) tuples to values, or 2D array.

    Returns:
        Tuple of (h_dense, J_dense, var_labels) where:
            - h_dense: 1D numpy array of linear biases (float64).
            - J_dense: 2D numpy array of quadratic couplings (float64, symmetric).
            - var_labels: List mapping array indices to original variable labels.
              For dict input, this preserves the original variable names.
              For array input, this is simply [0, 1, 2, ..., n-1].

    Notes:
        - The returned J matrix is always symmetric: J[i, j] = J[j, i].
        - For dict input with non-contiguous variable indices (e.g., {0: ..., 5: ...}),
          variables are remapped to [0, 1] and var_labels preserves the mapping.
        - Time complexity: O(n²) for array input, O(m + n²) for dict input where
          m is the number of entries and n is the number of variables.
        - Space complexity: O(n²) where n is the number of variables.

    Example:
        Dict input with non-contiguous indices:
        >>> h = {0: -1.0, 5: 1.0}
        >>> J = {(0, 5): -0.5}
        >>> h_dense, J_dense, var_labels = normalize_ising_to_dense(h, J)
        >>> print(var_labels)  # [0, 5] - preserves original indices
        >>> print(h_dense.shape)  # (2,)
        >>> print(J_dense.shape)  # (2, 2)

        Array input:
        >>> h = np.array([-1.0, 1.0])
        >>> J = np.array([[0, -0.5], [-0.5, 0]])
        >>> h_dense, J_dense, var_labels = normalize_ising_to_dense(h, J)
        >>> print(var_labels)  # [0, 1]
    """
    if isinstance(h, dict) and isinstance(J, dict):
        h_dict = cast(dict[int, float], h)
        j_dict = cast(dict[tuple[int, int], float], J)
        return _dict_to_dense(h_dict, j_dict)

    if isinstance(h, np.ndarray) and isinstance(J, np.ndarray):
        h_array = cast(NDArray[np.floating[Any]], h)
        j_array = cast(NDArray[np.floating[Any]], J)
        h_dense = _as_float64_vector(h_array)
        J_dense = _symmetrize_dense_matrix(j_array)
        var_labels = list(range(len(h_dense)))
        return h_dense, J_dense, var_labels

    if isinstance(h, dict):
        raise TypeError("If h is a dict, J must also be a dict")

    if isinstance(h, np.ndarray):
        raise TypeError("If h is a numpy array, J must also be a numpy array")

    raise TypeError(f"h must be a dict or numpy array, got {type(h)}")


def _dict_to_dense(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Convert Ising dict representation to dense numpy arrays.

    Uses np.fromiter with operator.itemgetter for fast dict-to-array
    conversion, then numpy vectorized indexing for O(m) fill of J_dense.
    This avoids expensive np.array(list-of-tuples) conversions.

    Args:
        h: Linear biases as a dict mapping integer variable index to float value.
        J: Quadratic couplings as a dict mapping (i, j) integer tuple to float value.

    Returns:
        Tuple of (h_dense, J_dense, var_labels) where:
            - h_dense: 1D numpy array of linear biases (float64).
            - J_dense: 2D symmetric numpy array of quadratic couplings (float64).
            - var_labels: Sorted list of original variable labels (ints).

    Notes:
        - Variables are mapped to contiguous indices [0, n-1] sorted by label.
        - J is symmetrized: both (i,j) and (j,i) entries are set to the coupling value.
        - Time complexity: O(n**2 + m) where m is the number of J entries and n is
          the number of variables (dense (n, n) allocation dominates for large n).
        - Space complexity: O(n**2) due to the dense J_dense matrix.
    """
    dict_arrays = _extract_ising_dict_arrays(h, J)
    all_vars = _sorted_variable_labels(dict_arrays)

    var_list = all_vars.tolist()
    n = len(var_list)

    # Fast path: contiguous [0, n-1] labels (common for generated problems)
    if _is_zero_based_contiguous(all_vars):
        # No remapping needed — direct numpy fill
        h_dense = np.zeros(n, dtype=np.float64)
        h_dense[dict_arrays.h_keys] = dict_arrays.h_values

        J_dense = np.zeros((n, n), dtype=np.float64)
        if dict_arrays.j_rows.size > 0:
            J_dense[dict_arrays.j_rows, dict_arrays.j_cols] = dict_arrays.j_values
            J_dense[dict_arrays.j_cols, dict_arrays.j_rows] = (
                dict_arrays.j_values
            )  # Symmetrize

        return h_dense, J_dense, var_list

    # General path with variable remapping
    var_to_idx = {v: idx for idx, v in enumerate(var_list)}

    h_dense = np.zeros(n, dtype=np.float64)
    h_idx_arr = _remap_indices(dict_arrays.h_keys, var_to_idx)
    h_dense[h_idx_arr] = dict_arrays.h_values

    J_dense = np.zeros((n, n), dtype=np.float64)
    if dict_arrays.j_rows.size > 0:
        rows = _remap_indices(dict_arrays.j_rows, var_to_idx)
        cols = _remap_indices(dict_arrays.j_cols, var_to_idx)
        J_dense[rows, cols] = dict_arrays.j_values
        J_dense[cols, rows] = dict_arrays.j_values  # Symmetrize

    return h_dense, J_dense, var_list


def normalize_ising_to_sparse(
    h: dict[int, float] | NDArray[np.floating[Any]],
    J: dict[tuple[int, int], float] | NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    list[int],
]:
    """Normalize Ising input to sparse CSR format.

    Converts dict or array input to Compressed Sparse Row (CSR) format suitable
    for use with the sparse GPU kernels. This format is efficient for sparse graphs
    where most variable pairs have no interaction.

    Args:
        h: Linear biases. Dict mapping variable indices to values, or 1D array.
        J: Quadratic couplings. Dict mapping (i, j) tuples to values, or 2D array.

    Returns:
        Tuple of (h_dense, row_offsets, col_indices, values, var_labels) where:
            - h_dense: 1D numpy array of linear biases (float64).
            - row_offsets: CSR row pointers array of length [n_spins + 1] (int32).
              row_offsets[i] gives the starting index in col_indices/values for row i.
            - col_indices: CSR column indices array of length [nnz] (int32).
            - values: CSR non-zero values array of length [nnz] (float64).
            - var_labels: List mapping array indices to original variable labels.

    Notes:
        - CSR format stores only non-zero off-diagonal elements of J.
        - Diagonal elements are ignored (should be encoded in h instead).
        - For dict input, this is more efficient as it builds CSR directly without
          creating a dense intermediate representation.
        - For array input, this extracts non-zero elements from the dense matrix.
        - The sparse representation is symmetric: if J[i, j] exists, J[j, i] also exists.
        - Time complexity: O(nnz) for dict input, O(n²) for array input where
          nnz is the number of non-zero elements and n is the number of variables.
        - Space complexity: O(nnz) for CSR arrays plus O(n) for h and metadata.

    Example:
        Dict input (naturally sparse):
        >>> h = {0: -1.0, 1: 1.0, 2: 0.5}
        >>> J = {(0, 1): -0.5, (1, 2): 0.3}  # Only 2 interactions
        >>> h_dense, row_offsets, col_indices, values, var_labels = normalize_ising_to_sparse(h, J)
        >>> print(row_offsets)  # [0, 1, 3, 4] - row 0 has 1 edge, row 1 has 2, row 2 has 1
        >>> print(col_indices)  # [1, 0, 2, 1] - neighbor indices for each row
        >>> print(values)  # [-0.5, -0.5, 0.3, 0.3] - symmetric edge weights

        Dense array converted to sparse (less efficient):
        >>> h = np.array([-1.0, 1.0])
        >>> J = np.array([[0, -0.5], [-0.5, 0]])
        >>> h_dense, row_offsets, col_indices, values, var_labels = normalize_ising_to_sparse(h, J)
    """
    if isinstance(h, dict) and isinstance(J, dict):
        h_dict = cast(dict[int, float], h)
        j_dict = cast(dict[tuple[int, int], float], J)
        return _dict_to_sparse(h_dict, j_dict)

    if isinstance(h, np.ndarray) and isinstance(J, np.ndarray):
        h_array = cast(NDArray[np.floating[Any]], h)
        j_array = cast(NDArray[np.floating[Any]], J)
        return _array_to_sparse(h_array, j_array)

    if isinstance(h, dict):
        raise TypeError("If h is a dict, J must also be a dict")

    if isinstance(h, np.ndarray):
        raise TypeError("If h is a numpy array, J must also be a numpy array")

    raise TypeError(f"h must be a dict or numpy array, got {type(h)}")


def _dict_to_sparse(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    list[int],
]:
    """Convert Ising dict representation to CSR sparse format.

    Builds CSR arrays directly from the dict without creating a dense intermediate
    representation. Symmetric edges (both i→j and j→i) are stored explicitly.
    Duplicate entries for the same (i,j) pair are deduplicated (last-write wins).

    Args:
        h: Linear biases as a dict mapping integer variable index to float value.
        J: Quadratic couplings as a dict mapping (i, j) integer tuple to float value.

    Returns:
        Tuple of (h_dense, row_offsets, col_indices, values, var_labels) as described
        in normalize_ising_to_sparse.

    Notes:
        - Diagonal entries (i == j) are skipped as they are captured in h.
        - Each off-diagonal edge (i,j) produces two CSR entries: i→j and j→i.
        - Adjacency lists are sorted by column index for consistent CSR layout.
        - Time complexity: O(m log m) where m is the number of non-zero J entries.
    """
    dict_arrays = _extract_ising_dict_arrays(h, J)
    all_vars = _sorted_variable_labels(dict_arrays)
    var_list = all_vars.tolist()
    var_to_idx = {v: idx for idx, v in enumerate(var_list)}
    n = len(var_list)

    # Build h
    h_dense = np.zeros(n, dtype=np.float64)
    if dict_arrays.h_keys.size > 0:
        h_indices = (
            dict_arrays.h_keys
            if _is_zero_based_contiguous(all_vars)
            else _remap_indices(dict_arrays.h_keys, var_to_idx)
        )
        h_dense[h_indices] = dict_arrays.h_values

    # Build adjacency list for CSR
    # Store symmetric entries (both i->j and j->i)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    if dict_arrays.j_rows.size > 0:
        rows = (
            dict_arrays.j_rows
            if _is_zero_based_contiguous(all_vars)
            else _remap_indices(dict_arrays.j_rows, var_to_idx)
        )
        cols = (
            dict_arrays.j_cols
            if _is_zero_based_contiguous(all_vars)
            else _remap_indices(dict_arrays.j_cols, var_to_idx)
        )
    else:
        rows = np.empty(0, dtype=np.intp)
        cols = np.empty(0, dtype=np.intp)

    for idx_i, idx_j, coupling in zip(rows, cols, dict_arrays.j_values, strict=True):
        if idx_i != idx_j:  # Skip diagonal
            adjacency[int(idx_i)].append((int(idx_j), float(coupling)))
            adjacency[int(idx_j)].append((int(idx_i), float(coupling)))

    # Remove duplicates and sort by column index (keep last value for same (i,j) pair).
    # `sorted(seen.items())` provides the required column ordering for CSR layout.
    for i in range(n):
        seen: dict[int, float] = {}
        for col, val in adjacency[i]:
            seen[col] = val
        adjacency[i] = [(col, val) for col, val in sorted(seen.items())]

    # Build CSR arrays
    row_offsets = np.zeros(n + 1, dtype=np.int32)
    col_list: list[int] = []
    val_list: list[float] = []

    for i in range(n):
        row_offsets[i + 1] = row_offsets[i] + len(adjacency[i])
        for col, val in adjacency[i]:
            col_list.append(col)
            val_list.append(val)

    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float64)

    return h_dense, row_offsets, col_indices, values, var_list


def _array_to_sparse(
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    list[int],
]:
    """Convert dense array Ising representation to CSR sparse format.

    Extracts non-zero off-diagonal elements from the (symmetrized) dense J matrix
    row by row to build CSR arrays. The input is first symmetrized: J_sym = (J + J.T) / 2.

    Args:
        h: Linear biases as a 1D numpy array of shape (n,).
        J: Quadratic couplings as a 2D numpy array of shape (n, n).

    Returns:
        Tuple of (h_dense, row_offsets, col_indices, values, var_labels) as described
        in normalize_ising_to_sparse.

    Notes:
        For truly sparse problems, prefer using dict input (via _dict_to_sparse) which
        builds CSR directly without creating a dense intermediate representation.
        This function processes a dense matrix and is O(n²) regardless of sparsity.
        - Diagonal elements are excluded (dimod convention encodes diagonals in h).
        - Only strictly non-zero entries (value != 0.0) are included.
    """
    h_dense = _as_float64_vector(h)
    J_sym = _symmetrize_dense_matrix(J)

    n = len(h_dense)
    var_labels = list(range(n))

    # Build CSR from dense J iterating row-by-row to avoid n×n mask allocation
    row_offsets = np.zeros(n + 1, dtype=np.int32)
    col_list: list[int] = []
    val_list: list[float] = []

    for i in range(n):
        row = J_sym[i]
        # Find non-zero off-diagonal elements in this row
        for j in range(n):
            if i != j and row[j] != 0.0:
                col_list.append(j)
                val_list.append(row[j])
        row_offsets[i + 1] = len(col_list)

    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float64)

    return h_dense, row_offsets, col_indices, values, var_labels
