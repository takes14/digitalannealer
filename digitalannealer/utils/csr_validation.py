"""CSR structure validation helpers for sparse kernels."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_csr_structure(
    row_offsets: NDArray[np.int32],
    col_indices: NDArray[np.int32],
    n_spins: int,
    values: NDArray[np.floating[Any]] | None = None,
) -> None:
    """Validate CSR structure invariants required by sparse CUDA kernels.

    Checks:
      - ``row_offsets[0] == 0``
      - ``row_offsets`` is monotonically non-decreasing
      - All ``col_indices`` values are in ``[0, n_spins)``
      - No diagonal entries (``col_indices[k] != row_index_of_k`` for all ``k``)
      - No duplicate column indices within any row
      - Symmetry contract: each edge ``(i, j)`` must have reverse ``(j, i)``
      - When ``values`` is provided, reverse edges must have equal values

    This function is automatically satisfied when CSR data was produced by
    :func:`normalize_ising_to_sparse`.  Call it explicitly when CSR arrays are
    provided from an external source and bypass the normaliser.

    Args:
        row_offsets: CSR row pointers array of length ``n_spins + 1`` (int32).
        col_indices: CSR column indices array of length ``nnz`` (int32).
        n_spins: Number of spins (== number of rows == number of columns).
        values: Optional CSR values array of length ``nnz``. When provided,
            symmetry check also enforces exact reverse-edge value equality.

    Raises:
        ValueError: On any CSR structure violation.

    Example:
        >>> import numpy as np
        >>> from digitalannealer.utils.validation import validate_csr_structure
        >>> row_offsets = np.array([0, 1, 2], dtype=np.int32)  # 2-spin chain
        >>> col_indices = np.array([1, 0], dtype=np.int32)
        >>> validate_csr_structure(row_offsets, col_indices, n_spins=2)  # passes
    """
    if len(row_offsets) != n_spins + 1:
        raise ValueError(
            f"row_offsets must have length n_spins+1={n_spins + 1}, got {len(row_offsets)}"
        )

    if int(row_offsets[0]) != 0:
        raise ValueError(f"row_offsets[0] must be 0, got {row_offsets[0]}")

    diffs = np.diff(row_offsets)
    if (diffs < 0).any():
        bad = int(np.argmax(diffs < 0))
        raise ValueError(
            f"row_offsets must be non-decreasing: "
            f"row_offsets[{bad}]={row_offsets[bad]} > "
            f"row_offsets[{bad + 1}]={row_offsets[bad + 1]}"
        )

    nnz_declared = int(row_offsets[-1])
    if nnz_declared != len(col_indices):
        raise ValueError(
            f"row_offsets[-1]={nnz_declared} must equal len(col_indices)={len(col_indices)}"
        )

    if len(col_indices) == 0:
        return

    col_min = int(col_indices.min())
    col_max = int(col_indices.max())
    if col_min < 0 or col_max >= n_spins:
        bad_mask = (col_indices < 0) | (col_indices >= n_spins)
        bad_idx = int(np.argmax(bad_mask))
        raise ValueError(
            f"col_indices[{bad_idx}] = {int(col_indices[bad_idx])} is out of range [0, {n_spins})"
        )

    # Vectorised diagonal check: build per-entry row index using repeat.
    # O(nnz) total, no Python loop over rows.
    nnz = int(row_offsets[-1])
    if values is not None and len(values) != nnz:
        raise ValueError(f"values length must match nnz={nnz}, got {len(values)}")
    row_lengths = diffs.astype(np.int32)
    row_of_k = np.repeat(np.arange(n_spins, dtype=np.int32), row_lengths)
    diagonal_mask = row_of_k == col_indices[:nnz]
    if diagonal_mask.any():
        k = int(np.argmax(diagonal_mask))
        i = int(row_of_k[k])
        raise ValueError(
            f"CSR must not contain diagonal entries: diagonal found at row {i} "
            f"(col_indices[{k}] = {int(col_indices[k])})"
        )

    for i in range(n_spins):
        row_start = int(row_offsets[i])
        row_end = int(row_offsets[i + 1])
        first_seen_col_index: dict[int, int] = {}
        for k in range(row_start, row_end):
            c = int(col_indices[k])
            if c in first_seen_col_index:
                raise ValueError(
                    "CSR must not contain duplicate column indices within a row: "
                    f"row {i} has duplicate col {c} at col_indices[{k}] "
                    f"(first seen at col_indices[{first_seen_col_index[c]}])"
                )
            first_seen_col_index[c] = k

    if values is None:
        edge_set = {(int(row_of_k[k]), int(col_indices[k])) for k in range(nnz)}
        for i, j in edge_set:
            if (j, i) not in edge_set:
                raise ValueError(
                    f"CSR must be symmetric: missing reverse edge for ({i}, {j})"
                )
    else:
        edge_values: dict[tuple[int, int], float] = {}
        for k in range(nnz):
            edge_values[(int(row_of_k[k]), int(col_indices[k]))] = float(values[k])

        for (i, j), value in edge_values.items():
            reverse = (j, i)
            if reverse not in edge_values:
                raise ValueError(
                    f"CSR must be symmetric: missing reverse edge for ({i}, {j})"
                )
            if edge_values[reverse] != value:
                raise ValueError(
                    "CSR must be symmetric: value mismatch between "
                    f"({i}, {j})={value} and ({j}, {i})={edge_values[reverse]}"
                )
