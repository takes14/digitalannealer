"""Energy computation utilities for Ising model.

This module contains shared energy computation functions used by all samplers.
These functions compute delta_E for spin flips and total Ising energy.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Minimum value for random numbers to avoid log(0) in Metropolis acceptance.
# curand_uniform returns (0, 1], but numpy.random.random returns [0, 1)
# so we clamp to this small positive value for numerical stability.
_MIN_RANDOM_VALUE = 1e-30


def compute_all_delta_e(
    spins: NDArray[np.int8],
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Compute delta_E for flipping each spin.

    Uses dimod convention: E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
    delta_E_i = -2 * s_i * (h_i + sum_j J_ij s_j)

    Args:
        spins: Current spin configuration as 1D array of +1/-1 values.
        h: Linear biases as 1D array.
        J: Quadratic couplings as 2D symmetric array.

    Returns:
        1D array of delta_E values for flipping each spin.
    """
    # dimod convention ignores J_ii in the quadratic term, so remove diagonal
    # contribution from the local field used for flip delta_E.
    local_field = h + J @ spins - np.diag(J) * spins
    return -2.0 * spins * local_field


def update_delta_e(
    delta_e: NDArray[np.float64],
    spins: NDArray[np.int8],
    J: NDArray[np.floating[Any]],
    flipped_idx: int,
) -> None:
    """Update delta_E after flipping spin at flipped_idx (differential update).

    After flipping spin k: delta_E_i_new = delta_E_i_old - 4 * s_k * s_i * J_ki
    Note: s_k is already flipped in the spins array when this is called.

    Args:
        delta_e: Current delta_E array to update in-place.
        spins: Current spin configuration (with spin at flipped_idx already flipped).
        J: Quadratic couplings as 2D array.
        flipped_idx: Index of the spin that was flipped.
    """
    s_k = spins[flipped_idx]
    # dimod convention ignores diagonal terms; only off-diagonal couplings
    # contribute to differential updates.
    row = J[flipped_idx, :].copy()
    row[flipped_idx] = 0.0
    delta_e -= 4 * s_k * spins * row
    delta_e[flipped_idx] = -delta_e[flipped_idx]


def compute_energy(
    spins: NDArray[np.int8],
    h: NDArray[np.floating[Any]],
    J: NDArray[np.floating[Any]],
) -> float:
    """Compute the Ising energy of a configuration.

    Uses dimod convention: E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j

    The quadratic form 0.5 * spins @ J @ spins includes diagonal terms
    (0.5 * sum_i J_ii * s_i^2 = 0.5 * sum_i J_ii since s_i^2 = 1),
    which must be subtracted to match the dimod convention that only
    counts off-diagonal interactions.

    Args:
        spins: Spin configuration as 1D array of +1/-1 values.
        h: Linear biases as 1D array.
        J: Quadratic couplings as 2D symmetric array.

    Returns:
        Total Ising energy as a float.
    """
    linear_term = np.dot(h, spins)
    # 0.5 * spins @ J @ spins includes 0.5 * sum_i J_ii; subtract it to drop diagonal terms
    quadratic_term = 0.5 * spins @ J @ spins - 0.5 * np.trace(J)
    return float(linear_term + quadratic_term)
