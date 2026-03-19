"""Shared problem generators for benchmarks.

This module provides common functions for generating Ising problems
with different structures (dense, sparse) for use across benchmark scripts.
"""

import numpy as np


def generate_dense_problem_dict(
    n_spins: int, seed: int = 42
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Generate a fully-connected random Ising problem in dict format.

    Args:
        n_spins: Number of spins in the problem.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (h, J) in dict format for OpenJij compatibility.
    """
    rng = np.random.default_rng(seed)

    # Random linear biases
    h = {i: float(rng.uniform(-1.0, 1.0)) for i in range(n_spins)}

    # Random symmetric couplings (upper triangular to avoid duplicates)
    J = {}
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            J[(i, j)] = float(rng.uniform(-1.0, 1.0))

    return h, J


def generate_dense_problem_array(n_spins: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fully-connected random Ising problem in array format.

    Creates a dense interaction matrix where all spin pairs have non-zero
    couplings, representing the worst-case scenario for memory usage.

    Args:
        n_spins: Number of spins in the problem.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (h, J) where:
            h: Linear biases, shape (n_spins,)
            J: Quadratic couplings, shape (n_spins, n_spins), symmetric
    """
    rng = np.random.default_rng(seed)

    # Random linear biases
    h = rng.uniform(-1.0, 1.0, size=n_spins).astype(np.float32)

    # Random symmetric coupling matrix
    J = rng.uniform(-1.0, 1.0, size=(n_spins, n_spins)).astype(np.float32)
    J = (J + J.T) / 2  # Symmetrize
    np.fill_diagonal(J, 0.0)  # Zero diagonal

    return h, J


def generate_sparse_problem_ring(
    n_spins: int, seed: int = 42
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Generate a ring (1D periodic chain) Ising problem.

    Each spin connects only to its neighbors, creating a very sparse graph
    with exactly N edges.

    Args:
        n_spins: Number of spins in the ring.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (h, J) in dict format.
    """
    rng = np.random.default_rng(seed)

    h = {i: float(rng.uniform(-1.0, 1.0)) for i in range(n_spins)}
    J = {(i, (i + 1) % n_spins): float(rng.uniform(-1.0, 1.0)) for i in range(n_spins)}

    return h, J


def generate_sparse_problem_2d_lattice(
    n_side: int, seed: int = 42
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Generate a 2D square lattice Ising problem with periodic boundaries.

    Creates a sparse graph where each spin has exactly 4 neighbors,
    typical of 2D lattice physics simulations.

    Args:
        n_side: Side length of the square lattice (total spins = n_side²).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (h, J) in dict format.
    """
    rng = np.random.default_rng(seed)
    n_spins = n_side * n_side

    h = {i: float(rng.uniform(-1.0, 1.0)) for i in range(n_spins)}
    J = {}

    for row in range(n_side):
        for col in range(n_side):
            i = row * n_side + col

            # Right neighbor (with periodic boundary)
            j_right = row * n_side + (col + 1) % n_side
            edge_right = (min(i, j_right), max(i, j_right))
            if edge_right not in J:
                J[edge_right] = float(rng.uniform(-1.0, 1.0))

            # Down neighbor (with periodic boundary)
            j_down = ((row + 1) % n_side) * n_side + col
            edge_down = (min(i, j_down), max(i, j_down))
            if edge_down not in J:
                J[edge_down] = float(rng.uniform(-1.0, 1.0))

    return h, J


def generate_sparse_problem_random(
    n_spins: int, connectivity: int = 4, seed: int = 42
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Generate a random sparse Ising problem with fixed average connectivity.

    Creates a random graph where each spin has approximately `connectivity`
    neighbors on average.

    Args:
        n_spins: Number of spins.
        connectivity: Average number of neighbors per spin.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (h, J) in dict format.
    """
    rng = np.random.default_rng(seed)

    h = {i: float(rng.uniform(-1.0, 1.0)) for i in range(n_spins)}
    J = {}

    # Number of edges to create
    n_edges = int(n_spins * connectivity / 2)

    # Generate random edges
    edge_count = 0
    max_attempts = n_edges * 10  # Prevent infinite loop
    attempts = 0

    while edge_count < n_edges and attempts < max_attempts:
        i = rng.integers(0, n_spins)
        j = rng.integers(0, n_spins)
        if i != j:
            edge = (min(i, j), max(i, j))
            if edge not in J:
                J[edge] = float(rng.uniform(-1.0, 1.0))
                edge_count += 1
        attempts += 1

    return h, J
