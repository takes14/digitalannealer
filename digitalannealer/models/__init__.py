"""Model utilities for Digital Annealer.

Provides conversion functions between the QUBO and Ising problem
representations that are used throughout the library.

Functions
---------
qubo_to_ising
    Convert a QUBO matrix (dict or 2-D array) to Ising ``(h, J, offset)``.
ising_to_qubo
    Convert Ising ``(h, J)`` parameters to a dense QUBO matrix.
"""

from digitalannealer.models.converters import ising_to_qubo, qubo_to_ising

__all__ = ["qubo_to_ising", "ising_to_qubo"]
