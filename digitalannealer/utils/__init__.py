"""Utility functions for Digital Annealer.

This package re-exports all public validation, normalisation and energy
helpers so that callers can import them directly from ``digitalannealer.utils``
without navigating the internal submodule structure.

Modules
-------
validation
    Input validation and normalisation (Ising/QUBO dict/array → dense/CSR).
energy
    Shared energy and delta-E computation helpers used by CPU samplers.
"""

from digitalannealer.utils.energy import (
    compute_all_delta_e,
    compute_energy,
    update_delta_e,
)
from digitalannealer.utils.validation import (
    normalize_ising_to_dense,
    normalize_ising_to_sparse,
    validate_csr_structure,
    validate_ising_input,
    validate_qubo_input,
)

__all__ = [
    "validate_ising_input",
    "validate_qubo_input",
    "normalize_ising_to_dense",
    "normalize_ising_to_sparse",
    "validate_csr_structure",
    "compute_all_delta_e",
    "update_delta_e",
    "compute_energy",
]
