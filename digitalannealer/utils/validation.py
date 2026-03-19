"""Compatibility facade for validation/normalization utilities.

This module preserves the existing public import surface while delegating
implementations to focused modules.
"""

from digitalannealer.utils.csr_validation import validate_csr_structure
from digitalannealer.utils.normalization import (
    normalize_ising_to_dense,
    normalize_ising_to_sparse,
)
from digitalannealer.utils.validation_input import (
    validate_dtype,
    validate_ising_input,
    validate_qubo_input,
)

__all__ = [
    "validate_ising_input",
    "validate_qubo_input",
    "validate_dtype",
    "normalize_ising_to_dense",
    "normalize_ising_to_sparse",
    "validate_csr_structure",
]
