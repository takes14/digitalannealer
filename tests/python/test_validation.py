"""Validation tests for Digital Annealer utilities."""

import numpy as np
import pytest

from digitalannealer.utils.validation import (
    normalize_ising_to_dense,
    normalize_ising_to_sparse,
    validate_ising_input,
)


def test_validate_ising_dict_rejects_non_finite_bias():
    """Dict h containing NaN should be rejected."""
    h = {0: np.nan}
    J = {(0, 0): 0.0}

    with pytest.raises(ValueError, match="non-finite value"):
        validate_ising_input(h, J)


def test_validate_ising_dict_rejects_non_finite_coupling():
    """Dict J containing Inf should be rejected."""
    h = {0: 0.0, 1: 0.0}
    J = {(0, 1): np.inf}

    with pytest.raises(ValueError, match="non-finite value"):
        validate_ising_input(h, J)


def test_normalize_dense_rejects_mixed_input_types():
    """Dense normalizer should reject mixed dict/array input types explicitly."""
    with pytest.raises(TypeError, match="If h is a dict, J must also be a dict"):
        normalize_ising_to_dense({0: 0.0}, np.zeros((1, 1)))


def test_normalize_sparse_rejects_mixed_input_types():
    """Sparse normalizer should reject mixed dict/array input types explicitly."""
    with pytest.raises(TypeError, match="If h is a dict, J must also be a dict"):
        normalize_ising_to_sparse({0: 0.0}, np.zeros((1, 1)))
