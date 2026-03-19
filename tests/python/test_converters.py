"""Tests for QUBO/Ising converters."""

import numpy as np
import pytest

from digitalannealer.models.converters import ising_to_qubo, qubo_to_ising


class TestQuboToIsing:
    """Tests for QUBO to Ising conversion."""

    def test_diagonal_only(self):
        """Test conversion with only diagonal terms."""
        Q = {(0, 0): -2.0, (1, 1): 4.0}
        h, J, offset = qubo_to_ising(Q)

        assert len(h) == 2
        assert J.shape == (2, 2)

        # h_i = Q_ii/2 + sum_j Q_ij/2 = Q_ii/2 (no off-diag, dimod convention)
        assert np.isclose(h[0], (-2.0) / 2)  # = -1.0
        assert np.isclose(h[1], (4.0) / 2)  # = 2.0
        # offset = sum_i Q_ii/2
        assert np.isclose(offset, (-2.0 + 4.0) / 2)  # = 1.0

    def test_off_diagonal(self):
        """Test conversion with off-diagonal terms."""
        Q = {(0, 1): 4.0}  # Only coupling, no bias
        h, J, offset = qubo_to_ising(Q)

        # dimod QUBO convention counts the (0,1) interaction once in sum_{i<=j} Q_ij x_i x_j.
        # For Q_01 = 4:
        #   J_01 = Q_01/4 = 1
        #   h_0 = h_1 = Q_01/4 = 1
        #   offset += Q_01/4 = 1
        assert np.isclose(J[0, 1], 4.0 / 4)  # = 1.0
        assert np.isclose(J[1, 0], 4.0 / 4)  # Symmetric
        assert np.isclose(h[0], 4.0 / 4)  # = 1.0
        assert np.isclose(h[1], 4.0 / 4)  # = 1.0
        assert np.isclose(offset, 4.0 / 4)  # = 1.0

    def test_off_diagonal_duplicate_entries_equivalent(self):
        """(i,j) and (j,i) are canonicalized and summed before conversion."""
        Q_single = {(0, 1): 4.0}
        Q_duplicate = {(0, 1): 4.0, (1, 0): 4.0}
        Q_asymmetric = {(0, 1): 2.0, (1, 0): 4.0}

        h_single, J_single, offset_single = qubo_to_ising(Q_single)
        h_duplicate, J_duplicate, offset_duplicate = qubo_to_ising(Q_duplicate)
        h_asym, J_asym, offset_asym = qubo_to_ising(Q_asymmetric)

        # Duplicates are summed after canonicalization
        h_sum, J_sum, offset_sum = qubo_to_ising({(0, 1): 8.0})
        assert np.allclose(h_duplicate, h_sum)
        assert np.allclose(J_duplicate, J_sum)
        assert np.isclose(offset_duplicate, offset_sum)
        assert not np.allclose(h_single, h_duplicate)
        assert not np.isclose(offset_single, offset_duplicate)

        h_asym_sum, J_asym_sum, offset_asym_sum = qubo_to_ising({(0, 1): 6.0})
        assert np.allclose(h_asym, h_asym_sum)
        assert np.allclose(J_asym, J_asym_sum)
        assert np.isclose(offset_asym, offset_asym_sum)

    def test_array_input(self):
        """Test conversion with numpy array input."""
        Q = np.array([[1.0, 2.0], [0.0, 3.0]])  # Upper triangular
        h, J, offset = qubo_to_ising(Q)

        assert len(h) == 2
        assert J.shape == (2, 2)
        # Should symmetrize the off-diagonal
        assert np.isclose(J[0, 1], J[1, 0])

    def test_empty_qubo_rejected(self):
        """Test that empty QUBO raises error."""
        with pytest.raises(ValueError):
            qubo_to_ising({})

    def test_energy_preservation(self):
        """Test that conversion preserves energy."""
        Q = {(0, 0): -1.0, (1, 1): 2.0, (0, 1): 3.0}
        h, J, offset = qubo_to_ising(Q)

        # Build an upper-triangular Q matrix for energy calculation.
        # This matches the dimod QUBO convention E = sum_{i<=j} Q_ij x_i x_j.
        Q_matrix = np.zeros((2, 2))
        Q_matrix[0, 0] = -1.0
        Q_matrix[1, 1] = 2.0
        Q_matrix[0, 1] = 3.0

        # Test all 4 binary configurations
        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1])
                s = 2 * x - 1

                # QUBO energy: E = x^T Q x
                qubo_energy = x @ Q_matrix @ x

                # Ising energy
                ising_energy = np.dot(h, s) + 0.5 * s @ J @ s

                # QUBO energy should equal Ising energy + offset
                assert np.isclose(qubo_energy, ising_energy + offset, atol=1e-10), (
                    f"Energy mismatch for x=({x0},{x1}): QUBO={qubo_energy}, Ising+offset={ising_energy + offset}"
                )

    def test_energy_preservation_with_duplicate_off_diagonal(self):
        """Duplicate off-diagonal entries are summed in the effective interaction."""
        Q = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 4.0}
        h, J, offset = qubo_to_ising(Q)

        # Use a matrix that represents the same energy as the provided dict under x^T Q x
        # (i.e., Q[0,1]=2 and Q[1,0]=4 contributes 6*x0*x1).
        Q_matrix = np.array([[1.0, 2.0], [4.0, 0.0]])

        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1])
                s = 2 * x - 1

                qubo_energy = x @ Q_matrix @ x
                ising_energy = np.dot(h, s) + 0.5 * s @ J @ s

                assert np.isclose(qubo_energy, ising_energy + offset, atol=1e-10), (
                    f"Energy mismatch for x=({x0},{x1}): QUBO={qubo_energy}, "
                    f"Ising+offset={ising_energy + offset}"
                )

    def test_nan_in_qubo_dict_rejected(self):
        """Test that NaN values in QUBO dict are rejected."""
        Q = {(0, 0): np.nan, (1, 1): 2.0}
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_inf_in_qubo_dict_rejected(self):
        """Test that Inf values in QUBO dict are rejected."""
        Q = {(0, 0): 1.0, (0, 1): np.inf}
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_neg_inf_in_qubo_dict_rejected(self):
        """Test that -Inf values in QUBO dict are rejected."""
        Q = {(0, 0): -np.inf, (1, 1): 2.0}
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_nan_in_qubo_array_rejected(self):
        """Test that NaN values in QUBO array are rejected."""
        Q = np.array([[np.nan, 2.0], [0.0, 3.0]])
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_inf_in_qubo_array_rejected(self):
        """Test that Inf values in QUBO array are rejected."""
        Q = np.array([[1.0, np.inf], [0.0, 3.0]])
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_neg_inf_in_qubo_array_rejected(self):
        """Test that -Inf values in QUBO array are rejected."""
        Q = np.array([[1.0, 2.0], [0.0, -np.inf]])
        with pytest.raises(ValueError, match="non-finite value"):
            qubo_to_ising(Q)

    def test_integer_array_input(self):
        """Test conversion with integer array input."""
        Q = np.array([[1, 2], [2, 3]], dtype=int)
        h, J, offset = qubo_to_ising(Q)

        assert len(h) == 2
        assert J.shape == (2, 2)
        # Should work just like float arrays
        assert h.dtype == np.float64
        assert J.dtype == np.float64


class TestIsingToQubo:
    """Tests for Ising to QUBO conversion."""

    def test_simple_ising(self):
        """Test simple Ising to QUBO conversion."""
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 0.5], [0.5, 0.0]])

        Q, offset = ising_to_qubo(h, J)

        assert Q.shape == (2, 2)
        assert isinstance(offset, float)

    def test_dict_input(self):
        """Test Ising to QUBO conversion with dict input."""
        h = {0: 1.0, 1: -1.0}
        J = {(0, 1): 0.5}

        Q, offset = ising_to_qubo(h, J)

        assert Q.shape == (2, 2)
        assert isinstance(offset, float)

    def test_asymmetric_array_input_matches_explicit_symmetrization(self):
        """Asymmetric dense J should match conversion of its symmetrized form."""
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 0.25], [0.75, 0.0]])

        Q_asym, offset_asym = ising_to_qubo(h, J)
        Q_sym, offset_sym = ising_to_qubo(h, (J + J.T) / 2.0)

        assert np.allclose(Q_asym, Q_sym)
        assert np.isclose(offset_asym, offset_sym)

    def test_roundtrip(self):
        """Test QUBO -> Ising -> QUBO roundtrip preserves energy."""
        Q_orig = np.array([[1.0, 2.0], [2.0, 3.0]])
        Q_orig = (Q_orig + Q_orig.T) / 2  # Symmetrize

        h, J, offset1 = qubo_to_ising(Q_orig)

        # Verify energy is preserved for all configurations
        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1])
                s = 2 * x - 1

                qubo_energy = x @ Q_orig @ x
                ising_energy = np.dot(h, s) + 0.5 * s @ J @ s

                assert np.isclose(qubo_energy, ising_energy + offset1, atol=1e-10), (
                    f"Energy mismatch at x={x}: QUBO={qubo_energy}, Ising+offset={ising_energy + offset1}"
                )

    def test_nan_in_h_rejected(self):
        """Test that NaN values in h raise ValueError."""
        h = np.array([np.nan, 1.0])
        J = np.array([[0.0, 0.5], [0.5, 0.0]])
        with pytest.raises(ValueError):
            ising_to_qubo(h, J)

    def test_inf_in_j_rejected(self):
        """Test that Inf values in J raise ValueError."""
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, np.inf], [np.inf, 0.0]])
        with pytest.raises(ValueError):
            ising_to_qubo(h, J)

    def test_mismatched_types_rejected(self):
        """Test that mismatched h/J types raise TypeError."""
        h_dict = {0: 1.0}
        J_array = np.array([[0.0]])
        with pytest.raises(TypeError):
            ising_to_qubo(h_dict, J_array)

    def test_mismatched_dimensions_rejected(self):
        """Test that mismatched h/J dimensions raise ValueError."""
        h = np.array([1.0, -1.0])  # 2 spins
        J = np.array([[0.0]])  # 1x1
        with pytest.raises(ValueError):
            ising_to_qubo(h, J)
