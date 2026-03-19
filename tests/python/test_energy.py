"""Tests for energy computation utilities."""

import numpy as np

from digitalannealer.utils.energy import (
    compute_all_delta_e,
    compute_energy,
    update_delta_e,
)


class TestComputeEnergy:
    """Tests for compute_energy function."""

    def test_simple_ferromagnetic(self):
        """Test energy computation for simple ferromagnetic system."""
        # Two spins with ferromagnetic coupling
        h = np.array([0.0, 0.0])
        J = np.array([[0.0, -1.0], [-1.0, 0.0]])

        # Aligned spins: E = 0 + 0.5 * (-1) * 2 = -1.0
        spins_aligned = np.array([1, 1], dtype=np.int8)
        energy = compute_energy(spins_aligned, h, J)
        assert np.isclose(energy, -1.0)

        # Anti-aligned spins: E = 0 + 0.5 * (1) * 2 = 1.0
        spins_anti = np.array([1, -1], dtype=np.int8)
        energy = compute_energy(spins_anti, h, J)
        assert np.isclose(energy, 1.0)

    def test_with_linear_bias(self):
        """Test energy computation with linear biases."""
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 0.0], [0.0, 0.0]])

        # E = h_0 * s_0 + h_1 * s_1 = 1*1 + (-1)*1 = 0
        spins = np.array([1, 1], dtype=np.int8)
        energy = compute_energy(spins, h, J)
        assert np.isclose(energy, 0.0)

        # E = 1*(-1) + (-1)*(-1) = -1 + 1 = 0
        spins = np.array([-1, -1], dtype=np.int8)
        energy = compute_energy(spins, h, J)
        assert np.isclose(energy, 0.0)

        # E = 1*1 + (-1)*(-1) = 1 + 1 = 2
        spins = np.array([1, -1], dtype=np.int8)
        energy = compute_energy(spins, h, J)
        assert np.isclose(energy, 2.0)

    def test_larger_system(self):
        """Test energy computation for a 3-spin chain."""
        h = np.array([0.5, 0.0, -0.5])
        J = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 0.0]])

        # All aligned (+1, +1, +1):
        # Linear: 0.5*1 + 0.0*1 + (-0.5)*1 = 0.5 + 0 - 0.5 = 0
        # Quadratic: 0.5 * (J[0,1] + J[1,0] + J[1,2] + J[2,1]) = 0.5 * (-1 - 1 - 1 - 1) = -2
        # Total: 0 + (-2) = -2
        spins = np.array([1, 1, 1], dtype=np.int8)
        energy = compute_energy(spins, h, J)
        expected = 0.5 + 0.0 - 0.5 + 0.5 * (-1 - 1 - 1 - 1)  # linear + quadratic
        assert np.isclose(energy, expected)


class TestComputeAllDeltaE:
    """Tests for compute_all_delta_e function."""

    def test_simple_system(self):
        """Test delta_E computation for simple system."""
        h = np.array([1.0, 0.0])
        J = np.array([[0.0, -1.0], [-1.0, 0.0]])
        spins = np.array([1, 1], dtype=np.int8)

        delta_e = compute_all_delta_e(spins, h, J)

        # For spin 0: delta_E_0 = -2 * s_0 * (h_0 + J @ spins[0])
        # local_field[0] = h[0] + J[0,:] @ spins = 1.0 + (-1) = 0.0
        # delta_E_0 = -2 * 1 * 0 = 0
        assert np.isclose(delta_e[0], 0.0)

        # For spin 1: local_field[1] = h[1] + J[1,:] @ spins = 0 + (-1) = -1.0
        # delta_E_1 = -2 * 1 * (-1) = 2.0
        assert np.isclose(delta_e[1], 2.0)

    def test_energy_consistency(self):
        """Test that flipping a spin changes energy by delta_E."""
        h = np.array([0.5, -0.3, 0.2])
        J = np.array([[0.0, 0.7, -0.4], [0.7, 0.0, 0.3], [-0.4, 0.3, 0.0]])
        spins = np.array([1, -1, 1], dtype=np.int8)

        delta_e = compute_all_delta_e(spins, h, J)

        # Flip each spin and verify energy change matches delta_E
        for i in range(len(spins)):
            original_energy = compute_energy(spins, h, J)
            flipped_spins = spins.copy()
            flipped_spins[i] *= -1
            new_energy = compute_energy(flipped_spins, h, J)

            actual_change = new_energy - original_energy
            assert np.isclose(delta_e[i], actual_change, atol=1e-10), (
                f"delta_E[{i}]={delta_e[i]} does not match actual change={actual_change}"
            )

    def test_energy_consistency_with_diagonal_terms(self):
        """delta_E must match energy differences even when J has diagonal entries."""
        h = np.array([0.2, -0.3, 0.1])
        J = np.array(
            [
                [1.5, 0.4, -0.2],
                [0.4, -0.7, 0.3],
                [-0.2, 0.3, 0.9],
            ]
        )
        spins = np.array([1, -1, 1], dtype=np.int8)

        delta_e = compute_all_delta_e(spins, h, J)

        for i in range(len(spins)):
            original_energy = compute_energy(spins, h, J)
            flipped_spins = spins.copy()
            flipped_spins[i] *= -1
            new_energy = compute_energy(flipped_spins, h, J)
            actual_change = new_energy - original_energy
            assert np.isclose(delta_e[i], actual_change, atol=1e-10), (
                f"delta_E[{i}]={delta_e[i]} does not match actual change={actual_change}"
            )


class TestUpdateDeltaE:
    """Tests for update_delta_e function."""

    def test_differential_update_correctness(self):
        """Test that differential update gives correct delta_E."""
        h = np.array([0.5, -0.3, 0.2, 0.1])
        J = np.array(
            [
                [0.0, 0.7, -0.4, 0.2],
                [0.7, 0.0, 0.3, -0.1],
                [-0.4, 0.3, 0.0, 0.5],
                [0.2, -0.1, 0.5, 0.0],
            ]
        )
        spins = np.array([1, -1, 1, -1], dtype=np.int8)

        # Compute initial delta_E
        delta_e = compute_all_delta_e(spins, h, J)

        # Flip spin 1
        flip_idx = 1
        spins[flip_idx] *= -1

        # Update delta_E differentially
        update_delta_e(delta_e, spins, J, flip_idx)

        # Compute delta_E from scratch for comparison
        expected_delta_e = compute_all_delta_e(spins, h, J)

        np.testing.assert_allclose(
            delta_e,
            expected_delta_e,
            atol=1e-10,
            err_msg="Differential update does not match recomputation",
        )

    def test_multiple_flips(self):
        """Test that multiple differential updates remain correct."""
        np.random.seed(42)
        n_spins = 5
        h = np.random.randn(n_spins)
        J = np.random.randn(n_spins, n_spins)
        J = (J + J.T) / 2  # Symmetrize
        np.fill_diagonal(J, 0)

        spins = np.random.choice([-1, 1], size=n_spins).astype(np.int8)
        delta_e = compute_all_delta_e(spins, h, J)

        # Perform several flips and verify consistency
        for _ in range(10):
            flip_idx = np.random.randint(n_spins)
            spins[flip_idx] *= -1
            update_delta_e(delta_e, spins, J, flip_idx)

            expected_delta_e = compute_all_delta_e(spins, h, J)
            np.testing.assert_allclose(
                delta_e,
                expected_delta_e,
                atol=1e-10,
                err_msg="Differential update diverged from recomputation",
            )

    def test_differential_update_with_diagonal_terms(self):
        """Differential update remains correct when J has diagonal entries."""
        h = np.array([0.2, -0.3, 0.1, -0.4])
        J = np.array(
            [
                [1.1, 0.7, -0.4, 0.2],
                [0.7, -0.8, 0.3, -0.1],
                [-0.4, 0.3, 0.6, 0.5],
                [0.2, -0.1, 0.5, -1.3],
            ]
        )
        spins = np.array([1, -1, 1, -1], dtype=np.int8)

        delta_e = compute_all_delta_e(spins, h, J)
        flip_idx = 2
        spins[flip_idx] *= -1
        update_delta_e(delta_e, spins, J, flip_idx)

        expected_delta_e = compute_all_delta_e(spins, h, J)
        np.testing.assert_allclose(
            delta_e,
            expected_delta_e,
            atol=1e-10,
            err_msg="Differential update does not match recomputation with diagonal terms",
        )


class TestEnergyUtilsEdgeCases:
    """Tests for edge cases in energy utilities."""

    def test_single_spin(self):
        """Test with a single spin system."""
        h = np.array([2.0])
        J = np.array([[0.0]])
        spins = np.array([1], dtype=np.int8)

        energy = compute_energy(spins, h, J)
        assert np.isclose(energy, 2.0)

        delta_e = compute_all_delta_e(spins, h, J)
        assert np.isclose(delta_e[0], -4.0)  # Flipping gives E = -2, so delta = -4

    def test_zero_couplings(self):
        """Test with zero coupling matrix."""
        h = np.array([1.0, -1.0, 0.5])
        J = np.zeros((3, 3))
        spins = np.array([1, 1, -1], dtype=np.int8)

        energy = compute_energy(spins, h, J)
        expected = 1.0 + (-1.0) + (-0.5)
        assert np.isclose(energy, expected)

    def test_large_values(self):
        """Test numerical stability with large values."""
        h = np.array([1e6, -1e6])
        J = np.array([[0.0, 1e5], [1e5, 0.0]])
        spins = np.array([1, 1], dtype=np.int8)

        energy = compute_energy(spins, h, J)
        # Should not overflow or have numerical issues
        assert np.isfinite(energy)

        delta_e = compute_all_delta_e(spins, h, J)
        assert np.all(np.isfinite(delta_e))

    def test_energy_with_diagonal_terms(self):
        """Test that energy calculation correctly handles non-zero diagonal terms in J.

        Diagonal terms in J should NOT contribute to energy since the dimod convention
        only counts off-diagonal interactions: E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
        """
        h = np.array([1.0, -1.0])
        # J with non-zero diagonal terms
        J = np.array([[0.5, 0.5], [0.5, -0.3]])

        # Symmetrize J
        J_sym = (J + J.T) / 2.0

        spins = np.array([1, 1], dtype=np.int8)

        # Compute energy using the utility
        energy = compute_energy(spins, h, J_sym)

        # Manually compute energy using dimod convention (no diagonal contribution)
        # E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
        # Linear: 1.0*1 + (-1.0)*1 = 0
        # Quadratic: J_sym[0,1] * s_0 * s_1 = 0.5 * 1 * 1 = 0.5
        # Total: 0.0 + 0.5 = 0.5
        expected_energy = (
            np.dot(h, spins) + 0.5 * spins @ J_sym @ spins - 0.5 * np.trace(J_sym)
        )
        assert np.isclose(energy, expected_energy, atol=1e-10), (
            f"Energy mismatch: {energy} vs {expected_energy}"
        )

        # Test with anti-aligned spins
        spins_anti = np.array([1, -1], dtype=np.int8)
        energy_anti = compute_energy(spins_anti, h, J_sym)
        expected_anti = (
            np.dot(h, spins_anti)
            + 0.5 * spins_anti @ J_sym @ spins_anti
            - 0.5 * np.trace(J_sym)
        )
        assert np.isclose(energy_anti, expected_anti, atol=1e-10), (
            f"Energy mismatch for anti-aligned: {energy_anti} vs {expected_anti}"
        )
