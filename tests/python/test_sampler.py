"""Tests for Digital Annealer samplers."""

import inspect
from typing import Any

import dimod
import numpy as np
import pytest

from digitalannealer import DASampler, PTDASampler

# Import conftest helper for checking CUDA availability
from .conftest import CUDA_AVAILABLE

# Mark to skip GPU-dependent tests when CUDA backend is unavailable
SKIP_CUDA = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")


class TestDASampler:
    """Tests for DASampler class."""

    pytestmark = SKIP_CUDA

    def test_sample_ising_dict_input(self):
        """Test sampling with dict input format."""
        sampler = DASampler(seed=42)
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -1.0}

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=5)

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 5
        assert result.vartype == dimod.SPIN
        assert set(result.variables) == {0, 1}

    def test_sample_ising_array_input(self):
        """Test sampling with numpy array input."""
        sampler = DASampler(seed=42)
        h = np.array([-1.0, 1.0])
        J = np.array([[0.0, -1.0], [-1.0, 0.0]])

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=3)

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3
        assert result.vartype == dimod.SPIN

    def test_sample_qubo(self):
        """Test QUBO sampling."""
        sampler = DASampler(seed=42)
        Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}

        result = sampler.sample_qubo(Q, num_sweeps=100, num_reads=3)

        assert isinstance(result, dimod.SampleSet)
        assert result.vartype == dimod.BINARY
        # Values should be 0 or 1
        for sample in result.samples():
            for val in sample.values():
                assert val in [0, 1]

    def test_finds_ground_state_simple(self):
        """Test that sampler finds ground state for simple problem."""
        sampler = DASampler(seed=42)
        # Simple ferromagnetic chain - ground state is all +1 or all -1
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0}

        result = sampler.sample_ising(
            h, J, num_sweeps=1000, num_reads=20, beta_max=20.0
        )

        # At least one sample should be a ground state (all aligned)
        found_ground_state = False
        for sample in result.samples():
            spins = [sample[i] for i in range(3)]
            if abs(sum(spins)) == 3:
                found_ground_state = True
                break

        assert found_ground_state, (
            "Should find at least one ground state (all spins aligned)"
        )

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results."""
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -0.5}

        sampler1 = DASampler(seed=12345)
        result1 = sampler1.sample_ising(h, J, num_sweeps=100, num_reads=3)

        sampler2 = DASampler(seed=12345)
        result2 = sampler2.sample_ising(h, J, num_sweeps=100, num_reads=3)

        # Same seed should give same samples
        for s1, s2 in zip(result1.samples(), result2.samples(), strict=True):
            assert dict(s1) == dict(s2)

    def test_energy_calculation(self):
        """Test that reported energies are correct."""
        sampler = DASampler(seed=42)
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 0.5], [0.5, 0.0]])

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=5)

        # Symmetrize J to match what the sampler does internally
        J_sym = (J + J.T) / 2.0

        # Use result.data() to get properly paired samples and energies
        for sample, energy in result.data(["sample", "energy"]):
            # Manually compute energy using dimod convention
            # E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
            spins = np.array([sample[0], sample[1]])
            # 0.5 * spins @ J_sym @ spins includes 0.5 * sum_i J_sym_ii; subtract it to drop diagonal terms
            expected_energy = (
                np.dot(h, spins) + 0.5 * spins @ J_sym @ spins - 0.5 * np.trace(J_sym)
            )
            assert np.isclose(energy, expected_energy, atol=1e-5), (
                f"Energy mismatch: {energy} vs {expected_energy}"
            )

    def test_energy_calculation_with_diagonal(self):
        """Test that energy calculation correctly handles non-zero diagonal terms in J."""
        sampler = DASampler(seed=42)
        h = np.array([1.0, -1.0])
        # J with non-zero diagonal terms
        J = np.array([[0.5, 0.5], [0.5, -0.3]])

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=5)

        # Symmetrize J to match what the sampler does internally
        J_sym = (J + J.T) / 2.0

        # Use result.data() to get properly paired samples and energies
        for sample, energy in result.data(["sample", "energy"]):
            # Manually compute energy using dimod convention
            # E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
            spins = np.array([sample[0], sample[1]])
            # 0.5 * spins @ J_sym @ spins includes 0.5 * sum_i J_sym_ii; subtract it to drop diagonal terms
            expected_energy = (
                np.dot(h, spins) + 0.5 * spins @ J_sym @ spins - 0.5 * np.trace(J_sym)
            )
            assert np.isclose(energy, expected_energy, atol=1e-5), (
                f"Energy mismatch: {energy} vs {expected_energy}"
            )

    def test_energy_calculation_with_asymmetric_j(self):
        """Test that energy calculation correctly handles asymmetric J with diagonal terms.

        This test verifies that the sampler correctly symmetrizes J before computing energies,
        which is important when J has asymmetric off-diagonal elements and non-zero diagonals.
        """
        sampler = DASampler(seed=42)
        h = np.array([1.0, -1.0])
        # J with asymmetric off-diagonal elements and non-zero diagonal
        # This will be symmetrized by the sampler: J[0,1] = J[1,0] = (0.3 + 0.7)/2 = 0.5
        J = np.array([[0.5, 0.3], [0.7, -0.3]])

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=5)

        # Symmetrize J to match what the sampler does internally
        # After symmetrization: J_sym = [[0.5, 0.5], [0.5, -0.3]]
        J_sym = (J + J.T) / 2.0

        # Use result.data() to get properly paired samples and energies
        for sample, energy in result.data(["sample", "energy"]):
            # Manually compute energy using dimod convention
            # E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
            spins = np.array([sample[0], sample[1]])
            # 0.5 * spins @ J_sym @ spins includes 0.5 * sum_i J_sym_ii; subtract it to drop diagonal terms
            expected_energy = (
                np.dot(h, spins) + 0.5 * spins @ J_sym @ spins - 0.5 * np.trace(J_sym)
            )
            assert np.isclose(energy, expected_energy, atol=1e-5), (
                f"Energy mismatch: {energy} vs {expected_energy}"
            )

    def test_energy_non_power_of_two_spins_dense(self):
        """Energy computation matches reference for non-power-of-two spin count."""
        sampler = DASampler(seed=7)
        n_spins = 10  # Block size would be non-power-of-two without clamping
        rng = np.random.default_rng(0)
        h = rng.standard_normal(n_spins).astype(np.float32)
        upper = rng.standard_normal((n_spins, n_spins)).astype(np.float32)
        J = np.triu(upper, 1)
        J = J + J.T  # Symmetric with zero diagonal

        result = sampler.sample_ising(h, J, num_sweeps=50, num_reads=2)
        J_upper = np.triu(J, 1)

        for sample, energy in result.data(["sample", "energy"]):
            spins = np.array([sample[i] for i in range(n_spins)], dtype=np.float32)
            expected_energy = float(
                np.dot(h, spins) + np.sum(J_upper * np.outer(spins, spins))
            )
            assert np.isclose(energy, expected_energy, atol=1e-4)


class TestDASamplerSparse:
    """Tests for DASampler sparse mode (CSR format)."""

    pytestmark = SKIP_CUDA

    def test_sample_ising_sparse_dict_input(self):
        """Test sparse mode sampling with dict input format."""
        sampler = DASampler(seed=42)
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -1.0}

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=5, sparse=True)

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 5
        assert result.vartype == dimod.SPIN
        assert set(result.variables) == {0, 1}

    def test_sample_ising_sparse_array_input(self):
        """Test sparse mode sampling with numpy array input."""
        sampler = DASampler(seed=42)
        h = np.array([-1.0, 1.0])
        J = np.array([[0.0, -1.0], [-1.0, 0.0]])

        result = sampler.sample_ising(h, J, num_sweeps=100, num_reads=3, sparse=True)

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3
        assert result.vartype == dimod.SPIN

    def test_sparse_finds_ground_state_simple(self):
        """Test that sparse mode finds ground state for simple problem."""
        sampler = DASampler(seed=42)
        # Simple ferromagnetic chain - ground state is all +1 or all -1
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0}

        result = sampler.sample_ising(
            h, J, num_sweeps=1000, num_reads=20, beta_max=20.0, sparse=True
        )

        # At least one sample should be a ground state (all aligned)
        found_ground_state = False
        for sample in result.samples():
            spins = [sample[i] for i in range(3)]
            if abs(sum(spins)) == 3:
                found_ground_state = True
                break

        assert found_ground_state, (
            "Sparse mode should find ground state (all spins aligned)"
        )

    def test_sparse_reproducibility_with_seed(self):
        """Test that sparse mode with same seed gives same results."""
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -0.5}

        sampler1 = DASampler(seed=12345)
        result1 = sampler1.sample_ising(h, J, num_sweeps=100, num_reads=3, sparse=True)

        sampler2 = DASampler(seed=12345)
        result2 = sampler2.sample_ising(h, J, num_sweeps=100, num_reads=3, sparse=True)

        # Same seed should give same samples
        for s1, s2 in zip(result1.samples(), result2.samples(), strict=True):
            assert dict(s1) == dict(s2)

    def test_sparse_larger_problem(self):
        """Test sparse mode with a larger sparse graph (ring topology)."""
        sampler = DASampler(seed=42)
        n = 10  # 10 spins in a ring

        # Ring topology: each spin connected only to neighbors
        h = dict.fromkeys(range(n), 0.0)
        J = {(i, (i + 1) % n): -1.0 for i in range(n)}

        result = sampler.sample_ising(
            h, J, num_sweeps=500, num_reads=10, beta_max=15.0, sparse=True
        )

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 10

        # For a ferromagnetic ring, ground state should have all spins aligned
        best_sample = result.first.sample
        spins = [best_sample[i] for i in range(n)]
        alignment = abs(sum(spins))
        assert alignment == n, f"Expected all spins aligned, got alignment {alignment}"

    def test_sparse_vs_dense_consistency(self):
        """Test that sparse and dense modes give consistent results."""
        sampler = DASampler(seed=42)
        h = {0: -1.0, 1: 1.0, 2: 0.5}
        J = {(0, 1): -1.0, (1, 2): 0.5}

        # Run dense mode
        result_dense = sampler.sample_ising(
            h, J, num_sweeps=500, num_reads=5, sparse=False
        )

        # Run sparse mode with same seed
        sampler_sparse = DASampler(seed=42)
        result_sparse = sampler_sparse.sample_ising(
            h, J, num_sweeps=500, num_reads=5, sparse=True
        )

        # Both should find valid solutions with similar quality
        # Note: exact match not guaranteed due to different code paths
        assert result_dense.first.energy <= 10.0  # Reasonable upper bound
        assert result_sparse.first.energy <= 10.0
        # Best energies should be close relative to their magnitude
        energy_dense = result_dense.first.energy
        energy_sparse = result_sparse.first.energy
        energy_diff = abs(energy_dense - energy_sparse)
        best_energy = min(energy_dense, energy_sparse)
        # Allow up to 20% relative difference, with a small absolute floor
        tolerance = max(0.2 * abs(best_energy), 1e-6)
        assert energy_diff <= tolerance

    def test_energy_non_power_of_two_spins_sparse(self):
        """Sparse energy computation matches reference for non-power-of-two spin count."""
        sampler = DASampler(seed=9)
        n_spins = 10
        rng = np.random.default_rng(1)
        h = {i: float(val) for i, val in enumerate(rng.standard_normal(n_spins))}
        dense_upper = np.triu(rng.standard_normal((n_spins, n_spins)), 1)
        J_matrix = dense_upper + dense_upper.T
        J = {
            (i, j): float(J_matrix[i, j])
            for i in range(n_spins)
            for j in range(i + 1, n_spins)
        }

        result = sampler.sample_ising(h, J, num_sweeps=50, num_reads=2, sparse=True)
        h_vec = np.array([h[i] for i in range(n_spins)], dtype=np.float32)

        for sample, energy in result.data(["sample", "energy"]):
            spins = np.array([sample[i] for i in range(n_spins)], dtype=np.float32)
            expected_energy = float(
                np.dot(h_vec, spins)
                + np.sum(np.triu(J_matrix, 1) * np.outer(spins, spins))
            )
            assert np.isclose(energy, expected_energy, atol=1e-4)


class TestPTDASampler:
    """Tests for PTDASampler class."""

    pytestmark = SKIP_CUDA

    def test_sample_ising_basic(self):
        """Test basic PT sampling."""
        sampler = PTDASampler(seed=42)
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -1.0}

        result = sampler.sample_ising(
            h, J, num_sweeps=100, num_reads=3, num_replicas=5, exchange_interval=10
        )

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3
        assert result.vartype == dimod.SPIN

    def test_pt_finds_ground_state(self):
        """Test that PT sampler finds ground state."""
        sampler = PTDASampler(seed=42)
        # Frustrated system - harder to solve
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0, (0, 2): 1.0}

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=500,
            num_reads=10,
            num_replicas=10,
            beta_min=0.1,
            beta_max=10.0,
        )

        # Ground state energy is -1 (two bonds satisfied, one frustrated)
        best_energy = result.first.energy
        assert best_energy <= -0.9, (
            f"Expected ground state energy ~-1, got {best_energy}"
        )


class TestPTDASamplerSparse:
    """Tests for PTDASampler sparse mode (CSR format)."""

    pytestmark = SKIP_CUDA

    def test_sample_ising_sparse_dict_input(self):
        """Test sparse mode PT sampling with dict input format."""
        sampler = PTDASampler(seed=42)
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -1.0}

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=100,
            num_reads=3,
            num_replicas=5,
            exchange_interval=10,
            sparse=True,
        )

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3
        assert result.vartype == dimod.SPIN
        assert set(result.variables) == {0, 1}

    def test_sample_ising_sparse_array_input(self):
        """Test sparse mode PT sampling with numpy array input."""
        sampler = PTDASampler(seed=42)
        h = np.array([-1.0, 1.0])
        J = np.array([[0.0, -1.0], [-1.0, 0.0]])

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=100,
            num_reads=3,
            num_replicas=5,
            exchange_interval=10,
            sparse=True,
        )

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 3
        assert result.vartype == dimod.SPIN

    def test_sparse_pt_finds_ground_state(self):
        """Test that sparse PT sampler finds ground state for a frustrated system."""
        sampler = PTDASampler(seed=42)
        # Frustrated system - harder to solve
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0, (0, 2): 1.0}

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=500,
            num_reads=10,
            num_replicas=10,
            beta_min=0.1,
            beta_max=10.0,
            sparse=True,
        )

        # Ground state energy is -1 (two bonds satisfied, one frustrated)
        best_energy = result.first.energy
        assert best_energy <= -0.9, (
            f"Expected ground state energy ~-1, got {best_energy}"
        )

    def test_sparse_pt_larger_problem(self):
        """Test sparse PT mode with a larger sparse graph (ring topology)."""
        sampler = PTDASampler(seed=42)
        n = 10  # 10 spins in a ring

        # Ring topology: each spin connected only to neighbors
        h = dict.fromkeys(range(n), 0.0)
        J = {(i, (i + 1) % n): -1.0 for i in range(n)}

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=500,
            num_reads=5,
            num_replicas=8,
            beta_min=0.1,
            beta_max=15.0,
            exchange_interval=10,
            sparse=True,
        )

        assert isinstance(result, dimod.SampleSet)
        assert len(result) == 5

        # For a ferromagnetic ring, ground state should have all spins aligned
        best_sample = result.first.sample
        spins = [best_sample[i] for i in range(n)]
        alignment = abs(sum(spins))
        assert alignment == n, f"Expected all spins aligned, got alignment {alignment}"

    def test_sparse_vs_dense_consistency(self):
        """Test that sparse and dense PT modes give consistent results."""
        h = {0: -1.0, 1: 1.0, 2: 0.5}
        J = {(0, 1): -1.0, (1, 2): 0.5}

        # Run dense mode
        sampler_dense = PTDASampler(seed=42)
        result_dense = sampler_dense.sample_ising(
            h,
            J,
            num_sweeps=500,
            num_reads=5,
            num_replicas=6,
            exchange_interval=10,
            sparse=False,
        )

        # Run sparse mode with same seed
        sampler_sparse = PTDASampler(seed=42)
        result_sparse = sampler_sparse.sample_ising(
            h,
            J,
            num_sweeps=500,
            num_reads=5,
            num_replicas=6,
            exchange_interval=10,
            sparse=True,
        )

        # Both should find valid solutions with similar quality
        # Note: exact match not guaranteed due to different code paths
        assert result_dense.first.energy <= 10.0  # Reasonable upper bound
        assert result_sparse.first.energy <= 10.0
        # Best energies should be close relative to their magnitude
        energy_dense = result_dense.first.energy
        energy_sparse = result_sparse.first.energy
        energy_diff = abs(energy_dense - energy_sparse)
        # Allow up to 20% relative difference, using max of absolute values to handle near-zero
        tolerance = max(0.2 * max(abs(energy_dense), abs(energy_sparse)), 1e-6)
        assert energy_diff <= tolerance

    def test_sparse_pt_reproducibility_with_seed(self):
        """Test that sparse PT mode with same seed gives same results."""
        h = {0: -1.0, 1: 1.0}
        J = {(0, 1): -0.5}

        sampler1 = PTDASampler(seed=12345)
        result1 = sampler1.sample_ising(
            h,
            J,
            num_sweeps=100,
            num_reads=3,
            num_replicas=5,
            exchange_interval=10,
            sparse=True,
        )

        sampler2 = PTDASampler(seed=12345)
        result2 = sampler2.sample_ising(
            h,
            J,
            num_sweeps=100,
            num_reads=3,
            num_replicas=5,
            exchange_interval=10,
            sparse=True,
        )

        # Same seed should give same samples
        for s1, s2 in zip(result1.samples(), result2.samples(), strict=True):
            assert dict(s1) == dict(s2)


class TestPTBenefitOverDA:
    """Verify PT correctness properties related to RNG continuity and best-replica selection.

    These tests exercise the three fixes that were critical to PT effectiveness:
    1. Philox RNG sweep_offset: random sequences no longer repeat across
       exchange intervals.
    2. Persistent dynamic offset: escape mechanism accumulates across the
       full run instead of resetting every exchange_interval.
    3. Best-solution selection: at the end of a read, returns the lowest-energy
       final state among all replicas, based on their final energies.
    """

    pytestmark = SKIP_CUDA

    def test_pt_reproducibility_with_fixed_seed(self):
        """PT with a fixed seed must produce identical results across runs.

        This validates that RNG continuity (sweep_offset) and persistent
        dynamic offsets yield deterministic behaviour when the seed is fixed.
        """
        n = 30
        rng = np.random.default_rng(seed=999)
        h_arr = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        J_arr = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
        J_arr = (J_arr + J_arr.T) / 2
        np.fill_diagonal(J_arr, 0.0)

        h = {i: float(h_arr[i]) for i in range(n)}
        J = {(i, j): float(J_arr[i, j]) for i in range(n) for j in range(i + 1, n)}

        kwargs: dict[str, Any] = {
            "num_sweeps": 500,
            "num_reads": 3,
            "num_replicas": 8,
            "beta_min": 0.1,
            "beta_max": 10.0,
            "exchange_interval": 10,
        }

        pt1 = PTDASampler(seed=42).sample_ising(h, J, **kwargs)
        pt2 = PTDASampler(seed=42).sample_ising(h, J, **kwargs)

        # Every read must match sample-for-sample
        for s1, s2 in zip(pt1.samples(), pt2.samples(), strict=True):
            assert dict(s1) == dict(s2)

    def test_best_replica_selection_returns_consistent_energy(self):
        """Returned sample's energy must match the Ising energy of its spin configuration.

        This validates that the end-of-read best-replica selection correctly
        pairs the returned spin configuration with its energy, confirming
        the engine copies the right replica's spins.
        """
        n = 30
        rng = np.random.default_rng(seed=777)
        h_arr = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        J_arr = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
        J_arr = (J_arr + J_arr.T) / 2
        np.fill_diagonal(J_arr, 0.0)

        h = {i: float(h_arr[i]) for i in range(n)}
        J = {(i, j): float(J_arr[i, j]) for i in range(n) for j in range(i + 1, n)}

        pt_sampler = PTDASampler(seed=42)
        pt_result = pt_sampler.sample_ising(
            h,
            J,
            num_sweeps=1000,
            num_reads=3,
            num_replicas=8,
            beta_min=0.1,
            beta_max=10.0,
            exchange_interval=10,
        )

        # Verify every returned sample's energy matches independent computation
        J_sym = (J_arr + J_arr.T) / 2.0
        for sample, energy in pt_result.data(["sample", "energy"]):
            spins = np.array([sample[i] for i in range(n)])
            expected_energy = (
                np.dot(h_arr, spins)
                + 0.5 * spins @ J_sym @ spins
                - 0.5 * np.trace(J_sym)
            )
            assert np.isclose(energy, expected_energy, atol=1e-4), (
                f"Energy mismatch: reported {energy} vs computed {expected_energy}"
            )


class TestPTDAAdaptiveSchedule:
    """Tests for PTDASampler beta adaptation (CPU fallback)."""

    pytestmark = SKIP_CUDA

    def test_adaptive_betas_reduce_acceptance_variance(self):
        sampler = PTDASampler(seed=123)
        h = np.zeros(4, dtype=np.float32)
        J = np.full((4, 4), 0.0, dtype=np.float32)
        # Encourage correlated spins to create meaningful energy gaps
        for i in range(3):
            J[i, i + 1] = -1.5
            J[i + 1, i] = -1.5

        initial_schedule = np.linspace(0.2, 5.0, num=6, dtype=np.float32)
        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=80,
            num_reads=1,
            num_replicas=6,
            beta_min=0.2,
            beta_max=5.0,
            # Use frequent exchanges to gather adaptation statistics quickly
            exchange_interval=1,
            adapt_betas=True,
            adapt_rounds=2,
            adapt_sweeps=40,
            log_adaptation=True,
            seed=123,
        )

        history = result.info.get("beta_adaptation_history")
        assert history is not None and len(history) == 2

        initial_var = float(np.var(history[0]["acceptance"]))
        final_var = float(np.var(history[-1]["acceptance"]))

        variance_tolerance = 1e-4
        assert final_var <= initial_var + variance_tolerance

        adapted_betas = np.array(result.info["betas"], dtype=np.float32)
        assert np.isclose(adapted_betas[0], 0.2)
        assert np.isclose(adapted_betas[-1], 5.0)
        assert np.all(np.diff(adapted_betas) >= 0.0)
        assert np.any(np.abs(adapted_betas - initial_schedule) > 1e-4)

    def test_adaptive_betas_enabled_by_default(self):
        sampler = PTDASampler(seed=321)
        h = np.zeros(4, dtype=np.float32)
        J = np.zeros((4, 4), dtype=np.float32)

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=20,
            num_reads=1,
            num_replicas=4,
            exchange_interval=1,
            log_adaptation=True,
            seed=321,
        )

        assert result.info["beta_adaptation_used"] is True
        history = result.info.get("beta_adaptation_history")
        assert history is not None and len(history) > 0

    def test_adaptive_betas_can_be_disabled_explicitly(self):
        sampler = PTDASampler(seed=654)
        h = np.zeros(4, dtype=np.float32)
        J = np.zeros((4, 4), dtype=np.float32)

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=20,
            num_reads=1,
            num_replicas=4,
            exchange_interval=1,
            adapt_betas=False,
            log_adaptation=True,
            seed=654,
        )

        assert result.info["beta_adaptation_used"] is False
        assert result.info.get("beta_adaptation_history") is None


class TestPTDABetaUpdateMath:
    """Tests for PT beta schedule update math."""

    def test_sample_ising_uses_linear_initial_beta_schedule(self, monkeypatch):
        captured: dict[str, np.ndarray] = {}

        def fake_sample_dense(self, h, J, betas, *args, **kwargs):
            captured["betas"] = np.asarray(betas, dtype=np.float64)
            return dimod.SampleSet.from_samples(
                [{0: 1, 1: -1}],
                vartype=dimod.SPIN,
                energy=[0.0],
            )

        monkeypatch.setattr(PTDASampler, "_sample_dense", fake_sample_dense)

        PTDASampler(seed=7).sample_ising(
            np.zeros(2, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            num_sweeps=2,
            num_reads=1,
            num_replicas=6,
            beta_min=0.2,
            beta_max=3.0,
            adapt_betas=False,
        )

        expected = np.linspace(0.2, 3.0, 6, dtype=np.float64)
        assert np.allclose(captured["betas"], expected, atol=1e-6, rtol=1e-6)

    def test_update_betas_from_acceptance_uses_linear_beta_gaps(self):
        betas = np.array([0.5, 1.0, 2.0, 4.5, 10.0], dtype=np.float32)
        acceptance = np.array([0.01, 0.20, 0.35, 0.90], dtype=np.float32)
        target = float(np.mean(acceptance))

        updated = PTDASampler._update_betas_from_acceptance(betas, acceptance, target)

        scaling = acceptance / target

        linear_deltas = np.diff(betas.astype(np.float64))
        adjusted_linear_deltas = linear_deltas * scaling.astype(np.float64)

        # Endpoints must stay fixed; renormalize deltas so the total span is preserved.
        span = float(betas[-1] - betas[0])
        adjusted_linear_deltas *= span / float(np.sum(adjusted_linear_deltas))

        expected_linear = np.empty_like(betas, dtype=np.float64)
        expected_linear[0] = float(betas[0])
        expected_linear[1:] = expected_linear[0] + np.cumsum(adjusted_linear_deltas)
        expected_linear[-1] = float(betas[-1])

        assert np.allclose(
            updated, expected_linear.astype(np.float32), atol=1e-6, rtol=1e-6
        )
        assert np.isclose(updated[0], betas[0])
        assert np.isclose(updated[-1], betas[-1])
        assert np.all(np.diff(updated) >= 0.0)

        # This schedule must come from linear-gap updates, not log-space updates.
        log_betas = np.log(betas.astype(np.float64))
        log_deltas = np.diff(log_betas)
        adjusted_log_deltas = log_deltas * scaling.astype(np.float64)

        log_span = float(log_betas[-1] - log_betas[0])
        adjusted_log_deltas *= log_span / float(np.sum(adjusted_log_deltas))

        expected_log = np.empty_like(log_betas)
        expected_log[0] = log_betas[0]
        expected_log[1:] = expected_log[0] + np.cumsum(adjusted_log_deltas)
        expected_log = np.exp(expected_log).astype(np.float32)
        expected_log[-1] = float(betas[-1])

        assert not np.allclose(updated, expected_log, atol=1e-4, rtol=1e-4)

    def test_update_betas_from_acceptance_returns_input_when_target_is_zero(self):
        betas = np.array([0.5, 1.0, 2.0, 4.5, 10.0], dtype=np.float32)
        acceptance = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        updated = PTDASampler._update_betas_from_acceptance(
            betas, acceptance, target=0.0
        )

        assert np.array_equal(updated, betas)


class TestInputValidation:
    """Tests for input validation."""

    def test_da_sample_qubo_signature_exposes_da_parameters(self):
        """DASampler.sample_qubo should expose DA-specific keyword parameters."""
        params = inspect.signature(DASampler.sample_qubo).parameters
        assert "Q" in params
        assert "num_sweeps" in params
        assert "beta_min" in params
        assert "sparse" in params
        assert "dtype" in params

    def test_ptda_sample_qubo_signature_exposes_pt_parameters(self):
        """PTDASampler.sample_qubo should expose PT-specific keyword parameters."""
        params = inspect.signature(PTDASampler.sample_qubo).parameters
        assert "Q" in params
        assert "num_replicas" in params
        assert "exchange_interval" in params
        assert "adapt_betas" in params
        assert "adapt_sweeps" in params

    def test_mismatched_types_rejected(self):
        """Test that mismatched h/J types raise error."""
        sampler = DASampler()
        h_dict = {0: 1.0}
        J_array = np.array([[0.0]])

        with pytest.raises(TypeError):
            sampler.sample_ising(h_dict, J_array)

    def test_invalid_h_dimension_rejected(self):
        """Test that wrong h dimension raises error."""
        sampler = DASampler()
        h = np.array([[1.0, 2.0]])  # 2D instead of 1D
        J = np.array([[0.0]])

        with pytest.raises(ValueError):
            sampler.sample_ising(h, J)

    def test_mismatched_dimensions_rejected(self):
        """Test that mismatched h/J dimensions raise error."""
        sampler = DASampler()
        h = np.array([1.0, 2.0])  # 2 spins
        J = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])  # 3x3

        with pytest.raises(ValueError):
            sampler.sample_ising(h, J)

    def test_nan_values_rejected(self):
        """Test that NaN values in input raise error."""
        sampler = DASampler()
        h = np.array([1.0, np.nan])
        J = np.array([[0.0, 1.0], [1.0, 0.0]])

        with pytest.raises(ValueError):
            sampler.sample_ising(h, J)

    def test_unsupported_dtype_rejected(self):
        """float16 and other non-float32/float64 dtypes should fail clearly."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))

        with pytest.raises(ValueError, match="Unsupported dtype"):
            sampler.sample_ising(h, J, num_sweeps=5, num_reads=1, dtype=np.float16)

    def test_float64_dtype_accepted(self):
        """float64 should be accepted and produce correct results."""
        sampler = DASampler(seed=0)
        h = np.array([0.0, 0.0], dtype=np.float64)
        J = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)

        result = sampler.sample_ising(
            h, J, num_sweeps=10, num_reads=1, dtype=np.float64
        )

        assert result.info.get("dtype") == np.dtype(np.float64).name
        assert len(result.record.sample) == 1


class TestDtypeMetadata:
    """Tests for dtype exposure in SampleSet.info."""

    def test_dtype_recorded_for_da(self):
        sampler = DASampler(seed=0)
        h = np.array([0.0, 0.0], dtype=np.float32)
        J = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)

        result = sampler.sample_ising(h, J, num_sweeps=5, num_reads=1, dtype=np.float32)

        assert result.info.get("dtype") == np.dtype(np.float32).name

    def test_dtype_recorded_for_ptda(self):
        sampler = PTDASampler(seed=0)
        h = np.array([0.0, 0.0], dtype=np.float32)
        J = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)

        result = sampler.sample_ising(
            h,
            J,
            num_replicas=3,
            num_sweeps=5,
            num_reads=1,
            exchange_interval=1,
            dtype=np.float32,
        )

        assert result.info.get("dtype") == np.dtype(np.float32).name

    def test_qubo_dtype_forwarding(self):
        sampler = DASampler(seed=0)
        Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}

        result = sampler.sample_qubo(Q, num_sweeps=5, num_reads=1, dtype=np.float32)
        assert result.info.get("dtype") == np.dtype(np.float32).name

        result64 = sampler.sample_qubo(Q, num_sweeps=5, num_reads=1, dtype=np.float64)
        assert result64.info.get("dtype") == np.dtype(np.float64).name

        with pytest.raises(ValueError):
            sampler.sample_qubo(Q, num_sweeps=5, num_reads=1, dtype=np.float16)

    def test_pt_qubo_dtype_forwarding(self):
        sampler = PTDASampler(seed=0)
        Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}

        result = sampler.sample_qubo(
            Q,
            num_sweeps=5,
            num_reads=1,
            num_replicas=3,
            exchange_interval=1,
            adapt_betas=False,
            dtype=np.float32,
        )

        assert result.vartype == dimod.BINARY
        assert result.info.get("dtype") == np.dtype(np.float32).name


class TestErrorCases:
    """Tests for error handling in samplers.

    These tests verify that the sampler properly handles error conditions including:
    - GPU memory exhaustion
    - Invalid parameter values
    - Edge cases like empty or very large problems
    """

    @pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA backend not available - test requires GPU"
    )
    def test_extremely_large_problem_raises_memory_error(self):
        """Test that extremely large problem sizes raise appropriate memory error.

        Uses the dedicated VRAM pre-check binding to verify that the C++ engine
        correctly rejects problem sizes requiring more VRAM than available.
        This avoids allocating huge host output arrays (which could OOM before
        the VRAM check runs).
        """
        from digitalannealer._core import check_vram_dense_da

        # 100,000 spins with dense J requires ~40 GB for J alone — exceeds any GPU.
        # check_vram_dense_da only estimates VRAM, no host allocation needed.
        with pytest.raises(RuntimeError, match=r"insufficient GPU memory"):
            check_vram_dense_da(n_spins=100_000, n_replicas=1, num_sweeps=1000)

    @pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA backend not available - test requires GPU"
    )
    def test_large_problem_with_many_reads_raises_memory_error(self):
        """Test that combination of large problem and many reads triggers memory check.

        Uses the dedicated VRAM pre-check binding to verify that the C++ engine
        accounts for per-read buffers (spins, delta_E, selection_rng) in its
        VRAM estimate.
        """
        from digitalannealer._core import check_vram_dense_da

        # 1,000 spins with 100,000,000 reads:
        # - J ≈ 4 MB, spins ≈ 100 GB, delta_e ≈ 400 GB — far exceeds any GPU.
        with pytest.raises(RuntimeError, match=r"insufficient GPU memory"):
            check_vram_dense_da(n_spins=1000, n_replicas=100_000_000, num_sweeps=10)

    @pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA backend not available - test requires GPU"
    )
    def test_ptda_large_problem_raises_memory_error(self):
        """Test that PT sampler also properly checks GPU memory for large problems.

        PT samplers multiply memory requirements by num_replicas since each replica
        maintains its own state. This test verifies the PT engine's VRAM check
        accounts for the increased memory usage.

        With Philox RNG (no per-spin curandState), the dominant cost is
        M × N × (int8 + float32) for spins+delta_e per replica.
        Use small n_spins to keep host arrays tiny, large num_replicas to blow VRAM.
        """
        sampler = PTDASampler(seed=42)
        # 1,000 spins with 10,000,000 replicas:
        # - J matrix: 1,000² × 4 bytes ≈ 4 MB (tiny host array)
        # - Per-replica: spins (1 KB) + delta_e (4 KB) = 5 KB
        # - 10M replicas × 5 KB ≈ 50 TB — exceeds any GPU
        n_spins = 1000
        num_replicas = 10_000_000
        h = np.zeros(n_spins, dtype=np.float32)
        J = np.zeros((n_spins, n_spins), dtype=np.float32)

        # Should raise RuntimeError from PT engine's VRAM check
        with pytest.raises(RuntimeError, match=r"insufficient GPU memory"):
            sampler.sample_ising(
                h,
                J,
                num_sweeps=10,
                num_reads=1,
                num_replicas=num_replicas,
                exchange_interval=5,
                adapt_betas=False,
            )

    @pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA backend not available - test requires GPU"
    )
    def test_large_num_sweeps_raises_memory_error(self):
        """Test that extremely large num_sweeps triggers memory check for schedule buffer.

        Uses the dedicated VRAM pre-check binding to verify that the C++ engine
        accounts for schedule buffer size (num_sweeps × sizeof(float)) in its
        VRAM estimate. Combined with many reads, this exceeds GPU memory.
        """
        from digitalannealer._core import check_vram_dense_da

        # 1,000 spins with 100,000,000 reads and 500,000 sweeps:
        # - J ≈ 4 MB, spins+delta_e ≈ 500 GB, schedule ≈ 2 MB
        # - Total far exceeds any GPU
        with pytest.raises(RuntimeError, match=r"insufficient GPU memory"):
            check_vram_dense_da(
                n_spins=1000, n_replicas=100_000_000, num_sweeps=500_000
            )

    @pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA backend not available - test requires GPU"
    )
    def test_ptda_large_exchange_interval_raises_memory_error(self):
        """Test that PT sampler accounts for pt_schedule buffer size.

        PT samplers allocate a schedule buffer sized num_replicas × exchange_interval × sizeof(float).
        With Philox, the per-replica state is smaller, so we use a large exchange_interval
        combined with many replicas to ensure the pt_schedule buffer alone exceeds VRAM.
        """
        sampler = PTDASampler(seed=42)
        # 1,000 spins with 100 replicas and 100,000,000 exchange_interval:
        # - J matrix: 1,000² × 4 bytes ≈ 4 MB (tiny host)
        # - PT schedule: 100 × 100,000,000 × 4 bytes ≈ 40 GB (exceeds GPU alone)
        # - Per-replica state: 100 × 1,000 × 5 bytes ≈ 500 KB (negligible)
        n_spins = 1000
        num_replicas = 100
        exchange_interval = 100_000_000
        h = np.zeros(n_spins, dtype=np.float32)
        J = np.zeros((n_spins, n_spins), dtype=np.float32)

        # Should raise RuntimeError from PT engine's VRAM check accounting for PT buffers
        with pytest.raises(RuntimeError, match=r"insufficient GPU memory"):
            sampler.sample_ising(
                h,
                J,
                num_sweeps=10,
                num_reads=1,
                num_replicas=num_replicas,
                exchange_interval=exchange_interval,
            )

    def test_zero_size_problem_handled_gracefully(self):
        """Test edge case of zero-size problem.

        While not a typical use case, the sampler should handle or reject
        empty problems without crashing or producing obscure CUDA errors.
        This test verifies defensive programming at the boundaries.
        """
        sampler = DASampler(seed=42)
        h = np.array([], dtype=np.float32)
        J = np.zeros((0, 0), dtype=np.float32)

        # This should either:
        # 1. Work and return a valid (empty) SampleSet, or
        # 2. Raise a clear validation error
        # Either is acceptable; we just verify no crash or obscure error
        try:
            result = sampler.sample_ising(h, J, num_sweeps=10, num_reads=1)
            # If it succeeds, verify the result structure is valid
            assert isinstance(result, dimod.SampleSet)
            assert len(result) == 1
        except (ValueError, RuntimeError, IndexError) as e:
            # If it raises an error, verify it's a proper exception
            # (not a segfault or silent failure)
            error_msg = str(e).lower()
            assert len(error_msg) > 0  # Non-empty error message

    def test_inf_values_rejected(self):
        """Test that infinite values in h raise appropriate error.

        The validation layer checks for non-finite values (NaN/Inf) and should
        raise ValueError with a clear message. This is caught before the GPU code.
        """
        sampler = DASampler()
        h = np.array([1.0, np.inf])
        J = np.array([[0.0, 1.0], [1.0, 0.0]])

        with pytest.raises(ValueError, match=r"(?i)(infinite|inf).*value|non.?finite"):
            sampler.sample_ising(h, J)

    def test_inf_in_j_matrix_rejected(self):
        """Test that infinite values in J matrix raise appropriate error.

        Similar to test_inf_values_rejected but for the J matrix.
        Verifies that validation catches non-finite values in couplings.
        """
        sampler = DASampler()
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, np.inf], [np.inf, 0.0]])

        with pytest.raises(ValueError, match=r"(?i)(infinite|inf).*value|non.?finite"):
            sampler.sample_ising(h, J)


class TestMetropolisAcceptance:
    """Tests for Metropolis-Hastings acceptance with inverse function table lookup.

    These tests verify that the sampler correctly implements the probabilistic
    acceptance criterion, enabling escape from local minima through uphill moves.
    """

    pytestmark = SKIP_CUDA

    def test_accepts_uphill_moves_at_high_temperature(self):
        """Test that uphill moves are accepted at high temperature (low beta).

        At high temperature (low beta), the acceptance probability for uphill
        moves is higher. This test verifies that the sampler can accept
        energy-increasing flips when beta is low.

        The Metropolis acceptance: A(delta_E) = min(1, exp(-beta * delta_E))
        For delta_E > 0 and small beta, exp(-beta * delta_E) approaches 1.
        """
        sampler = DASampler(seed=42)

        # Create a problem where all spins start aligned (ground state)
        # At high temperature, the sampler should make moves even when they increase energy
        h = np.zeros(5, dtype=np.float32)
        J = np.zeros((5, 5), dtype=np.float32)
        for i in range(4):
            J[i, i + 1] = -1.0  # Ferromagnetic chain
            J[i + 1, i] = -1.0

        # Run with very low beta (high temperature) - should explore more states
        result_high_temp = sampler.sample_ising(
            h, J, num_sweeps=500, num_reads=10, beta_min=0.01, beta_max=0.1
        )

        # At high temperature, we should see more diversity in samples
        # (not all samples should be ground states)
        unique_energies = set()
        for _, energy in result_high_temp.data(["sample", "energy"]):
            unique_energies.add(round(energy, 2))

        # With probabilistic acceptance, we expect some energy variation
        # Ground state energy for ferromagnetic chain is -4
        assert len(unique_energies) >= 2, (
            "High temperature runs should yield multiple unique energy values"
        )

    def test_escapes_local_minimum_with_metropolis(self):
        """Test that Metropolis acceptance helps escape local minima.

        This test creates a frustrated system where the sampler might get stuck
        without probabilistic acceptance. With Metropolis-Hastings, the sampler
        should be able to escape and find better solutions.
        """
        sampler = DASampler(seed=123)

        # Frustrated triangle: cannot satisfy all interactions
        # Ground state has energy -1 (two satisfied, one frustrated)
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0, (0, 2): 1.0}  # Antiferromagnetic (0,2)

        # Run multiple times to test exploration capability
        result = sampler.sample_ising(
            h, J, num_sweeps=1000, num_reads=20, beta_min=0.5, beta_max=5.0
        )

        # Should find ground state (energy = -1) in at least some samples
        ground_state_found = any(energy <= -0.9 for energy in result.record.energy)
        assert ground_state_found, (
            f"Metropolis acceptance should help find ground state. "
            f"Best energy: {min(result.record.energy)}"
        )

    def test_metropolis_temperature_dependence(self):
        """Test that acceptance rate varies with temperature as expected.

        At low temperature (high beta), the sampler should be more selective,
        primarily accepting energy-decreasing moves.
        At high temperature (low beta), the sampler should accept more moves,
        including energy-increasing ones.
        """
        # Use simple 2-spin system for controlled testing
        h = np.array([0.5, -0.5], dtype=np.float32)  # Bias to create energy landscape
        J = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)

        # High temperature run
        sampler_high = DASampler(seed=42)
        result_high = sampler_high.sample_ising(
            h, J, num_sweeps=200, num_reads=10, beta_min=0.1, beta_max=0.5
        )

        # Low temperature run
        sampler_low = DASampler(seed=42)
        result_low = sampler_low.sample_ising(
            h, J, num_sweeps=200, num_reads=10, beta_min=5.0, beta_max=20.0
        )

        # Low temperature should produce lower average energy (more selective)
        avg_energy_high = np.mean(result_high.record.energy)
        avg_energy_low = np.mean(result_low.record.energy)

        # With enough sweeps, low temperature should find better (lower) energy
        assert avg_energy_low <= avg_energy_high, (
            f"Low temperature ({avg_energy_low}) should find better or equal energy "
            f"than high temperature ({avg_energy_high})"
        )


@pytest.mark.skipif(CUDA_AVAILABLE, reason="CPU fallback behavior only")
class TestPTFallbackMetropolis:
    """CPU fallback-specific tests for PTDASampler Metropolis acceptance."""

    def test_accepts_uphill_moves_at_high_temperature(self):
        """PT CPU fallback should accept uphill moves when temperature is high."""
        sampler = PTDASampler(seed=7)

        # Simple ferromagnetic chain
        h = np.zeros(4, dtype=np.float32)
        J = np.zeros((4, 4), dtype=np.float32)
        for i in range(3):
            J[i, i + 1] = -1.0
            J[i + 1, i] = -1.0

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=200,
            num_reads=10,
            num_replicas=4,
            beta_min=0.01,
            beta_max=0.1,
        )

        # High temperature should yield a mix of energies (not all ground states)
        unique_energies = {round(float(e), 2) for e in result.record.energy}
        assert len(unique_energies) >= 2, (
            "Expected multiple energy levels at high temperature"
        )

    def test_accepts_uphill_with_zero_offset(self):
        """Metropolis acceptance should allow uphill flips even when offset is disabled."""
        sampler = PTDASampler(seed=123)

        # Single-spin system with positive field: ground state is s = -1, uphill state is +1
        h = np.array([1.0], dtype=np.float32)
        J = np.zeros((1, 1), dtype=np.float32)

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=50,
            num_reads=50,
            num_replicas=3,
            beta_min=0.01,
            beta_max=0.01,
            exchange_interval=5,
            offset_delta=0.0,  # isolate Metropolis acceptance from dynamic offset
        )

        samples = np.array([sample[0] for sample in result.record.sample])
        uphill_fraction = float(np.mean(samples == 1))

        # If only ΔE < 0 moves were allowed, uphill_fraction would be ~0 with offset disabled.
        assert 0.05 < uphill_fraction < 0.95

    def test_beta_adaptation_uses_cpu_metropolis(self):
        """Beta adaptation pre-runs should use the CPU Metropolis implementation."""
        sampler = PTDASampler(seed=11)

        # Zero-coupling system keeps exchange deltas at 0, ensuring frequent exchanges.
        h = np.zeros(3, dtype=np.float32)
        J = np.zeros((3, 3), dtype=np.float32)

        result = sampler.sample_ising(
            h,
            J,
            num_sweeps=30,
            num_reads=1,
            num_replicas=4,
            beta_min=0.2,
            beta_max=2.0,
            exchange_interval=1,
            adapt_betas=True,
            adapt_rounds=1,
            adapt_sweeps=15,
            log_adaptation=True,
            offset_delta=0.0,
        )

        assert result.info["beta_adaptation_used"] is True

        history = result.info.get("beta_adaptation_history")
        assert history is not None and len(history) == 1

        acceptance = np.asarray(history[0]["acceptance"], dtype=np.float32)
        # There are num_replicas - 1 exchange pairs; all should have been attempted
        assert acceptance.shape[0] == 3
        assert np.any(acceptance > 0.0)


class TestSamplingParamValidation:
    """Tests for unified sampling parameter validation in BaseSampler."""

    def test_zero_num_sweeps_rejected(self):
        """num_sweeps=0 must raise a clear ValueError."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="num_sweeps"):
            sampler.sample_ising(h, J, num_sweeps=0)

    def test_negative_num_sweeps_rejected(self):
        """Negative num_sweeps must raise ValueError."""
        sampler = DASampler()
        h = np.array([0.0])
        J = np.zeros((1, 1))
        with pytest.raises(ValueError, match="num_sweeps"):
            sampler.sample_ising(h, J, num_sweeps=-5)

    def test_zero_num_reads_rejected(self):
        """num_reads=0 must raise a clear ValueError."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="num_reads"):
            sampler.sample_ising(h, J, num_reads=0)

    def test_non_positive_beta_min_rejected(self):
        """beta_min <= 0 must raise ValueError."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="beta_min"):
            sampler.sample_ising(h, J, beta_min=0.0)
        with pytest.raises(ValueError, match="beta_min"):
            sampler.sample_ising(h, J, beta_min=-1.0)

    def test_beta_max_less_than_beta_min_rejected(self):
        """beta_max < beta_min must raise ValueError."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="beta_max"):
            sampler.sample_ising(h, J, beta_min=5.0, beta_max=1.0)

    def test_negative_offset_delta_rejected(self):
        """Negative offset_delta must raise ValueError."""
        sampler = DASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="offset_delta"):
            sampler.sample_ising(h, J, offset_delta=-0.1)

    def test_pt_num_replicas_too_small_rejected(self):
        """num_replicas < 2 must raise ValueError for PTDASampler."""
        sampler = PTDASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="num_replicas"):
            sampler.sample_ising(h, J, num_replicas=1)

    def test_pt_zero_exchange_interval_rejected(self):
        """exchange_interval < 1 must raise ValueError for PTDASampler."""
        sampler = PTDASampler()
        h = np.array([0.0, 0.0])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match="exchange_interval"):
            sampler.sample_ising(h, J, exchange_interval=0)

    def test_pt_adapt_rounds_zero_with_adapt_enabled_rejected(self):
        """adapt_rounds=0 with adapt_betas=True must raise ValueError."""
        sampler = PTDASampler()
        h = np.array([0.0, 0.0, 0.0])
        J = np.zeros((3, 3))
        with pytest.raises(ValueError, match="adapt_rounds"):
            sampler.sample_ising(h, J, num_replicas=4, adapt_betas=True, adapt_rounds=0)

    def test_da_unknown_kwargs_warned(self, monkeypatch):
        """Unknown keyword arguments should emit a warning and remain non-breaking."""
        sampler = DASampler()
        h = np.array([0.0], dtype=np.float32)
        J = np.zeros((1, 1), dtype=np.float32)

        def fake_sample_dense(*_args, **_kwargs):
            return dimod.SampleSet.from_samples(
                [{0: 1}],
                vartype=dimod.SPIN,
                energy=[0.0],
            )

        monkeypatch.setattr(DASampler, "_sample_dense", fake_sample_dense)

        with pytest.warns(UserWarning, match="unsupported keyword argument"):
            sampler.sample_ising(h, J, unknown_option=True)

    def test_pt_unknown_kwargs_warned(self, monkeypatch):
        """PT sampler should also warn (not fail) on unknown keyword arguments."""
        sampler = PTDASampler()
        h = np.array([0.0], dtype=np.float32)
        J = np.zeros((1, 1), dtype=np.float32)

        def fake_sample_dense(*_args, **_kwargs):
            return dimod.SampleSet.from_samples(
                [{0: 1}],
                vartype=dimod.SPIN,
                energy=[0.0],
            )

        monkeypatch.setattr(PTDASampler, "_sample_dense", fake_sample_dense)

        with pytest.warns(UserWarning, match="unsupported keyword argument"):
            sampler.sample_ising(h, J, unknown_option=True)


class TestPTDAQubo:
    """Tests for QUBO sampling via PTDASampler."""

    pytestmark = SKIP_CUDA

    def test_pt_sample_qubo_returns_binary(self):
        """PTDASampler.sample_qubo should return BINARY vartype with correct metadata."""
        sampler = PTDASampler(seed=42)
        Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        result = sampler.sample_qubo(
            Q, num_sweeps=100, num_reads=3, num_replicas=4, adapt_betas=False
        )
        assert isinstance(result, dimod.SampleSet)
        assert result.vartype == dimod.BINARY
        for sample in result.samples():
            for val in sample.values():
                assert val in [0, 1]

    def test_pt_qubo_finds_ground_state(self):
        """PT should find ground state of a simple QUBO."""
        sampler = PTDASampler(seed=42)
        # Max-cut on triangle: optimal is cutting one vertex
        Q = {(0, 1): -1.0, (1, 2): -1.0, (0, 2): -1.0}
        result = sampler.sample_qubo(
            Q, num_sweeps=500, num_reads=10, num_replicas=6, beta_max=20.0
        )
        assert result.first.energy <= -2.0


class TestSampleSetMetadata:
    """Tests for SampleSet.info metadata produced by the refactored _build_sample_set."""

    pytestmark = SKIP_CUDA

    def test_da_sample_set_has_dtype(self):
        """DASampler SampleSet.info must contain 'dtype'."""
        result = DASampler(seed=0).sample_ising({0: 0.0}, {}, num_sweeps=5, num_reads=1)
        assert result.info["dtype"] == "float32"

    def test_pt_sample_set_has_betas_and_dtype(self):
        """PTDASampler SampleSet.info must contain 'betas', 'beta_adaptation_used', and 'dtype'."""
        result = PTDASampler(seed=0).sample_ising(
            {0: 0.0, 1: 0.0},
            {(0, 1): -1.0},
            num_sweeps=10,
            num_reads=1,
            num_replicas=3,
            adapt_betas=False,
        )
        assert "betas" in result.info
        assert "beta_adaptation_used" in result.info
        assert result.info["dtype"] == "float32"
        assert result.info["beta_adaptation_used"] is False

    def test_pt_adaptation_history_logged_when_requested(self):
        """log_adaptation=True should include adaptation history in SampleSet.info."""
        result = PTDASampler(seed=0).sample_ising(
            {0: 0.0, 1: 0.0, 2: 0.0},
            {(0, 1): -1.0, (1, 2): -1.0},
            num_sweeps=50,
            num_reads=1,
            num_replicas=4,
            adapt_betas=True,
            adapt_rounds=1,
            log_adaptation=True,
        )
        assert "beta_adaptation_history" in result.info
        assert len(result.info["beta_adaptation_history"]) == 1
