"""RNG correctness verification for large N (100-2000+) and high sweep counts (>10000).

Checks:
1. Determinism: same seed always produces the same result
2. Variation: different seeds produce different results (RNG is active)
3. Energy improvement: annealing actually reduces energy (not stuck)
4. Spin diversity: not all spins are frozen in one direction
5. num_reads independence: different reads produce different samples
"""

import numpy as np
import pytest

from digitalannealer import DASampler, PTDASampler

from .conftest import CUDA_AVAILABLE

SKIP_CUDA = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")


def _make_random_ising(n_spins: int, rng_seed: int = 0):
    """Create a random fully-connected Ising problem (dense)."""
    rng = np.random.default_rng(rng_seed)
    h = rng.standard_normal(n_spins).astype(np.float32)
    upper = rng.standard_normal((n_spins, n_spins)).astype(np.float32) / np.sqrt(
        n_spins
    )
    J = np.triu(upper, 1)
    J = J + J.T  # symmetric, zero diagonal
    return h, J


def _compute_energy(h, J, sample, n_spins):
    """Compute Ising energy: E = sum h_i s_i + sum_{i<j} J_ij s_i s_j."""
    spins = np.array([sample[i] for i in range(n_spins)], dtype=np.float32)
    J_upper = np.triu(J, 1)
    return float(np.dot(h, spins) + np.sum(J_upper * np.outer(spins, spins)))


@SKIP_CUDA
class TestRngLargeN:
    """Verify RNG works correctly for large N and high sweep counts."""

    @pytest.mark.parametrize("n_spins", [100, 500, 1000, 2000])
    def test_determinism_large_n_high_sweeps(self, n_spins):
        """Same seed must produce identical results even with large N and many sweeps."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        sampler1 = DASampler(seed=42)
        result1 = sampler1.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)

        sampler2 = DASampler(seed=42)
        result2 = sampler2.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)

        s1 = result1.first.sample
        s2 = result2.first.sample
        for i in range(n_spins):
            assert s1[i] == s2[i], f"Spin {i} differs: {s1[i]} vs {s2[i]} (N={n_spins})"

    @pytest.mark.parametrize("n_spins", [100, 500, 1000, 2000])
    def test_different_seeds_produce_different_results(self, n_spins):
        """Different seeds must give different samples (RNG is actually active)."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        results = []
        for seed in [10, 20, 30]:
            sampler = DASampler(seed=seed)
            result = sampler.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)
            spins = np.array([result.first.sample[i] for i in range(n_spins)])
            results.append(spins)

        # At least one pair must differ (extremely unlikely all 3 give identical spins)
        all_same = np.array_equal(results[0], results[1]) and np.array_equal(
            results[1], results[2]
        )
        assert not all_same, (
            f"All 3 seeds produced identical results for N={n_spins}. RNG may be broken."
        )

    @pytest.mark.parametrize("n_spins", [100, 500, 1000, 2000])
    def test_energy_improves_with_more_sweeps(self, n_spins):
        """More sweeps should yield lower energy, proving RNG-driven exploration works."""
        h, J = _make_random_ising(n_spins)

        sampler_short = DASampler(seed=42)
        result_short = sampler_short.sample_ising(h, J, num_sweeps=100, num_reads=5)

        sampler_long = DASampler(seed=42)
        result_long = sampler_long.sample_ising(h, J, num_sweeps=15000, num_reads=5)

        best_short = min(result_short.data_vectors["energy"])
        best_long = min(result_long.data_vectors["energy"])

        assert best_long <= best_short, (
            f"N={n_spins}: 15000 sweeps (E={best_long:.2f}) did NOT improve over "
            f"100 sweeps (E={best_short:.2f}). RNG-driven annealing may be broken."
        )

    @pytest.mark.parametrize("n_spins", [100, 500, 1000, 2000])
    def test_spin_diversity_across_reads(self, n_spins):
        """Multiple reads should produce diverse results, not identical copies."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        sampler = DASampler(seed=42)
        result = sampler.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=4)

        samples = []
        for sample in result.samples():
            spins = np.array([sample[i] for i in range(n_spins)])
            samples.append(spins)

        # Count how many pairs are identical
        n_identical = 0
        total_pairs = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                total_pairs += 1
                if np.array_equal(samples[i], samples[j]):
                    n_identical += 1

        assert n_identical < total_pairs, (
            f"N={n_spins}: All {total_pairs} read pairs are identical. "
            f"Per-read RNG independence may be broken."
        )

    @pytest.mark.parametrize("n_spins", [100, 500, 1000, 2000])
    def test_not_all_spins_same_direction(self, n_spins):
        """For random h/J, final spins should not all be +1 or all -1."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        sampler = DASampler(seed=42)
        result = sampler.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)

        spins = np.array([result.first.sample[i] for i in range(n_spins)])
        n_plus = np.sum(spins == 1)

        # For a random problem, the fraction of +1 spins should be roughly 50%
        # Allow wide tolerance: between 5% and 95%
        frac_plus = n_plus / n_spins
        assert 0.05 < frac_plus < 0.95, (
            f"N={n_spins}: {frac_plus * 100:.1f}% spins are +1. Spins may be stuck or RNG broken."
        )


@SKIP_CUDA
class TestRngLargeNSparse:
    """Verify RNG works in sparse mode for sizes that fit in shared memory."""

    @pytest.mark.parametrize("n_spins", [100, 500])
    def test_determinism_sparse(self, n_spins):
        """Sparse mode: same seed must produce identical results."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        sampler1 = DASampler(seed=42)
        result1 = sampler1.sample_ising(
            h, J, num_sweeps=num_sweeps, num_reads=1, sparse=True
        )

        sampler2 = DASampler(seed=42)
        result2 = sampler2.sample_ising(
            h, J, num_sweeps=num_sweeps, num_reads=1, sparse=True
        )

        s1 = result1.first.sample
        s2 = result2.first.sample
        for i in range(n_spins):
            assert s1[i] == s2[i], f"Sparse spin {i} differs (N={n_spins})"

    @pytest.mark.parametrize("n_spins", [100, 500])
    def test_different_seeds_sparse(self, n_spins):
        """Sparse mode: different seeds must give different results."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        results = []
        for seed in [10, 20, 30]:
            sampler = DASampler(seed=seed)
            result = sampler.sample_ising(
                h, J, num_sweeps=num_sweeps, num_reads=1, sparse=True
            )
            spins = np.array([result.first.sample[i] for i in range(n_spins)])
            results.append(spins)

        all_same = np.array_equal(results[0], results[1]) and np.array_equal(
            results[1], results[2]
        )
        assert not all_same, (
            f"Sparse: All 3 seeds identical for N={n_spins}. RNG may be broken."
        )


@SKIP_CUDA
class TestRngPTDA:
    """Verify RNG works in Parallel Tempering mode."""

    @pytest.mark.parametrize("n_spins", [100, 500, 1000])
    def test_ptda_determinism_high_sweeps(self, n_spins):
        """PT mode: same seed + high sweeps must produce identical results."""
        h, J = _make_random_ising(n_spins)
        num_sweeps = 15000

        sampler1 = PTDASampler(seed=42)
        result1 = sampler1.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)

        sampler2 = PTDASampler(seed=42)
        result2 = sampler2.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)

        s1 = result1.first.sample
        s2 = result2.first.sample
        for i in range(n_spins):
            assert s1[i] == s2[i], f"PT spin {i} differs (N={n_spins})"

    @pytest.mark.parametrize("n_spins", [100, 500, 1000])
    def test_ptda_different_seeds(self, n_spins):
        """PT mode: different seeds must give different results."""
        h, J = _make_random_ising(n_spins)
        # Scale sweeps with problem size to avoid over-convergence on easy
        # problems.  With best-solution tracking across all replicas, PT
        # reliably finds the global optimum for small N even at modest sweep
        # counts, so use a very short run to keep seeds distinguishable.
        num_sweeps = n_spins * 3

        results = []
        for seed in [10, 20, 30]:
            sampler = PTDASampler(seed=seed)
            result = sampler.sample_ising(h, J, num_sweeps=num_sweeps, num_reads=1)
            spins = np.array([result.first.sample[i] for i in range(n_spins)])
            results.append(spins)

        all_same = np.array_equal(results[0], results[1]) and np.array_equal(
            results[1], results[2]
        )
        assert not all_same, (
            f"PT: All 3 seeds identical for N={n_spins}. RNG may be broken."
        )


@SKIP_CUDA
class TestRngExtremeSweeps:
    """Verify RNG still works with very high sweep counts (50000+)."""

    def test_50k_sweeps_n500(self):
        """50000 sweeps with N=500: should still produce valid results."""
        h, J = _make_random_ising(500)
        sampler = DASampler(seed=42)
        result = sampler.sample_ising(h, J, num_sweeps=50000, num_reads=2)

        energies = result.data_vectors["energy"]
        # Energy should be finite
        assert all(np.isfinite(e) for e in energies), (
            "Non-finite energy with 50k sweeps"
        )

        # Two reads should differ
        samples = list(result.samples())
        s0 = np.array([samples[0][i] for i in range(500)])
        s1 = np.array([samples[1][i] for i in range(500)])
        assert not np.array_equal(s0, s1), "Two reads identical at 50k sweeps"

    def test_100k_sweeps_n100(self):
        """100000 sweeps with N=100: curandState should not cycle or degrade."""
        h, J = _make_random_ising(100)
        sampler = DASampler(seed=42)
        result = sampler.sample_ising(h, J, num_sweeps=100000, num_reads=2)

        energies = result.data_vectors["energy"]
        assert all(np.isfinite(e) for e in energies), (
            "Non-finite energy with 100k sweeps"
        )

        # Energy should be meaningfully negative for random h/J
        # (not stuck at zero or positive)
        best_energy = min(energies)
        assert best_energy < 0, (
            f"Best energy {best_energy:.2f} is >=0 with 100k sweeps — annealing may be frozen"
        )
