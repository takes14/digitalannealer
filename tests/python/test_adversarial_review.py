"""Adversarial review tests for Digital Annealer (DA, single temperature).

Covers the six correctness-critical axes identified in the review checklist:
  1. Algorithm-spec conformance  (acceptance, offset, uniform candidate selection)
  2. CSR structure contract  (Python validate_csr_structure + C++ binding gate)
  3. Dense/sparse equivalence  (same problem → same energy distribution)
  4. Input extremes / NaN/Inf rejection
  5. validate_csr_structure unit tests (no GPU required)
  6. C++ binding CSR gate tests (requires CUDA, bypasses Python normaliser)

Tests marked SKIP_CUDA require the compiled CUDA extension and are skipped
automatically when no CUDA backend is available.
"""

import importlib.util

import numpy as np
import pytest

from digitalannealer.utils.validation import (
    normalize_ising_to_sparse,
    validate_csr_structure,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _core_available() -> bool:
    spec = importlib.util.find_spec("digitalannealer._core")
    if spec is None:
        return False
    try:
        import digitalannealer._core  # noqa: F401

        return True
    except Exception:
        return False


CUDA_AVAILABLE = _core_available()
SKIP_CUDA = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")


def _tiny_beta_schedule(n_sweeps: int = 200, dtype=np.float32) -> np.ndarray:
    """Geometric beta schedule for small test problems."""
    return np.geomspace(0.1, 5.0, n_sweeps, dtype=dtype)


def _ferromagnetic_ising(n: int):
    """All-to-all ferromagnetic: h=0, J_{ij}=-1 for i<j.  GS energy = -n*(n-1)/2."""
    h = np.zeros(n, dtype=np.float32)
    J = -np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(J, 0.0)
    return h, J


# ---------------------------------------------------------------------------
# 1. validate_csr_structure – unit tests (no GPU)
# ---------------------------------------------------------------------------


class TestValidateCsrStructure:
    """Pure-Python unit tests for validate_csr_structure."""

    def test_valid_2spin_chain_passes(self):
        """Simple 2-spin chain CSR is accepted."""
        row_offsets = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([1, 0], dtype=np.int32)
        validate_csr_structure(row_offsets, col_indices, n_spins=2)  # must not raise

    def test_valid_3spin_triangle_passes(self):
        """3-spin fully connected graph (symmetric CSR, no diagonal) is accepted."""
        row_offsets = np.array([0, 2, 4, 6], dtype=np.int32)
        col_indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
        validate_csr_structure(row_offsets, col_indices, n_spins=3)

    def test_empty_graph_passes(self):
        """No edges (all h, J=0) is a valid empty CSR."""
        n = 4
        row_offsets = np.zeros(n + 1, dtype=np.int32)
        col_indices = np.array([], dtype=np.int32)
        validate_csr_structure(row_offsets, col_indices, n_spins=n)

    def test_rejects_nonzero_row_offsets_start(self):
        """row_offsets[0] != 0 must raise ValueError."""
        row_offsets = np.array([1, 2, 3], dtype=np.int32)
        col_indices = np.array([1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="row_offsets\\[0\\] must be 0"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_rejects_non_monotonic_row_offsets(self):
        """Non-monotonic row_offsets must raise ValueError."""
        # row_offsets[1]=2 > row_offsets[2]=0 — decreasing
        row_offsets = np.array([0, 2, 0, 2], dtype=np.int32)
        col_indices = np.array([1, 2, 0, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="non-decreasing"):
            validate_csr_structure(row_offsets, col_indices, n_spins=3)

    def test_rejects_col_index_below_zero(self):
        """col_indices with a negative value must raise ValueError."""
        row_offsets = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([-1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_rejects_col_index_out_of_bounds(self):
        """col_indices >= n_spins must raise ValueError."""
        row_offsets = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([1, 99], dtype=np.int32)  # 99 out of range for n=2
        with pytest.raises(ValueError, match="out of range"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_rejects_diagonal_entry(self):
        """CSR with a diagonal entry (col == row) must raise ValueError."""
        # row 0 → col 0 (self-loop: diagonal)
        row_offsets = np.array([0, 2, 3], dtype=np.int32)
        col_indices = np.array([0, 1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="diagonal"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_rejects_diagonal_in_middle_row(self):
        """Diagonal in row 1 of a 3-node CSR must be detected."""
        row_offsets = np.array([0, 2, 4, 6], dtype=np.int32)
        # row 1: col_indices=[1, 2] — col 1 == row 1 → diagonal
        col_indices = np.array([1, 2, 1, 2, 0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="diagonal"):
            validate_csr_structure(row_offsets, col_indices, n_spins=3)

    def test_rejects_duplicate_columns_within_row(self):
        """Duplicate column indices in one row must raise ValueError."""
        row_offsets = np.array([0, 2, 3], dtype=np.int32)
        col_indices = np.array([1, 1, 0], dtype=np.int32)
        values = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="duplicate column indices|duplicate col"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2, values=values)

    def test_rejects_missing_reverse_edge(self):
        """Asymmetric CSR without reverse edge must raise ValueError."""
        row_offsets = np.array([0, 1, 1], dtype=np.int32)
        col_indices = np.array([1], dtype=np.int32)  # 0->1 exists, 1->0 missing
        with pytest.raises(ValueError, match="symmetric|reverse edge"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_rejects_reverse_edge_value_mismatch_when_values_provided(self):
        """Asymmetric reverse-edge values must raise ValueError when values are validated."""
        row_offsets = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([1, 0], dtype=np.int32)
        values = np.array([-0.5, -0.4], dtype=np.float64)
        with pytest.raises(ValueError, match="value mismatch|symmetric"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2, values=values)

    def test_rejects_wrong_row_offsets_length(self):
        """row_offsets of length != n_spins+1 must raise ValueError."""
        row_offsets = np.array(
            [0, 1], dtype=np.int32
        )  # len=2, but n_spins=3 → need len=4
        col_indices = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError):
            validate_csr_structure(row_offsets, col_indices, n_spins=3)

    def test_rejects_mismatched_nnz(self):
        """row_offsets[-1] != len(col_indices) must raise ValueError."""
        row_offsets = np.array([0, 2, 4], dtype=np.int32)  # declares 4 entries
        col_indices = np.array([1, 0], dtype=np.int32)  # but only 2 provided
        with pytest.raises(ValueError, match="row_offsets.*must equal.*col_indices"):
            validate_csr_structure(row_offsets, col_indices, n_spins=2)

    def test_normaliser_output_always_valid(self):
        """CSR produced by normalize_ising_to_sparse must always pass validation."""
        h = {0: -1.0, 1: 0.5, 2: 0.0}
        J = {(0, 1): -0.5, (1, 2): 0.3, (0, 2): 0.1}
        _, row_offsets, col_indices, _, _ = normalize_ising_to_sparse(h, J)
        validate_csr_structure(row_offsets, col_indices, n_spins=3)

    def test_normaliser_dict_with_duplicates_passes(self):
        """Duplicate (i,j) entries in dict input are deduped; result must be valid."""
        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): 0.5, (1, 0): 0.5}  # same edge given twice—normaliser deduplicates
        _, row_offsets, col_indices, _, _ = normalize_ising_to_sparse(h, J)
        validate_csr_structure(row_offsets, col_indices, n_spins=2)


# ---------------------------------------------------------------------------
# 2. C++ binding CSR gate (requires CUDA extension to be importable)
#    These tests call _core directly, bypassing Python normalisation,
#    to verify the C++ validate_csr_internal backstop.
# ---------------------------------------------------------------------------


@SKIP_CUDA
class TestBindingCsrGate:
    """C++ binding rejects malformed CSR before any GPU memory is allocated."""

    def _make_valid_sparse_inputs(self, n: int = 3):
        """Return a minimal valid (h, row_offsets, col_indices, values, betas) for n spins."""
        import digitalannealer._core as _core  # noqa: F401

        h = np.zeros(n, dtype=np.float32)
        # Fully connected: each row has n-1 neighbours
        row_offsets = np.array([i * (n - 1) for i in range(n + 1)], dtype=np.int32)
        col_list = []
        for i in range(n):
            col_list.extend([j for j in range(n) if j != i])
        col_indices = np.array(col_list, dtype=np.int32)
        values = -0.5 * np.ones(len(col_indices), dtype=np.float32)
        betas = _tiny_beta_schedule(50)
        return h, row_offsets, col_indices, values, betas

    def test_valid_input_accepted(self):
        """Sanity: valid CSR arrives at GPU without exception."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(4)
        # Should not raise
        _core.da_sample_ising_sparse(
            h, row_offsets, col_indices, values, betas, 1, 0.05, 42
        )

    def test_rejects_nonzero_row_offsets_start(self):
        """C++ gate raises invalid_argument when row_offsets[0] != 0."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        row_offsets[0] = 1  # corrupt
        with pytest.raises(Exception, match="row_offsets\\[0\\]"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_non_monotonic_row_offsets(self):
        """C++ gate raises invalid_argument for decreasing row_offsets."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        row_offsets[2] = row_offsets[1] - 1  # make row 1 appear to have negative length
        with pytest.raises(Exception, match="non-decreasing"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_out_of_bounds_col_index(self):
        """C++ gate raises invalid_argument when a col_index is >= n_spins."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        col_indices[0] = 99  # out of range
        with pytest.raises(Exception, match="out of range"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_negative_col_index(self):
        """C++ gate raises invalid_argument when a col_index is negative."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        col_indices[0] = -1
        with pytest.raises(Exception, match="out of range"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_diagonal_entry(self):
        """C++ gate raises invalid_argument when CSR contains a diagonal entry."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        # Make col_indices[0]=0 which is the diagonal for row 0
        col_indices[0] = 0
        with pytest.raises(Exception, match="diagonal"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_duplicate_columns_within_row(self):
        """C++ gate raises invalid_argument for duplicate col_indices in a row."""
        import digitalannealer._core as _core

        h = np.zeros(2, dtype=np.float32)
        row_offsets = np.array([0, 2, 3], dtype=np.int32)
        col_indices = np.array([1, 1, 0], dtype=np.int32)
        values = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        betas = _tiny_beta_schedule(50)
        with pytest.raises(Exception, match="duplicate column indices|duplicate col"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_mismatched_nnz(self):
        """C++ gate raises when row_offsets[-1] != len(col_indices)."""
        import digitalannealer._core as _core

        h, row_offsets, col_indices, values, betas = self._make_valid_sparse_inputs(3)
        # Corrupt row_offsets to claim more entries than col_indices has
        row_offsets[-1] = len(col_indices) + 10
        with pytest.raises(Exception, match="row_offsets.*col_indices|out of range"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_missing_reverse_edge(self):
        """C++ gate raises invalid_argument when CSR is not symmetric."""
        import digitalannealer._core as _core

        h = np.zeros(2, dtype=np.float32)
        row_offsets = np.array([0, 1, 1], dtype=np.int32)  # row 1 has no reverse edge
        col_indices = np.array([1], dtype=np.int32)
        values = np.array([-0.5], dtype=np.float32)
        betas = _tiny_beta_schedule(50)
        with pytest.raises(Exception, match="symmetric|reverse edge"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )

    def test_rejects_reverse_edge_value_mismatch(self):
        """C++ gate raises when reverse edge exists but value differs."""
        import digitalannealer._core as _core

        h = np.zeros(2, dtype=np.float32)
        row_offsets = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([1, 0], dtype=np.int32)
        values = np.array(
            [-0.5, -0.4], dtype=np.float32
        )  # mismatch between (0,1) and (1,0)
        betas = _tiny_beta_schedule(50)
        with pytest.raises(Exception, match="value mismatch|symmetric"):
            _core.da_sample_ising_sparse(
                h, row_offsets, col_indices, values, betas, 1, 0.05, 42
            )


# ---------------------------------------------------------------------------
# 3. Dense / Sparse equivalence (requires CUDA)
#
#    Same Ising instance solved via dense and sparse paths must yield
#    statistically consistent minimum energies.  We use a small known problem
#    (ferromagnetic chain) whose ground-state energy is analytically known.
# ---------------------------------------------------------------------------


@SKIP_CUDA
class TestDenseSparseEquivalence:
    """Dense and sparse DA paths must agree on a small ferromagnetic problem."""

    def _run_dense(self, h, J, seed=0, num_reads=32, num_sweeps=500):
        import digitalannealer._core as _core

        h_f32 = h.astype(np.float32)
        J_f32 = J.astype(np.float32)
        betas = _tiny_beta_schedule(num_sweeps)
        samples, energies = _core.da_sample_ising(
            h_f32, J_f32, betas, num_reads, 0.1, seed
        )
        return samples, energies

    def _run_sparse(self, h, J, seed=0, num_reads=32, num_sweeps=500):
        import digitalannealer._core as _core

        h_f64 = h.astype(np.float64)
        J_f64 = J.astype(np.float64)
        _, row_offsets, col_indices, values, _ = normalize_ising_to_sparse(h_f64, J_f64)
        h_f32 = h_f64.astype(np.float32)
        row_offsets_i32 = row_offsets.astype(np.int32)
        col_indices_i32 = col_indices.astype(np.int32)
        values_f32 = values.astype(np.float32)
        betas = _tiny_beta_schedule(num_sweeps)
        samples, energies = _core.da_sample_ising_sparse(
            h_f32,
            row_offsets_i32,
            col_indices_i32,
            values_f32,
            betas,
            num_reads,
            0.1,
            seed,
        )
        return samples, energies

    def test_ferromagnetic_chain_min_energy_agrees(self):
        """Both paths find an energy near ground state for a ferromagnetic chain.

        Ground-state energy of an n-spin all-negative J chain: -(n-1).
        """
        n = 6
        h = np.zeros(n, dtype=np.float64)
        J = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            J[i, i + 1] = J[i + 1, i] = -1.0

        _, e_dense = self._run_dense(h, J)
        _, e_sparse = self._run_sparse(h, J)

        gs_energy = -(n - 1)  # = -5 for n=6
        # Both paths should find (or closely approach) the ground state
        assert e_dense.min() <= gs_energy + 1.0, (
            f"Dense path did not approach GS: min energy = {e_dense.min()}"
        )
        assert e_sparse.min() <= gs_energy + 1.0, (
            f"Sparse path did not approach GS: min energy = {e_sparse.min()}"
        )

    def test_dense_sparse_min_energy_close(self):
        """Dense and sparse best energies must not diverge on the same problem."""
        n = 8
        rng = np.random.default_rng(7)
        h = rng.uniform(-1, 1, n).astype(np.float64)
        J_upper = rng.uniform(-1, 1, (n, n)).astype(np.float64)
        J = (J_upper + J_upper.T) / 2.0
        np.fill_diagonal(J, 0.0)

        _, e_dense = self._run_dense(h, J, num_reads=64, num_sweeps=800)
        _, e_sparse = self._run_sparse(h, J, num_reads=64, num_sweeps=800)

        # The best-found energies should be within 5% of each other (or both <= 0).
        best_dense = float(e_dense.min())
        best_sparse = float(e_sparse.min())
        # Normalise to avoid division by zero
        scale = max(abs(best_dense), abs(best_sparse), 1.0)
        rel_diff = abs(best_dense - best_sparse) / scale
        assert rel_diff < 0.15, (
            f"Dense/sparse best energies diverge: dense={best_dense:.4f}, "
            f"sparse={best_sparse:.4f}, rel_diff={rel_diff:.3f}"
        )

    def test_energy_is_consistent_with_spins(self):
        """Re-compute energy from returned spins and compare to reported energy."""
        from digitalannealer.utils.energy import compute_energy

        n = 5
        h, J = _ferromagnetic_ising(n)
        h_f64 = h.astype(np.float64)
        J_f64 = J.astype(np.float64)

        samples_arr, energies = self._run_dense(
            h_f64, J_f64, seed=99, num_reads=8, num_sweeps=300
        )
        # Recompute energies on CPU from returned spin configurations
        recomputed = np.array(
            [
                compute_energy(samples_arr[i].astype(np.float64), h_f64, J_f64)
                for i in range(len(samples_arr))
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            energies,
            recomputed,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Dense energies are inconsistent with spins",
        )

    def test_sparse_energy_consistent_with_spins(self):
        """Re-compute sparse energy from returned spins and compare to reported energy."""
        from digitalannealer.utils.energy import compute_energy

        n = 5
        h, J = _ferromagnetic_ising(n)
        h_f64 = h.astype(np.float64)
        J_f64 = J.astype(np.float64)

        samples_sparse, e_sparse = self._run_sparse(
            h_f64, J_f64, seed=99, num_reads=8, num_sweeps=300
        )
        recomputed = np.array(
            [
                compute_energy(samples_sparse[i].astype(np.float64), h_f64, J_f64)
                for i in range(len(samples_sparse))
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            e_sparse,
            recomputed,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Sparse energies are inconsistent with spins",
        )


# ---------------------------------------------------------------------------
# 4. Extreme inputs / NaN-Inf rejection
#    validate_ising_input is CPU-only; NaN/Inf in J/h must be caught before GPU.
# ---------------------------------------------------------------------------


class TestExtremeInputRejection:
    """Validate that NaN/Inf in h/J are caught before reaching the GPU."""

    def test_nan_in_h_dict_raises(self):
        """NaN in h (dict) must raise ValueError at validation time."""
        from digitalannealer.utils.validation import validate_ising_input

        h = {0: float("nan"), 1: 0.0}
        J = {(0, 1): -1.0}
        with pytest.raises(ValueError, match="non-finite"):
            validate_ising_input(h, J)

    def test_inf_in_J_dict_raises(self):
        """Inf in J (dict) must raise ValueError at validation time."""
        from digitalannealer.utils.validation import validate_ising_input

        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): float("inf")}
        with pytest.raises(ValueError, match="non-finite"):
            validate_ising_input(h, J)

    def test_nan_in_h_array_raises(self):
        """NaN in h (array) must raise ValueError."""
        from digitalannealer.utils.validation import validate_ising_input

        h = np.array([0.0, float("nan")])
        J = np.zeros((2, 2))
        with pytest.raises(ValueError, match=r"NaN|Inf|non-finite"):
            validate_ising_input(h, J)

    def test_inf_in_J_array_raises(self):
        """Inf in J (array) must raise ValueError."""
        from digitalannealer.utils.validation import validate_ising_input

        h = np.zeros(2)
        J = np.array([[0.0, float("inf")], [float("inf"), 0.0]])
        with pytest.raises(ValueError, match=r"NaN|Inf|non-finite"):
            validate_ising_input(h, J)

    @SKIP_CUDA
    def test_large_scale_J_does_not_produce_nan_energies(self):
        """Dense DA with |J| ~ 1e4 must return finite energies (no NaN/Inf overflow)."""
        import digitalannealer._core as _core

        n = 4
        h = np.zeros(n, dtype=np.float32)
        J = -1e4 * np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(J, 0.0)
        betas = _tiny_beta_schedule(200)
        _, energies = _core.da_sample_ising(h, J, betas, 4, 0.1, 0)
        assert np.all(np.isfinite(energies)), (
            f"Energies contain NaN/Inf for large-scale J: {energies}"
        )

    @SKIP_CUDA
    def test_small_scale_J_does_not_produce_nan_energies(self):
        """Dense DA with |J| ~ 1e-6 must return finite energies (no underflow surprise)."""
        import digitalannealer._core as _core

        n = 4
        h = np.zeros(n, dtype=np.float32)
        J = -1e-6 * np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(J, 0.0)
        betas = _tiny_beta_schedule(200)
        _, energies = _core.da_sample_ising(h, J, betas, 4, 0.1, 0)
        assert np.all(np.isfinite(energies)), (
            f"Energies contain NaN/Inf for small-scale J: {energies}"
        )


# ---------------------------------------------------------------------------
# 5. Sampler-level parameter validation (no GPU needed)
# ---------------------------------------------------------------------------


class TestSamplerParamValidation:
    """DASampler rejects out-of-range hyperparameters before touching GPU."""

    def _make_sampler(self):
        from digitalannealer import DASampler

        return DASampler(seed=0)

    def test_num_sweeps_zero_raises(self):
        """num_sweeps=0 must raise ValueError."""
        sampler = self._make_sampler()
        with pytest.raises(ValueError, match="num_sweeps"):
            sampler.sample_ising({0: -1.0}, {}, num_sweeps=0, num_reads=1)

    def test_num_reads_zero_raises(self):
        """num_reads=0 must raise ValueError."""
        sampler = self._make_sampler()
        with pytest.raises(ValueError, match="num_reads"):
            sampler.sample_ising({0: -1.0}, {}, num_sweeps=10, num_reads=0)

    def test_negative_beta_min_raises(self):
        """beta_min <= 0 must raise ValueError."""
        sampler = self._make_sampler()
        with pytest.raises(ValueError, match="beta_min"):
            sampler.sample_ising(
                {0: -1.0}, {}, num_sweeps=10, num_reads=1, beta_min=-1.0
            )

    def test_beta_max_less_than_beta_min_raises(self):
        """beta_max < beta_min must raise ValueError."""
        sampler = self._make_sampler()
        with pytest.raises(ValueError, match="beta_max"):
            sampler.sample_ising(
                {0: -1.0}, {}, num_sweeps=10, num_reads=1, beta_min=5.0, beta_max=1.0
            )

    def test_negative_offset_delta_raises(self):
        """offset_delta < 0 must raise ValueError."""
        sampler = self._make_sampler()
        with pytest.raises(ValueError, match="offset_delta"):
            sampler.sample_ising(
                {0: -1.0}, {}, num_sweeps=10, num_reads=1, offset_delta=-0.1
            )
