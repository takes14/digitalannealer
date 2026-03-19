/**
 * @file main.cpp
 * @brief PyBind11 bindings for Digital Annealer
 *
 * Exposes C++ engine classes to Python.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../engine/da_engine.h"
#include "../engine/memory_strategy.h"
#include "../engine/pt_engine.h"

namespace py = pybind11;

namespace digitalannealer {

// ---- Type aliases for concise pybind11 array parameter declarations ----
using Int32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

template <typename FloatT>
using TypedArray = py::array_t<FloatT, py::array::c_style | py::array::forcecast>;

/**
 * @brief Validate CSR internal structure (CPU-side, before GPU launch).
 *
 * Checks:
 *  - row_offsets[0] == 0
 *  - row_offsets[n_spins] == nnz (declared nnz matches actual col_indices length)
 *  - row_offsets is monotonically non-decreasing
 *  - All col_indices values are in [0, n_spins)
 *  - No diagonal entries (col_indices[k] != row index of k)
 *  - No duplicate column indices within each row
 *  - Symmetry contract: each edge (i, j, v) must have reverse (j, i, v) exactly
 *
 * @throws std::invalid_argument on any violation.
 */
template <typename FloatT>
static void validate_csr_internal(const int32_t* row_offsets_ptr, const int32_t* col_indices_ptr,
                                  const FloatT* values_ptr, int n_spins, int nnz) {
    auto edge_key = [](int32_t i, int32_t j) -> uint64_t {
        return (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) | static_cast<uint32_t>(j);
    };

    if (row_offsets_ptr[0] != 0) {
        throw std::invalid_argument("row_offsets[0] must be 0, got " +
                                    std::to_string(row_offsets_ptr[0]));
    }
    int32_t nnz_declared = row_offsets_ptr[n_spins];
    if (nnz_declared != nnz) {
        throw std::invalid_argument("row_offsets[" + std::to_string(n_spins) +
                                    "]=" + std::to_string(nnz_declared) +
                                    " must equal len(col_indices)=" + std::to_string(nnz));
    }
    for (int i = 0; i < n_spins; ++i) {
        int32_t row_start = row_offsets_ptr[i];
        int32_t row_end = row_offsets_ptr[i + 1];
        if (row_end < row_start) {
            throw std::invalid_argument("row_offsets must be non-decreasing: row_offsets[" +
                                        std::to_string(i) + "] = " + std::to_string(row_start) +
                                        " > row_offsets[" + std::to_string(i + 1) +
                                        "] = " + std::to_string(row_end));
        }
        std::unordered_map<int32_t, int32_t> first_seen_col_index;
        first_seen_col_index.reserve(static_cast<size_t>(row_end - row_start));
        for (int32_t k = row_start; k < row_end; ++k) {
            int32_t c = col_indices_ptr[k];
            if (c < 0 || c >= n_spins) {
                throw std::invalid_argument("col_indices[" + std::to_string(k) +
                                            "] = " + std::to_string(c) + " is out of range [0, " +
                                            std::to_string(n_spins) + ")");
            }
            if (c == i) {
                throw std::invalid_argument(
                    "CSR must not contain diagonal entries: diagonal found at row " +
                    std::to_string(i) + " (col_indices[" + std::to_string(k) +
                    "] = " + std::to_string(c) + ")");
            }
            auto [it, inserted] = first_seen_col_index.emplace(c, k);
            if (!inserted) {
                throw std::invalid_argument(
                    "CSR must not contain duplicate column indices within a row: row " +
                    std::to_string(i) + " has duplicate col " + std::to_string(c) +
                    " at col_indices[" + std::to_string(k) + "] (first seen at col_indices[" +
                    std::to_string(it->second) + "])");
            }
        }
    }

    std::unordered_map<uint64_t, FloatT> edge_values;
    edge_values.reserve(static_cast<size_t>(nnz));
    for (int i = 0; i < n_spins; ++i) {
        int32_t row_start = row_offsets_ptr[i];
        int32_t row_end = row_offsets_ptr[i + 1];
        for (int32_t k = row_start; k < row_end; ++k) {
            edge_values[edge_key(i, col_indices_ptr[k])] = values_ptr[k];
        }
    }

    for (int i = 0; i < n_spins; ++i) {
        int32_t row_start = row_offsets_ptr[i];
        int32_t row_end = row_offsets_ptr[i + 1];
        for (int32_t k = row_start; k < row_end; ++k) {
            int32_t j = col_indices_ptr[k];
            FloatT v = values_ptr[k];
            auto it = edge_values.find(edge_key(j, i));
            if (it == edge_values.end()) {
                throw std::invalid_argument("CSR must be symmetric: missing reverse edge for (" +
                                            std::to_string(i) + ", " + std::to_string(j) + ")");
            }
            if (it->second != v) {
                throw std::invalid_argument(
                    "CSR symmetric edge value mismatch: (" + std::to_string(i) + ", " +
                    std::to_string(j) + ") = " + std::to_string(v) + " but (" + std::to_string(j) +
                    ", " + std::to_string(i) + ") = " + std::to_string(it->second));
            }
        }
    }
}

/**
 * @brief Validate dense h/J array shapes and return n_spins.
 * @throws std::invalid_argument on shape mismatch.
 */
template <typename FloatT>
static int validate_dense_inputs(const TypedArray<FloatT>& h, const TypedArray<FloatT>& J) {
    if (h.ndim() != 1)
        throw std::invalid_argument("h must be a 1D array");
    if (J.ndim() != 2)
        throw std::invalid_argument("J must be a 2D array");
    int n_spins = h.shape(0);
    if (J.shape(0) != n_spins || J.shape(1) != n_spins)
        throw std::invalid_argument("J dimensions must match h length");
    return n_spins;
}

/**
 * @brief Validate sparse CSR array shapes and run full structural check.
 * @returns (n_spins, nnz) pair.
 * @throws std::invalid_argument on shape mismatch or structural violation.
 */
template <typename FloatT>
static std::pair<int, int> validate_sparse_inputs(const TypedArray<FloatT>& h,
                                                  const Int32Array& row_offsets,
                                                  const Int32Array& col_indices,
                                                  const TypedArray<FloatT>& values) {
    if (h.ndim() != 1)
        throw std::invalid_argument("h must be a 1D array");
    if (row_offsets.ndim() != 1)
        throw std::invalid_argument("row_offsets must be a 1D array");
    if (col_indices.ndim() != 1)
        throw std::invalid_argument("col_indices must be a 1D array");
    if (values.ndim() != 1)
        throw std::invalid_argument("values must be a 1D array");
    int n_spins = h.shape(0);
    int nnz = col_indices.shape(0);
    if (row_offsets.shape(0) != n_spins + 1)
        throw std::invalid_argument("row_offsets length must be n_spins + 1 (got " +
                                    std::to_string(row_offsets.shape(0)) + ", expected " +
                                    std::to_string(n_spins + 1) + ")");
    if (values.shape(0) != nnz)
        throw std::invalid_argument("values length must match col_indices length");
    validate_csr_internal<FloatT>(row_offsets.data(), col_indices.data(), values.data(), n_spins,
                                  nnz);
    return {n_spins, nnz};
}

/**
 * @brief Templated implementation for DA sampling (dense)
 */
template <typename FloatT>
py::tuple da_sample_ising_typed(py::array h_arr, py::array J_arr, py::array beta_arr, int num_reads,
                                double offset_delta, int64_t seed) {
    TypedArray<FloatT> h(h_arr), J(J_arr), beta_schedule(beta_arr);
    int n_spins = validate_dense_inputs<FloatT>(h, J);
    if (beta_schedule.ndim() != 1)
        throw std::invalid_argument("beta_schedule must be a 1D array");
    int num_sweeps = beta_schedule.shape(0);

    const FloatT* h_ptr = h.data();
    const FloatT* J_ptr = J.data();
    const FloatT* beta_ptr = beta_schedule.data();

    py::array_t<int8_t> samples({num_reads, n_spins});
    py::array_t<FloatT> energies(num_reads);

    int8_t* samples_ptr = samples.mutable_data();
    FloatT* energies_ptr = energies.mutable_data();

    {
        py::gil_scoped_release release;
        da_sample_gpu<FloatT>(h_ptr, J_ptr, beta_ptr, n_spins, num_sweeps, num_reads,
                              static_cast<FloatT>(offset_delta), seed, samples_ptr, energies_ptr);
    }

    return py::make_tuple(samples, energies);
}

/**
 * @brief Python-callable wrapper for DA sampling — dispatches by dtype
 */
py::tuple da_sample_ising(py::array h, py::array J, py::array beta_schedule, int num_reads,
                          double offset_delta, int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return da_sample_ising_typed<double>(h, J, beta_schedule, num_reads, offset_delta, seed);
    }
    return da_sample_ising_typed<float>(h, J, beta_schedule, num_reads, offset_delta, seed);
}

/**
 * @brief Templated implementation for PT sampling (dense)
 */
template <typename FloatT>
py::tuple pt_sample_ising_typed(py::array h_arr, py::array J_arr, py::array betas_arr,
                                int num_sweeps, int num_reads, int exchange_interval,
                                double offset_delta, int64_t seed) {
    TypedArray<FloatT> h(h_arr), J(J_arr), betas(betas_arr);
    int n_spins = validate_dense_inputs<FloatT>(h, J);
    if (betas.ndim() != 1)
        throw std::invalid_argument("betas must be a 1D array");
    int num_replicas = betas.shape(0);

    const FloatT* h_ptr = h.data();
    const FloatT* J_ptr = J.data();
    const FloatT* betas_ptr = betas.data();

    py::array_t<int8_t> samples({num_reads, n_spins});
    py::array_t<FloatT> energies(num_reads);

    int8_t* samples_ptr = samples.mutable_data();
    FloatT* energies_ptr = energies.mutable_data();

    {
        py::gil_scoped_release release;
        pt_sample_gpu<FloatT>(h_ptr, J_ptr, betas_ptr, n_spins, num_replicas, num_sweeps, num_reads,
                              exchange_interval, static_cast<FloatT>(offset_delta), seed,
                              samples_ptr, energies_ptr);
    }

    return py::make_tuple(samples, energies);
}

/**
 * @brief Python-callable wrapper for PT sampling — dispatches by dtype
 */
py::tuple pt_sample_ising(py::array h, py::array J, py::array betas, int num_sweeps, int num_reads,
                          int exchange_interval, double offset_delta, int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return pt_sample_ising_typed<double>(h, J, betas, num_sweeps, num_reads, exchange_interval,
                                             offset_delta, seed);
    }
    return pt_sample_ising_typed<float>(h, J, betas, num_sweeps, num_reads, exchange_interval,
                                        offset_delta, seed);
}

/**
 * @brief Templated implementation for sparse DA sampling (CSR format)
 */
template <typename FloatT>
py::tuple da_sample_ising_sparse_typed(py::array h_arr, Int32Array row_offsets,
                                       Int32Array col_indices, py::array values_arr,
                                       py::array beta_arr, int num_reads, double offset_delta,
                                       int64_t seed) {
    TypedArray<FloatT> h(h_arr), values(values_arr), beta_schedule(beta_arr);
    auto [n_spins, nnz] = validate_sparse_inputs<FloatT>(h, row_offsets, col_indices, values);
    if (beta_schedule.ndim() != 1)
        throw std::invalid_argument("beta_schedule must be a 1D array");
    int num_sweeps = beta_schedule.shape(0);

    const FloatT* h_ptr = h.data();
    const int32_t* row_offsets_ptr = row_offsets.data();
    const int32_t* col_indices_ptr = col_indices.data();
    const FloatT* values_ptr = values.data();
    const FloatT* beta_ptr = beta_schedule.data();

    py::array_t<int8_t> samples({num_reads, n_spins});
    py::array_t<FloatT> energies(num_reads);

    int8_t* samples_ptr = samples.mutable_data();
    FloatT* energies_ptr = energies.mutable_data();

    {
        py::gil_scoped_release release;
        da_sample_sparse_gpu<FloatT>(
            h_ptr, row_offsets_ptr, col_indices_ptr, values_ptr, beta_ptr, n_spins, nnz, num_sweeps,
            num_reads, static_cast<FloatT>(offset_delta), seed, samples_ptr, energies_ptr);
    }

    return py::make_tuple(samples, energies);
}

/**
 * @brief Python-callable wrapper for sparse DA sampling — dispatches by dtype
 */
py::tuple da_sample_ising_sparse(py::array h, Int32Array row_offsets, Int32Array col_indices,
                                 py::array values, py::array beta_schedule, int num_reads,
                                 double offset_delta, int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return da_sample_ising_sparse_typed<double>(h, row_offsets, col_indices, values,
                                                    beta_schedule, num_reads, offset_delta, seed);
    }
    return da_sample_ising_sparse_typed<float>(h, row_offsets, col_indices, values, beta_schedule,
                                               num_reads, offset_delta, seed);
}

/**
 * @brief Templated implementation for sparse PT sampling (CSR format)
 */
template <typename FloatT>
py::tuple pt_sample_ising_sparse_typed(py::array h_arr, Int32Array row_offsets,
                                       Int32Array col_indices, py::array values_arr,
                                       py::array betas_arr, int num_sweeps, int num_reads,
                                       int exchange_interval, double offset_delta, int64_t seed) {
    TypedArray<FloatT> h(h_arr), values(values_arr), betas(betas_arr);
    auto [n_spins, nnz] = validate_sparse_inputs<FloatT>(h, row_offsets, col_indices, values);
    if (betas.ndim() != 1)
        throw std::invalid_argument("betas must be a 1D array");
    int num_replicas = betas.shape(0);

    const FloatT* h_ptr = h.data();
    const int32_t* row_offsets_ptr = row_offsets.data();
    const int32_t* col_indices_ptr = col_indices.data();
    const FloatT* values_ptr = values.data();
    const FloatT* betas_ptr = betas.data();

    py::array_t<int8_t> samples({num_reads, n_spins});
    py::array_t<FloatT> energies(num_reads);

    int8_t* samples_ptr = samples.mutable_data();
    FloatT* energies_ptr = energies.mutable_data();

    {
        py::gil_scoped_release release;
        pt_sample_sparse_gpu<FloatT>(h_ptr, row_offsets_ptr, col_indices_ptr, values_ptr, betas_ptr,
                                     n_spins, nnz, num_replicas, num_sweeps, num_reads,
                                     exchange_interval, static_cast<FloatT>(offset_delta), seed,
                                     samples_ptr, energies_ptr);
    }

    return py::make_tuple(samples, energies);
}

/**
 * @brief Python-callable wrapper for sparse PT sampling — dispatches by dtype
 */
py::tuple pt_sample_ising_sparse(py::array h, Int32Array row_offsets, Int32Array col_indices,
                                 py::array values, py::array betas, int num_sweeps, int num_reads,
                                 int exchange_interval, double offset_delta, int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return pt_sample_ising_sparse_typed<double>(h, row_offsets, col_indices, values, betas,
                                                    num_sweeps, num_reads, exchange_interval,
                                                    offset_delta, seed);
    }
    return pt_sample_ising_sparse_typed<float>(h, row_offsets, col_indices, values, betas,
                                               num_sweeps, num_reads, exchange_interval,
                                               offset_delta, seed);
}

// =============================================================================
// PT exchange statistics collection (for GPU-based beta adaptation)
// =============================================================================

/**
 * @brief Templated implementation for dense exchange stats collection
 */
template <typename FloatT>
py::array_t<FloatT> pt_exchange_stats_typed(py::array h_arr, py::array J_arr, py::array betas_arr,
                                            int num_sweeps, int num_reads, int exchange_interval,
                                            double offset_delta, int64_t seed) {
    TypedArray<FloatT> h(h_arr), J(J_arr), betas(betas_arr);
    int n_spins = validate_dense_inputs<FloatT>(h, J);
    if (betas.ndim() != 1)
        throw std::invalid_argument("betas must be a 1D array");
    int num_replicas = betas.shape(0);
    int num_pairs = num_replicas - 1;
    if (num_pairs < 1)
        throw std::invalid_argument("Need at least 2 replicas for exchange stats");

    const FloatT* h_ptr = h.data();
    const FloatT* J_ptr = J.data();
    const FloatT* betas_ptr = betas.data();

    py::array_t<FloatT> acceptance(num_pairs);
    FloatT* acceptance_ptr = acceptance.mutable_data();

    {
        py::gil_scoped_release release;
        pt_exchange_stats_gpu<FloatT>(h_ptr, J_ptr, betas_ptr, n_spins, num_replicas, num_sweeps,
                                      num_reads, exchange_interval,
                                      static_cast<FloatT>(offset_delta), seed, acceptance_ptr);
    }

    return acceptance;
}

/**
 * @brief Python-callable wrapper for dense exchange stats — dispatches by dtype
 */
py::array pt_exchange_stats(py::array h, py::array J, py::array betas, int num_sweeps,
                            int num_reads, int exchange_interval, double offset_delta,
                            int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return pt_exchange_stats_typed<double>(h, J, betas, num_sweeps, num_reads,
                                               exchange_interval, offset_delta, seed);
    }
    return pt_exchange_stats_typed<float>(h, J, betas, num_sweeps, num_reads, exchange_interval,
                                          offset_delta, seed);
}

/**
 * @brief Templated implementation for sparse exchange stats collection
 */
template <typename FloatT>
py::array_t<FloatT> pt_exchange_stats_sparse_typed(py::array h_arr, Int32Array row_offsets,
                                                   Int32Array col_indices, py::array values_arr,
                                                   py::array betas_arr, int num_sweeps,
                                                   int num_reads, int exchange_interval,
                                                   double offset_delta, int64_t seed) {
    TypedArray<FloatT> h(h_arr), values(values_arr), betas(betas_arr);
    auto [n_spins, nnz] = validate_sparse_inputs<FloatT>(h, row_offsets, col_indices, values);
    if (betas.ndim() != 1)
        throw std::invalid_argument("betas must be a 1D array");
    int num_replicas = betas.shape(0);
    int num_pairs = num_replicas - 1;
    if (num_pairs < 1)
        throw std::invalid_argument("Need at least 2 replicas for exchange stats");

    const FloatT* h_ptr = h.data();
    const int32_t* row_offsets_ptr = row_offsets.data();
    const int32_t* col_indices_ptr = col_indices.data();
    const FloatT* values_ptr = values.data();
    const FloatT* betas_ptr = betas.data();

    py::array_t<FloatT> acceptance(num_pairs);
    FloatT* acceptance_ptr = acceptance.mutable_data();

    {
        py::gil_scoped_release release;
        pt_exchange_stats_sparse_gpu<FloatT>(
            h_ptr, row_offsets_ptr, col_indices_ptr, values_ptr, betas_ptr, n_spins, nnz,
            num_replicas, num_sweeps, num_reads, exchange_interval,
            static_cast<FloatT>(offset_delta), seed, acceptance_ptr);
    }

    return acceptance;
}

/**
 * @brief Python-callable wrapper for sparse exchange stats — dispatches by dtype
 */
py::array pt_exchange_stats_sparse(py::array h, Int32Array row_offsets, Int32Array col_indices,
                                   py::array values, py::array betas, int num_sweeps, int num_reads,
                                   int exchange_interval, double offset_delta, int64_t seed) {
    if (h.dtype().is(py::dtype::of<double>())) {
        return pt_exchange_stats_sparse_typed<double>(h, row_offsets, col_indices, values, betas,
                                                      num_sweeps, num_reads, exchange_interval,
                                                      offset_delta, seed);
    }
    return pt_exchange_stats_sparse_typed<float>(h, row_offsets, col_indices, values, betas,
                                                 num_sweeps, num_reads, exchange_interval,
                                                 offset_delta, seed);
}

}  // namespace digitalannealer

PYBIND11_MODULE(_core, m) {
    m.doc() = "Digital Annealer GPU backend";

    m.def("da_sample_ising", &digitalannealer::da_sample_ising,
          "Run Digital Annealer sampling on Ising model", py::arg("h"), py::arg("J"),
          py::arg("beta_schedule"), py::arg("num_reads"), py::arg("offset_delta"), py::arg("seed"));

    m.def("pt_sample_ising", &digitalannealer::pt_sample_ising,
          "Run Parallel Tempering Digital Annealer sampling", py::arg("h"), py::arg("J"),
          py::arg("betas"), py::arg("num_sweeps"), py::arg("num_reads"),
          py::arg("exchange_interval"), py::arg("offset_delta"), py::arg("seed"));

    m.def("da_sample_ising_sparse", &digitalannealer::da_sample_ising_sparse,
          "Run Digital Annealer sampling on Ising model with sparse CSR input", py::arg("h"),
          py::arg("row_offsets"), py::arg("col_indices"), py::arg("values"),
          py::arg("beta_schedule"), py::arg("num_reads"), py::arg("offset_delta"), py::arg("seed"));

    m.def("pt_sample_ising_sparse", &digitalannealer::pt_sample_ising_sparse,
          "Run Parallel Tempering Digital Annealer sampling with sparse CSR input", py::arg("h"),
          py::arg("row_offsets"), py::arg("col_indices"), py::arg("values"), py::arg("betas"),
          py::arg("num_sweeps"), py::arg("num_reads"), py::arg("exchange_interval"),
          py::arg("offset_delta"), py::arg("seed"));

    m.def("pt_exchange_stats", &digitalannealer::pt_exchange_stats,
          "Collect per-pair exchange acceptance rates from short PT runs on GPU (dense)",
          py::arg("h"), py::arg("J"), py::arg("betas"), py::arg("num_sweeps"), py::arg("num_reads"),
          py::arg("exchange_interval"), py::arg("offset_delta"), py::arg("seed"));

    m.def("pt_exchange_stats_sparse", &digitalannealer::pt_exchange_stats_sparse,
          "Collect per-pair exchange acceptance rates from short PT runs on GPU (sparse CSR)",
          py::arg("h"), py::arg("row_offsets"), py::arg("col_indices"), py::arg("values"),
          py::arg("betas"), py::arg("num_sweeps"), py::arg("num_reads"),
          py::arg("exchange_interval"), py::arg("offset_delta"), py::arg("seed"));

    m.def(
        "check_vram_dense_da",
        [](int n_spins, int n_replicas, int num_sweeps, const std::string& dtype) {
            size_t base_required;
            if (dtype == "float64" || dtype == "double") {
                base_required =
                    digitalannealer::estimateVramDenseDA<double>(n_spins, n_replicas, num_sweeps);
            } else if (dtype == "float32" || dtype == "float") {
                base_required =
                    digitalannealer::estimateVramDenseDA<float>(n_spins, n_replicas, num_sweeps);
            } else {
                throw std::invalid_argument("Unsupported dtype '" + dtype +
                                            "'. Supported dtypes are 'float32' and 'float64'.");
            }
            const size_t required = digitalannealer::addPoolAlignmentOverhead(
                base_required, digitalannealer::kDenseDAAllocCount);
            size_t free_mem, total_mem;
            digitalannealer::getCudaMemInfoChecked(&free_mem, &total_mem);
            const size_t available_with_margin =
                static_cast<size_t>(free_mem * digitalannealer::kVramSafetyMargin);
            if (required > available_with_margin) {
                throw std::runtime_error(
                    "Estimated insufficient GPU memory for dense DA mode. "
                    "Required (including pool alignment overhead): " +
                    std::to_string(required / (1024.0 * 1024.0)) +
                    " MB, Available with safety margin: " +
                    std::to_string(available_with_margin / (1024.0 * 1024.0)) + " MB.");
            }
        },
        "Check if sufficient VRAM is available for dense DA mode (throws RuntimeError if not)",
        py::arg("n_spins"), py::arg("n_replicas"), py::arg("num_sweeps"),
        py::arg("dtype") = "float32");
}
