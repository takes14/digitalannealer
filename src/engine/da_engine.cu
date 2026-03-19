/**
 * @file da_engine.cu
 * @brief Digital Annealer Engine - CUDA implementation (optimised)
 *
 * Changes from original:
 * - DeviceMemoryPool (single cudaMalloc) replaces 8 individual allocations
 * - Philox stateless RNG eliminates d_rng_states (saves 48 × M × N bytes)
 * - MemoryConfig-driven SMR selection for dense kernel
 * - RAII cleanup via pool destructor
 * - Async memcpy + single-stream pipeline — no mid-setup syncs
 * - Cached device properties to avoid repeated driver queries
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../utils/cuda_check.cuh"
#include "../utils/cuda_helpers.cuh"
#include "../utils/cuda_stream.cuh"
#include "../utils/device_pool.cuh"
#include "../utils/device_props_cache.cuh"
#include "../utils/kernel_forward_decl.cuh"
#include "da_engine.h"
#include "memory_strategy.h"
#include "seed_constants.h"

namespace digitalannealer {

// =============================================================================
// Public API implementation
// =============================================================================

/**
 * @brief Run Digital Annealer sampling on GPU (dense mode).
 *
 * Allocates a single contiguous VRAM pool (RAII), uploads model data,
 * initializes spin states and RNG, runs the full annealing schedule inside
 * a single kernel launch, then copies results back to host.
 *
 * Pipeline (all ops on the same CUDA stream, single cudaStreamSynchronize):
 *  1. VRAM pre-check and strategy selection (SMR vs. global-memory path).
 *  2. Pool allocation and async host→device copies of h, J, beta_schedule.
 *  3. Philox-based spin initialisation (initSpinsKernel).
 *  4. Initial ΔE computation (computeInitialDeltaEKernel).
 *  5. Main annealing loop (daUpdateDenseKernel).
 *  6. Final energy computation (computeEnergyKernel).
 *  7. Async device→host copies of samples and energies.
 *
 * @tparam FloatT         float or double
 * @param h               Linear biases [n_spins].
 * @param J               Interaction matrix [n_spins × n_spins], row-major.
 * @param beta_schedule   Inverse temperature schedule [num_sweeps].
 * @param n_spins         Number of spin variables.
 * @param num_sweeps      Number of annealing sweeps.
 * @param num_reads       Number of independent runs.
 * @param offset_delta    Dynamic offset increment on rejection.
 * @param seed            Random seed (-1 for time-based entropy).
 * @param out_samples     Output spin configurations [num_reads × n_spins].
 * @param out_energies    Output energies [num_reads].
 *
 * @throws std::runtime_error if VRAM is insufficient or shared memory is exceeded.
 */
template <typename FloatT>
void da_sample_gpu(const FloatT* h, const FloatT* J, const FloatT* beta_schedule, int n_spins,
                   int num_sweeps, int num_reads, FloatT offset_delta, int64_t seed,
                   int8_t* out_samples, FloatT* out_energies) {
    // ---- Device properties (cached — avoids repeated driver queries) ----
    const cudaDeviceProp& prop = getCachedDeviceProperties();

    // ---- Memory strategy ----
    MemoryConfig cfg =
        selectDenseDAStrategy<FloatT>(n_spins, num_reads, num_sweeps, prop.sharedMemPerBlock);
    const size_t required_pool_bytes =
        addPoolAlignmentOverhead(cfg.total_pool_bytes, kDenseDAAllocCount);

    // ---- VRAM pre-check (Philox-aware, no rng_states) ----
    validateAvailableVram(required_pool_bytes, "dense DA mode (Philox)");

    // ---- Create CUDA stream for pipelining (RAII — destroyed on scope exit) ----
    CudaStream stream;

    // ---- Pool allocation ----
    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    FloatT* d_J = pool.allocate<FloatT>(static_cast<size_t>(n_spins) * n_spins);
    FloatT* d_betas = pool.allocate<FloatT>(num_sweeps);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_reads) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_reads) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_reads);
    FloatT* d_energies = pool.allocate<FloatT>(num_reads);
    // NOTE: d_rng_states is gone — Philox RNG is stateless

    // ---- Async copy model to device (ordered within stream) ----
    const size_t h_bytes = static_cast<size_t>(n_spins) * sizeof(FloatT);
    const size_t J_bytes =
        static_cast<size_t>(n_spins) * static_cast<size_t>(n_spins) * sizeof(FloatT);
    const size_t betas_bytes = static_cast<size_t>(num_sweeps) * sizeof(FloatT);

    CUDA_CHECK(cudaMemcpyAsync(d_h, h, h_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_J, J, J_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_betas, beta_schedule, betas_bytes, cudaMemcpyHostToDevice, stream));

    // ---- Seed resolution ----
    const uint64_t actual_seed = resolve_seed(seed);
    const uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

    // ---- Initialize selection RNG states (small: num_reads only) ----
    const int selection_threads = choose_threads_for_1d(prop, 256);
    const int selection_grid = (num_reads + selection_threads - 1) / selection_threads;
    initRngStates<<<selection_grid, selection_threads, 0, stream>>>(d_selection_rng, num_reads,
                                                                    selection_seed);
    CUDA_CHECK_LAST_ERROR();

    // ---- Initialize spins (Philox-based — no rng_states needed) ----
    const DenseKernelLaunchConfig dense_launch =
        makeDenseKernelLaunchConfig<FloatT>(prop, n_spins, cfg.use_dense_smr);

    // Validate shared memory against the actual warp-aligned launch configuration
    validateDenseSharedMemory(dense_launch.update_shared_mem_bytes, prop.sharedMemPerBlock);

    initSpinsKernel<<<num_reads, dense_launch.init_threads, 0, stream>>>(d_spins, n_spins,
                                                                         num_reads, actual_seed);
    CUDA_CHECK_LAST_ERROR();

    // ---- Compute initial delta_E ----
    // All previous ops are on the same stream → ordered, no explicit sync needed.
    computeInitialDeltaEKernel<FloatT><<<num_reads, dense_launch.init_threads, 0, stream>>>(
        d_spins, d_h, d_J, d_delta_e, n_spins, num_reads);
    CUDA_CHECK_LAST_ERROR();

    // ---- Run annealing ----
    daUpdateDenseKernel<FloatT>
        <<<num_reads, dense_launch.update_threads, dense_launch.update_shared_mem_bytes, stream>>>(
            d_spins, d_delta_e, d_selection_rng, d_h, d_J, d_betas, offset_delta, n_spins,
            num_sweeps,
            /*per_replica_schedule=*/false, /*replica_map=*/nullptr, actual_seed, cfg.use_dense_smr,
            /*energies=*/nullptr,
            /*sweep_offset=*/0, /*offsets=*/nullptr);
    CUDA_CHECK_LAST_ERROR();

    // ---- Compute final energies ----
    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));
    computeEnergyKernel<FloatT>
        <<<num_reads, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
            d_spins, d_h, d_J, d_energies, n_spins, num_reads);
    CUDA_CHECK_LAST_ERROR();

    // ---- Copy results (async, still on same stream → ordered after kernels) ----
    const size_t spin_bytes =
        static_cast<size_t>(num_reads) * static_cast<size_t>(n_spins) * sizeof(int8_t);
    const size_t energies_bytes = static_cast<size_t>(num_reads) * sizeof(FloatT);
    CUDA_CHECK(cudaMemcpyAsync(out_samples, d_spins, spin_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(out_energies, d_energies, energies_bytes, cudaMemcpyDeviceToHost, stream));

    // ---- Single sync — wait for the entire pipeline to finish ----
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Stream and pool destructors handle cleanup (RAII).
}

// =============================================================================
// Sparse mode implementation
// =============================================================================

/**
 * @brief Run Digital Annealer sampling on GPU (sparse CSR mode).
 *
 * Sparse mode uses the Shared Memory Resident (SMR) architecture: spins and
 * ΔE arrays are loaded into GPU shared memory at kernel start and all updates
 * are performed there for fast random access, avoiding global memory traffic.
 * This constrains the problem to what fits in shared memory (typically
 * N ≤ 4 000 spins depending on the GPU).
 *
 * Pipeline (all ops on the same CUDA stream, single cudaStreamSynchronize):
 *  1. VRAM and shared-memory pre-checks.
 *  2. Pool allocation and async host→device copies of h, CSR arrays, beta_schedule.
 *  3. Philox-based spin initialisation (initSpinsKernel).
 *  4. Initial ΔE computation (computeInitialDeltaESparseKernel).
 *  5. Main annealing loop (daUpdateSparseKernel).
 *  6. Final energy computation (computeEnergySparseKernel).
 *  7. Async device→host copies of samples and energies.
 *
 * @tparam FloatT         float or double
 * @param h               Linear biases [n_spins].
 * @param row_offsets     CSR row pointers [n_spins + 1].
 * @param col_indices     CSR column indices [nnz].
 * @param values          CSR non-zero coupling values [nnz].
 * @param beta_schedule   Inverse temperature schedule [num_sweeps].
 * @param n_spins         Number of spin variables.
 * @param nnz             Number of non-zero entries in the interaction matrix.
 * @param num_sweeps      Number of annealing sweeps.
 * @param num_reads       Number of independent runs.
 * @param offset_delta    Dynamic offset increment on rejection.
 * @param seed            Random seed (-1 for time-based entropy).
 * @param out_samples     Output spin configurations [num_reads × n_spins].
 * @param out_energies    Output energies [num_reads].
 *
 * @throws std::runtime_error if VRAM is insufficient or problem exceeds
 *         shared memory capacity.
 */
template <typename FloatT>
void da_sample_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                          const FloatT* values, const FloatT* beta_schedule, int n_spins, int nnz,
                          int num_sweeps, int num_reads, FloatT offset_delta, int64_t seed,
                          int8_t* out_samples, FloatT* out_energies) {
    const cudaDeviceProp& prop = getCachedDeviceProperties();
    const size_t pool_bytes = estimateVramSparseDA<FloatT>(n_spins, nnz, num_reads, num_sweeps);
    const size_t required_pool_bytes = addPoolAlignmentOverhead(pool_bytes, kSparseDAAllocCount);

    // Check VRAM (Philox-aware)
    validateAvailableVram(required_pool_bytes, "sparse DA mode (Philox)");

    // Check shared memory for sparse SMR
    const size_t shared_mem_size =
        validateSparseSharedMemory<FloatT>(n_spins, prop.sharedMemPerBlock);

    const int threads_per_block = choose_threads_for_spin_kernels(prop, n_spins);
    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));

    // ---- Create CUDA stream (RAII — destroyed on scope exit) ----
    CudaStream stream;

    // Pool allocation
    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    int* d_row_offsets = pool.allocate<int>(n_spins + 1);
    int* d_col_indices = pool.allocate<int>(nnz);
    FloatT* d_values = pool.allocate<FloatT>(nnz);
    FloatT* d_betas = pool.allocate<FloatT>(num_sweeps);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_reads) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_reads) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_reads);
    FloatT* d_energies = pool.allocate<FloatT>(num_reads);

    // Copy model (async on stream)
    CUDA_CHECK(cudaMemcpyAsync(d_h, h, n_spins * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_row_offsets, row_offsets, (n_spins + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_col_indices, col_indices, nnz * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_values, values, nnz * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_betas, beta_schedule, num_sweeps * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));

    // Seed
    const uint64_t actual_seed = resolve_seed(seed);
    const uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

    // Init selection RNG
    int rng_block = choose_threads_for_1d(prop, 256);
    int selection_grid = (num_reads + rng_block - 1) / rng_block;
    initRngStates<<<selection_grid, rng_block, 0, stream>>>(d_selection_rng, num_reads,
                                                            selection_seed);
    CUDA_CHECK_LAST_ERROR();

    // Init spins (Philox)
    initSpinsKernel<<<num_reads, threads_per_block, 0, stream>>>(d_spins, n_spins, num_reads,
                                                                 actual_seed);
    CUDA_CHECK_LAST_ERROR();

    // Initial delta_E
    computeInitialDeltaESparseKernel<FloatT><<<num_reads, threads_per_block, 0, stream>>>(
        d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_delta_e, n_spins, num_reads);
    CUDA_CHECK_LAST_ERROR();

    // Run annealing (sparse kernel)
    daUpdateSparseKernel<FloatT><<<num_reads, threads_per_block, shared_mem_size, stream>>>(
        d_spins, d_delta_e, d_selection_rng, d_h, d_row_offsets, d_col_indices, d_values, d_betas,
        offset_delta, n_spins, num_sweeps,
        /*per_replica_schedule=*/false, /*replica_map=*/nullptr, actual_seed,
        /*energies=*/nullptr,
        /*sweep_offset=*/0, /*offsets=*/nullptr);
    CUDA_CHECK_LAST_ERROR();

    // Energy
    computeEnergySparseKernel<FloatT>
        <<<num_reads, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
            d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_energies, n_spins, num_reads);
    CUDA_CHECK_LAST_ERROR();

    // Copy results (async)
    CUDA_CHECK(cudaMemcpyAsync(out_samples, d_spins,
                               static_cast<size_t>(num_reads) * n_spins * sizeof(int8_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(out_energies, d_energies, num_reads * sizeof(FloatT),
                               cudaMemcpyDeviceToHost, stream));

    // Single sync
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Stream and pool destructors free everything (RAII).
}

// Explicit instantiations
template void da_sample_gpu<float>(const float*, const float*, const float*, int, int, int, float,
                                   int64_t, int8_t*, float*);
template void da_sample_gpu<double>(const double*, const double*, const double*, int, int, int,
                                    double, int64_t, int8_t*, double*);
template void da_sample_sparse_gpu<float>(const float*, const int*, const int*, const float*,
                                          const float*, int, int, int, int, float, int64_t, int8_t*,
                                          float*);
template void da_sample_sparse_gpu<double>(const double*, const int*, const int*, const double*,
                                           const double*, int, int, int, int, double, int64_t,
                                           int8_t*, double*);

}  // namespace digitalannealer
