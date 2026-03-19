/**
 * @file memory_strategy.h
 * @brief Execution-mode-aware memory configuration for DA and PT engines.
 *
 * Provides a compile-time/runtime decision point for selecting:
 * - Pool vs individual cudaMalloc
 * - Philox stateless RNG vs curandState
 * - Shared Memory Resident (SMR) mode for dense kernels
 * - Shared memory sizing
 *
 * The MemoryConfig is computed once before kernel launch and threaded through
 * to the engine's allocation and kernel-launch paths.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../utils/cuda_check.cuh"
#include "../utils/cuda_helpers.cuh"
#include "../utils/device_pool.cuh"
#include "../utils/kernel_forward_decl.cuh"

namespace digitalannealer {

// =============================================================================
// Constants (migrated from vram_check.cuh)
// =============================================================================

/// Conversion factor: bytes × kBytesToMB = megabytes
constexpr double kBytesToMB = 1.0 / (1024.0 * 1024.0);

/// Safety margin for VRAM pre-checks (90% of free memory)
constexpr double kVramSafetyMargin = 0.9;

/// Conservative pool sub-allocation counts used for alignment-overhead estimates.
/// Keep these synchronized with engine allocation sites.
constexpr size_t kDenseDAAllocCount = 8;
constexpr size_t kSparseDAAllocCount = 10;
constexpr size_t kDensePTAllocCount = 12;
constexpr size_t kSparsePTAllocCount = 14;
constexpr size_t kDensePTStatsAllocCount = 14;
constexpr size_t kSparsePTStatsAllocCount = 16;

/**
 * @brief Add per-sub-allocation alignment overhead to a base VRAM estimate.
 *
 * DeviceMemoryPool aligns each sub-allocation to kPoolAlignment (256 bytes),
 * which adds up to (alloc_count × 256) bytes of padding in the worst case.
 * This function adds that overhead to the raw VRAM estimate so that
 * VRAM pre-checks are conservative and never underestimate.
 *
 * @param base_bytes  Raw VRAM estimate without alignment overhead.
 * @param alloc_count Number of sub-allocations made from the pool.
 * @return Total estimated bytes including alignment padding.
 */
inline size_t addPoolAlignmentOverhead(size_t base_bytes, size_t alloc_count) {
    return base_bytes + alloc_count * kPoolAlignment;
}

/// Query free/total device memory with error handling.
inline void getCudaMemInfoChecked(size_t* free_mem, size_t* total_mem) {
    // Clear stale asynchronous kernel errors from previous launches so VRAM
    // pre-checks do not fail with unrelated historical errors.
    (void)cudaGetLastError();

    cudaError_t err = cudaMemGetInfo(free_mem, total_mem);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at cudaMemGetInfo - ") +
                                 cudaGetErrorString(err));
    }
}

/**
 * @brief Validate that the required VRAM fits within the safety-capped free memory.
 *
 * @param required_bytes Total pool bytes required, including alignment overhead.
 * @param mode_label     Human-readable mode suffix used in the error message.
 *                       Example: "dense DA mode (Philox)".
 * @throws std::runtime_error if the requirement exceeds the safety margin.
 */
inline void validateAvailableVram(size_t required_bytes, const char* mode_label) {
    size_t free_mem = 0;
    size_t total_mem = 0;
    getCudaMemInfoChecked(&free_mem, &total_mem);
    (void)total_mem;
    if (required_bytes > static_cast<size_t>(free_mem * kVramSafetyMargin)) {
        throw std::runtime_error(
            std::string("Estimated insufficient GPU memory for ") + mode_label +
            ". Required: " + std::to_string(required_bytes * kBytesToMB) +
            " MB, Available: " + std::to_string(free_mem * kBytesToMB) + " MB.");
    }
}

/**
 * @brief Validate a dense-kernel per-block shared-memory requirement.
 *
 * @throws std::runtime_error if @p shared_mem_bytes exceeds the device limit.
 */
inline void validateDenseSharedMemory(size_t shared_mem_bytes, size_t smem_per_block) {
    if (shared_mem_bytes > smem_per_block) {
        throw std::runtime_error(
            "Per-block shared memory requirement (" + std::to_string(shared_mem_bytes) +
            " bytes) exceeds device limit (" + std::to_string(smem_per_block) + " bytes).");
    }
}

/**
 * @brief Memory configuration selected based on execution mode and problem size.
 *
 * This struct is populated by selectMemoryStrategy() and consumed by the
 * engine functions to parameterize allocation and kernel launches.
 */
struct MemoryConfig {
    /// Use shared-memory-resident path for spins/delta_e in dense kernel.
    bool use_dense_smr = false;

    /// Total shared memory bytes to request at kernel launch.
    size_t shared_mem_bytes = 0;

    /// Total VRAM bytes needed for Philox-based allocation (no d_rng_states).
    size_t total_pool_bytes = 0;
};

// ============================================================================
// VRAM size estimation helpers
// ============================================================================

/**
 * @brief Compute the total VRAM required for dense DA mode (Philox RNG).
 *
 * Philox eliminates the per-spin curandState array (48 bytes × M × N).
 * Only per-replica selection states remain (for sequential candidate selection).
 */
template <typename FloatT>
inline size_t estimateVramDenseDA(int n_spins, int num_reads, int num_sweeps) {
    const size_t N = static_cast<size_t>(n_spins);
    const size_t M = static_cast<size_t>(num_reads);

    size_t total = 0;
    total += N * sizeof(FloatT);                                // d_h
    total += N * N * sizeof(FloatT);                            // d_J
    total += static_cast<size_t>(num_sweeps) * sizeof(FloatT);  // d_betas
    total += M * N * sizeof(int8_t);                            // d_spins
    total += M * N * sizeof(FloatT);                            // d_delta_e
    total += M * sizeof(curandState);                           // d_selection_rng (kept)
    total += M * sizeof(FloatT);                                // d_energies
    return total;
}

/**
 * @brief Compute the total VRAM required for dense PT mode (Philox RNG).
 */
template <typename FloatT>
inline size_t estimateVramDensePT(int n_spins, int num_replicas, int exchange_interval) {
    const size_t N = static_cast<size_t>(n_spins);
    const size_t M = static_cast<size_t>(num_replicas);
    const size_t EI = static_cast<size_t>(exchange_interval);

    size_t total = 0;
    total += N * sizeof(FloatT);       // d_h
    total += N * N * sizeof(FloatT);   // d_J
    total += M * sizeof(FloatT);       // d_betas
    total += M * N * sizeof(int8_t);   // d_spins
    total += M * N * sizeof(FloatT);   // d_delta_e
    total += M * sizeof(curandState);  // d_selection_rng
    total += M * sizeof(FloatT);       // d_energies
    total += M * sizeof(int);          // d_replica_map
    total += M * sizeof(curandState);  // d_exchange_rng
    total += M * EI * sizeof(FloatT);  // d_pt_schedule
    total += M * sizeof(FloatT);       // d_offsets (persistent dynamic offset)
    return total;
}

/**
 * @brief Compute the total VRAM required for sparse DA mode (Philox RNG).
 */
template <typename FloatT>
inline size_t estimateVramSparseDA(int n_spins, int nnz, int num_reads, int num_sweeps) {
    const size_t N = static_cast<size_t>(n_spins);
    const size_t M = static_cast<size_t>(num_reads);

    size_t total = 0;
    total += N * sizeof(FloatT);                                // d_h
    total += static_cast<size_t>(n_spins + 1) * sizeof(int);    // d_row_offsets
    total += static_cast<size_t>(nnz) * sizeof(int);            // d_col_indices
    total += static_cast<size_t>(nnz) * sizeof(FloatT);         // d_values
    total += static_cast<size_t>(num_sweeps) * sizeof(FloatT);  // d_betas
    total += M * N * sizeof(int8_t);                            // d_spins
    total += M * N * sizeof(FloatT);                            // d_delta_e
    total += M * sizeof(curandState);                           // d_selection_rng
    total += M * sizeof(FloatT);                                // d_energies
    return total;
}

/**
 * @brief Compute the total VRAM required for sparse PT mode (Philox RNG).
 */
template <typename FloatT>
inline size_t estimateVramSparsePT(int n_spins, int nnz, int num_replicas, int exchange_interval) {
    const size_t N = static_cast<size_t>(n_spins);
    const size_t M = static_cast<size_t>(num_replicas);
    const size_t EI = static_cast<size_t>(exchange_interval);

    size_t total = 0;
    total += N * sizeof(FloatT);                              // d_h
    total += static_cast<size_t>(n_spins + 1) * sizeof(int);  // d_row_offsets
    total += static_cast<size_t>(nnz) * sizeof(int);          // d_col_indices
    total += static_cast<size_t>(nnz) * sizeof(FloatT);       // d_values
    total += M * sizeof(FloatT);                              // d_betas
    total += M * N * sizeof(int8_t);                          // d_spins
    total += M * N * sizeof(FloatT);                          // d_delta_e
    total += M * sizeof(curandState);                         // d_selection_rng
    total += M * sizeof(FloatT);                              // d_energies
    total += M * sizeof(int);                                 // d_replica_map
    total += M * sizeof(curandState);                         // d_exchange_rng
    total += M * EI * sizeof(FloatT);                         // d_pt_schedule
    total += M * sizeof(FloatT);  // d_offsets (persistent dynamic offset)
    return total;
}

// ============================================================================
// Dense shared-memory sizing
// ============================================================================

/**
 * @brief Compute shared memory bytes for the dense kernel.
 *
 * Layout (SMR):
 *   shared_spins[N]                + shared_delta_e[N]       (align to 4)
 *   shared_thread_prefix[blockDim] + shared_warp_sums[32]
 *   shared_count + shared_selected + shared_offset + shared_running_energy
 *
 * Layout (non-SMR):
 *   shared_thread_prefix[blockDim] + shared_warp_sums[32]
 *   shared_count + shared_selected + shared_offset + shared_running_energy
 *
 * @param n_spins    Problem size
 * @param use_smr    If true, include spins + delta_e in smem
 * @param block_size Thread block width (must be multiple of 32, default 1024)
 * @return           Required dynamic shared memory in bytes
 */
template <typename FloatT>
inline size_t getDenseSharedMemSize(int n_spins, bool use_smr, int block_size = 1024) {
    const size_t N = static_cast<size_t>(n_spins);
    size_t smem = 0;

    if (use_smr) {
        smem += N * sizeof(int8_t);  // shared_spins
        smem = (smem + alignof(FloatT) - 1) & ~(alignof(FloatT) - 1);
        smem += N * sizeof(FloatT);  // shared_delta_e
    }

    // Thread-level prefix sum (both paths): 1 int per thread
    smem += static_cast<size_t>(block_size) * sizeof(int);  // shared_thread_prefix
    smem += 32u * sizeof(int);                              // shared_warp_sums

    smem += sizeof(int);     // shared_count
    smem += sizeof(int);     // shared_selected
    smem += sizeof(FloatT);  // shared_offset
    smem += sizeof(FloatT);  // shared_running_energy

    return smem;
}

/**
 * @brief Dense-kernel launch parameters derived from device limits and problem size.
 */
struct DenseKernelLaunchConfig {
    int init_threads = 1;
    int update_threads = 32;
    size_t update_shared_mem_bytes = 0;
};

/// Align dense update blocks to full warps for warp-shuffle prefix sums.
inline int alignDenseUpdateThreads(int threads) {
    const int warp_aligned = (threads / 32) * 32;
    return warp_aligned > 0 ? warp_aligned : 32;
}

/**
 * @brief Build launch parameters shared by dense DA/PT update paths.
 *
 * The dense update kernel requires warp-aligned blocks for its warp-shuffle
 * prefix-sum path, while the init and initial-ΔE kernels can use the natural
 * spin-kernel thread count directly.
 */
template <typename FloatT>
inline DenseKernelLaunchConfig makeDenseKernelLaunchConfig(const cudaDeviceProp& prop, int n_spins,
                                                           bool use_dense_smr) {
    DenseKernelLaunchConfig cfg;
    cfg.init_threads = choose_threads_for_spin_kernels(prop, n_spins);
    cfg.update_threads = alignDenseUpdateThreads(cfg.init_threads);
    cfg.update_shared_mem_bytes =
        getDenseSharedMemSize<FloatT>(n_spins, use_dense_smr, cfg.update_threads);
    return cfg;
}

/**
 * @brief Decide whether the dense SMR path can be used for a given problem size.
 *
 * Returns true when the full SMR shared-memory layout fits in the device's
 * per-block shared-memory limit with some headroom.
 *
 * With the optimised layout (no per-element threshold/prefix arrays), the
 * SMR limit is approximately N ≤ 8700 on devices with 48 KB shared memory
 * (up from ~3700 with the old layout).
 */
template <typename FloatT>
inline bool canUseDenseSMR(int n_spins, size_t smem_per_block, int block_size = 1024) {
    const size_t required = getDenseSharedMemSize<FloatT>(n_spins, /*use_smr=*/true, block_size);
    // Leave 1 KB headroom for compiler-inserted shared variables
    return required + 1024 <= smem_per_block;
}

/**
 * @brief Validate sparse SMR shared-memory usage and return launch bytes.
 *
 * @param n_spins         Problem size in spins.
 * @param smem_per_block  Device shared-memory limit per block in bytes.
 * @param problem_prefix  Prefix used before the numeric spin count in the
 *                        error message (for example "Problem size N=" or "N=").
 * @throws std::runtime_error if the sparse SMR layout exceeds the device limit.
 */
template <typename FloatT>
inline size_t validateSparseSharedMemory(int n_spins, size_t smem_per_block,
                                         const char* problem_prefix = "Problem size N=") {
    const size_t shared_mem_bytes = getSparseSharedMemSize<FloatT>(n_spins);
    if (shared_mem_bytes > smem_per_block) {
        throw std::runtime_error(std::string(problem_prefix) + std::to_string(n_spins) +
                                 " exceeds shared memory capacity for sparse SMR mode.");
    }
    return shared_mem_bytes;
}

/**
 * @brief Select the memory strategy for dense DA (single-temperature) mode.
 *
 * Determines whether the Shared Memory Resident (SMR) path can be used,
 * computes the required dynamic shared-memory bytes, and estimates total
 * VRAM consumption for a dense Digital Annealer run.
 *
 * @tparam FloatT      float or double
 * @param n_spins      Number of spin variables.
 * @param num_reads    Number of independent runs (parallel replicas).
 * @param num_sweeps   Number of annealing sweeps. Affects d_betas size.
 * @param smem_per_block Device shared-memory limit per block in bytes.
 * @return Populated MemoryConfig for the DA engine.
 */
template <typename FloatT>
inline MemoryConfig selectDenseDAStrategy(int n_spins, int num_reads, int num_sweeps,
                                          size_t smem_per_block) {
    MemoryConfig cfg;
    cfg.use_dense_smr = canUseDenseSMR<FloatT>(n_spins, smem_per_block);
    cfg.shared_mem_bytes = getDenseSharedMemSize<FloatT>(n_spins, cfg.use_dense_smr);
    cfg.total_pool_bytes = estimateVramDenseDA<FloatT>(n_spins, num_reads, num_sweeps);
    return cfg;
}

/**
 * @brief Select the memory strategy for dense PT mode.
 *
 * Determines whether the Shared Memory Resident (SMR) path can be used,
 * computes the required dynamic shared-memory bytes, and estimates total
 * VRAM consumption for a dense Parallel Tempering run.
 *
 * @tparam FloatT       float or double
 * @param n_spins       Number of spin variables.
 * @param num_replicas  Number of temperature replicas.
 * @param exchange_interval Sweeps between replica exchange attempts.
 *                       Affects d_pt_schedule allocation size.
 * @param smem_per_block Device shared-memory limit per block in bytes.
 * @return Populated MemoryConfig for the PT engine.
 */
template <typename FloatT>
inline MemoryConfig selectDensePTStrategy(int n_spins, int num_replicas, int exchange_interval,
                                          size_t smem_per_block) {
    MemoryConfig cfg;
    cfg.use_dense_smr = canUseDenseSMR<FloatT>(n_spins, smem_per_block);
    cfg.shared_mem_bytes = getDenseSharedMemSize<FloatT>(n_spins, cfg.use_dense_smr);
    cfg.total_pool_bytes = estimateVramDensePT<FloatT>(n_spins, num_replicas, exchange_interval);
    return cfg;
}

}  // namespace digitalannealer
