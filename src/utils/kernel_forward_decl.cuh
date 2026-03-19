/**
 * @file kernel_forward_decl.cuh
 * @brief Forward declarations for CUDA kernels used by both DA and PT engines.
 *
 * This header consolidates kernel forward declarations that are shared between
 * da_engine.cu and pt_engine.cu, eliminating duplication and ensuring consistency.
 *
 * Updated for Philox-based RNG optimisation: initSpinsKernel no longer takes
 * curandState*, and daUpdateDenseKernel has a new signature (no rng_states,
 * added rng_seed + use_smr).  daUpdateSparseKernel similarly updated.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

namespace digitalannealer {

// =============================================================================
// Common initialization kernels (defined in da_update_dense.cu)
// =============================================================================

/**
 * @brief Initialize cuRAND states for per-replica selection/exchange RNG.
 * (Per-spin RNG states are no longer needed — Philox is stateless.)
 */
__global__ void initRngStates(curandState* states, int num_states, uint64_t seed);

/**
 * @brief Initialize spins randomly to +1 or -1 using stateless Philox RNG.
 *
 * New signature (Philox): no curandState* parameter.
 */
__global__ void initSpinsKernel(int8_t* spins, int n_spins, int n_replicas, uint64_t seed);

// =============================================================================
// Dense mode kernels (defined in da_update_dense.cu)
// =============================================================================

/**
 * @brief Compute initial delta_E for all spins (dense mode)
 */
template <typename FloatT>
__global__ void computeInitialDeltaEKernel(const int8_t* spins, const FloatT* h, const FloatT* J,
                                           FloatT* delta_e, int n_spins, int n_replicas);

/**
 * @brief Main DA update kernel for dense mode (optimised).
 *
 * Key changes from original:
 * - No curandState* rng_states parameter (Philox stateless RNG)
 * - Added rng_seed (uint64_t) and use_smr (bool) parameters
 * - Supports SMR path (spins/delta_e in shared memory) when use_smr=true
 * - Uses parallel prefix-sum for candidate selection
 * - Added energies pointer for incremental per-replica energy tracking
 *   (used by PT engine; pass nullptr to disable tracking)
 * - Added sweep_offset for Philox RNG continuity across kernel re-launches
 * - Added offsets pointer for persistent dynamic offset across kernel re-launches
 */
template <typename FloatT>
__global__ void daUpdateDenseKernel(int8_t* spins, FloatT* delta_e, curandState* selection_rng,
                                    const FloatT* h, const FloatT* J, const FloatT* betas,
                                    FloatT offset_delta, int n_spins, int num_sweeps,
                                    bool per_replica_schedule, const int* replica_map,
                                    uint64_t rng_seed, bool use_smr, FloatT* energies,
                                    unsigned int sweep_offset = 0, FloatT* offsets = nullptr);

/**
 * @brief Compute energy for all replicas (dense mode)
 */
template <typename FloatT>
__global__ void computeEnergyKernel(const int8_t* spins, const FloatT* h, const FloatT* J,
                                    FloatT* energies, int n_spins, int n_replicas);

// =============================================================================
// Sparse mode kernels (defined in da_update_sparse.cu)
// =============================================================================

/**
 * @brief Compute initial delta_E for all spins (sparse CSR mode)
 */
template <typename FloatT>
__global__ void computeInitialDeltaESparseKernel(const int8_t* spins, const FloatT* h,
                                                 const int* row_offsets, const int* col_indices,
                                                 const FloatT* values, FloatT* delta_e, int n_spins,
                                                 int n_replicas);

/**
 * @brief Main DA update kernel for sparse CSR mode (optimised).
 *
 * Key changes from original:
 * - No curandState* rng_states parameter (Philox stateless RNG)
 * - Added rng_seed (uint64_t) parameter
 * - Added energies pointer for incremental per-replica energy tracking
 *   (used by PT engine; pass nullptr to disable tracking)
 * - Added sweep_offset for Philox RNG continuity across kernel re-launches
 * - Added offsets pointer for persistent dynamic offset across kernel re-launches
 */
template <typename FloatT>
__global__ void daUpdateSparseKernel(int8_t* spins, FloatT* delta_e, curandState* selection_rng,
                                     const FloatT* h, const int* row_offsets,
                                     const int* col_indices, const FloatT* values,
                                     const FloatT* betas, FloatT offset_delta, int n_spins,
                                     int num_sweeps, bool per_replica_schedule,
                                     const int* replica_map, uint64_t rng_seed, FloatT* energies,
                                     unsigned int sweep_offset = 0, FloatT* offsets = nullptr);

/**
 * @brief Compute energy for all replicas (sparse CSR mode)
 */
template <typename FloatT>
__global__ void computeEnergySparseKernel(const int8_t* spins, const FloatT* h,
                                          const int* row_offsets, const int* col_indices,
                                          const FloatT* values, FloatT* energies, int n_spins,
                                          int n_replicas);

/**
 * @brief Get shared memory size required for sparse SMR kernel
 */
template <typename FloatT>
__host__ size_t getSparseSharedMemSize(int n_spins);

// =============================================================================
// PT-specific kernels (defined in pt_exchange.cu)
// =============================================================================

/**
 * @brief Attempt replica exchanges between adjacent temperature pairs
 *
 * @param attempt_counts Optional [num_replicas-1] per-pair attempt counter (nullptr to skip)
 * @param accept_counts  Optional [num_replicas-1] per-pair accept counter (nullptr to skip)
 */
template <typename FloatT>
__global__ void ptExchangeKernel(int* replica_map, const FloatT* energies, const FloatT* betas,
                                 curandState* rng_state, int num_replicas, int exchange_parity,
                                 int* attempt_counts = nullptr, int* accept_counts = nullptr);

/**
 * @brief Initialize replica map to identity mapping
 */
__global__ void initReplicaMapKernel(int* replica_map, int num_replicas);

}  // namespace digitalannealer
