/**
 * @file da_update_sparse.cu
 * @brief Sparse mode Digital Annealer update kernel (CSR format, optimised)
 *
 * Changes from original:
 * - Philox stateless RNG (no per-spin curandState parameter)
 * - Added rng_seed parameter for Philox generation
 * - __restrict__ pointer hints
 * - __ldg() for read-only CSR data access
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils/cuda_check.cuh"
#include "../utils/rng.cuh"

namespace digitalannealer {

namespace {

/**
 * @brief Align @p offset up to the next multiple of @p alignment.
 *
 * Used for shared-memory layout computations in getSparseSharedMemSize and
 * in the sparse kernel's runtime shared-memory pointer arithmetic to satisfy
 * type-alignment requirements.
 *
 * @param offset    Current byte offset.
 * @param alignment Required byte alignment (must be a power of two).
 * @return          Smallest value >= @p offset that is a multiple of @p alignment.
 */
__host__ __device__ inline size_t alignOffset(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

}  // namespace

/**
 * @brief Compute initial delta_E for all spins using CSR format.
 *
 * dimod convention: delta_E_i = −2 s_i (h_i + Σ_{j∈neighbours(i)} J_ij s_j)
 * Each thread handles one or more spins in a grid-stride loop, traversing
 * the sparse neighbourhood stored in CSR row @p row_offsets / @p col_indices.
 *
 * Grid:  (n_replicas, 1, 1)
 * Block: up to n_spins threads (grid-stride loop for larger problems).
 *
 * @tparam FloatT        float or double
 * @param spins          Spin configurations [n_replicas × n_spins] (int8, ±1).
 * @param h              Linear biases [n_spins].
 * @param row_offsets    CSR row pointers [n_spins + 1].
 * @param col_indices    CSR column indices [nnz].
 * @param values         CSR non-zero coupling values [nnz].
 * @param delta_e        Output ΔE values [n_replicas × n_spins].
 * @param n_spins        Number of spin variables.
 * @param n_replicas     Number of replicas (grid size).
 */
template <typename FloatT>
__global__ void computeInitialDeltaESparseKernel(
    const int8_t* __restrict__ spins, const FloatT* __restrict__ h,
    const int* __restrict__ row_offsets, const int* __restrict__ col_indices,
    const FloatT* __restrict__ values, FloatT* __restrict__ delta_e, int n_spins, int n_replicas) {
    int replica = blockIdx.x;
    if (replica >= n_replicas)
        return;

    const int8_t* my_spins = spins + replica * n_spins;
    FloatT* my_delta_e = delta_e + replica * n_spins;

    for (int i = threadIdx.x; i < n_spins; i += blockDim.x) {
        FloatT local_field = h[i];
        int row_start = __ldg(&row_offsets[i]);
        int row_end = __ldg(&row_offsets[i + 1]);
        for (int idx = row_start; idx < row_end; idx++) {
            int j = __ldg(&col_indices[idx]);
            local_field += __ldg(&values[idx]) * static_cast<FloatT>(my_spins[j]);
        }
        my_delta_e[i] = static_cast<FloatT>(-2.0) * static_cast<FloatT>(my_spins[i]) * local_field;
    }
}

/**
 * @brief Main sparse DA update kernel — optimised with Philox RNG.
 *
 * Uses Shared Memory Resident (SMR) architecture for spins and delta_E.
 * Philox eliminates per-spin curandState — no d_rng_states parameter.
 *
 * Grid: (n_replicas, 1, 1)
 * Block: (n_threads, 1, 1) where n_threads >= n_spins (or loop if larger)
 */
template <typename FloatT>
__launch_bounds__(1024, 1) __global__ void daUpdateSparseKernel(
    int8_t* __restrict__ spins, FloatT* __restrict__ delta_e,
    curandState* selection_rng,  // [n_replicas] per-replica selection RNG
    const FloatT* __restrict__ h, const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices, const FloatT* __restrict__ values,
    const FloatT* __restrict__ betas, FloatT offset_delta, int n_spins, int num_sweeps,
    bool per_replica_schedule, const int* __restrict__ replica_map,
    uint64_t rng_seed,              // base seed for Philox RNG
    FloatT* __restrict__ energies,  // [n_replicas] running energy (nullptr to skip)
    unsigned int sweep_offset,      // global sweep index offset for Philox RNG continuity
    FloatT* __restrict__ offsets    // [n_replicas] persistent dynamic offset (nullptr to reset)
) {
    extern __shared__ char shared_mem[];

    size_t off = 0;

    int8_t* shared_spins = reinterpret_cast<int8_t*>(shared_mem + off);
    off += static_cast<size_t>(n_spins) * sizeof(int8_t);

    off = alignOffset(off, alignof(FloatT));
    FloatT* shared_delta_e = reinterpret_cast<FloatT*>(shared_mem + off);
    off += static_cast<size_t>(n_spins) * sizeof(FloatT);

    off = alignOffset(off, alignof(FloatT));
    FloatT* shared_thresholds = reinterpret_cast<FloatT*>(shared_mem + off);
    off += static_cast<size_t>(n_spins) * sizeof(FloatT);

    off = alignOffset(off, alignof(int));
    int* shared_count = reinterpret_cast<int*>(shared_mem + off);
    off += sizeof(int);

    off = alignOffset(off, alignof(int));
    int* shared_selected = reinterpret_cast<int*>(shared_mem + off);
    off += sizeof(int);

    off = alignOffset(off, alignof(FloatT));
    FloatT* shared_offset = reinterpret_cast<FloatT*>(shared_mem + off);
    off += sizeof(FloatT);

    off = alignOffset(off, alignof(FloatT));
    FloatT* shared_running_energy = reinterpret_cast<FloatT*>(shared_mem + off);

    int temp_idx = blockIdx.x;
    int tid = threadIdx.x;

    int physical = (replica_map != nullptr) ? replica_map[temp_idx] : temp_idx;

    int8_t* my_spins_global = spins + physical * n_spins;
    FloatT* my_delta_e_global = delta_e + physical * n_spins;
    curandState* my_selection_rng = selection_rng + physical;

    const FloatT* my_betas = per_replica_schedule ? (betas + temp_idx * num_sweeps) : betas;

    // Load data from global memory to shared memory
    for (int i = tid; i < n_spins; i += blockDim.x) {
        shared_spins[i] = my_spins_global[i];
        shared_delta_e[i] = my_delta_e_global[i];
    }

    if (tid == 0) {
        *shared_offset = (offsets != nullptr) ? offsets[physical] : static_cast<FloatT>(0);
        *shared_running_energy =
            (energies != nullptr) ? energies[physical] : static_cast<FloatT>(0);
    }
    __syncthreads();

    // Main annealing loop
#pragma unroll 4
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        FloatT beta = my_betas[sweep];
        if (beta <= static_cast<FloatT>(0))
            beta = static_cast<FloatT>(1.0e-10);
        FloatT inv_beta = static_cast<FloatT>(1) / beta;
        FloatT offset = *shared_offset;

        // Pass 1: Generate thresholds and count candidates
        if (tid == 0) {
            *shared_count = 0;
        }
        __syncthreads();

        const unsigned int global_sweep = sweep_offset + static_cast<unsigned int>(sweep);
        int local_count = 0;
        for (int i = tid; i < n_spins; i += blockDim.x) {
            FloatT effective_delta_e = shared_delta_e[i] - offset;

            // Stateless Philox RNG — no global memory read/write
            FloatT r = philox_uniform<FloatT>(rng_seed, physical, i, global_sweep);
            FloatT threshold;
            if constexpr (std::is_same_v<FloatT, double>) {
                threshold = -log(static_cast<double>(r)) * inv_beta;
            } else {
                threshold = -__logf(r) * inv_beta;
            }

            shared_thresholds[i] = threshold;

            if (effective_delta_e <= threshold) {
                local_count++;
            }
        }

        atomicAdd(shared_count, local_count);
        __syncthreads();

        int total_candidates = *shared_count;
        int selected = -1;

        if (total_candidates > 0) {
            // Pass 2: Thread 0 selects uniformly at random from ALL accepted candidates
            if (tid == 0) {
                float u = curand_uniform(my_selection_rng);
                int target = min(static_cast<int>(u * total_candidates), total_candidates - 1);

                int count = 0;
                for (int i = 0; i < n_spins; ++i) {
                    FloatT effective_delta_e = shared_delta_e[i] - offset;
                    if (effective_delta_e <= shared_thresholds[i]) {
                        if (count == target) {
                            selected = i;
                            break;
                        }
                        ++count;
                    }
                }

                *shared_selected = selected;
                if (selected >= 0) {
                    *shared_offset = static_cast<FloatT>(0);
                }
            }
            __syncthreads();

            selected = *shared_selected;

            if (selected >= 0) {
                // Update running energy before flip: E_new = E_old + delta_e[selected]
                if (tid == 0 && energies != nullptr) {
                    *shared_running_energy += shared_delta_e[selected];
                }

                if (tid == 0) {
                    shared_spins[selected] = -shared_spins[selected];
                }
                __syncthreads();

                int8_t new_s_k = shared_spins[selected];

                if (tid == 0) {
                    shared_delta_e[selected] = -shared_delta_e[selected];
                }
                __syncthreads();

                int row_start = __ldg(&row_offsets[selected]);
                int row_end = __ldg(&row_offsets[selected + 1]);

                for (int idx = row_start + tid; idx < row_end; idx += blockDim.x) {
                    int j = __ldg(&col_indices[idx]);
                    FloatT J_val = __ldg(&values[idx]);
                    shared_delta_e[j] -= static_cast<FloatT>(4) *
                                         static_cast<FloatT>(shared_spins[j]) *
                                         static_cast<FloatT>(new_s_k) * J_val;
                }
            } else {
                if (tid == 0) {
                    *shared_offset = offset + offset_delta;
                }
            }
        } else {
            if (tid == 0) {
                *shared_offset = offset + offset_delta;
            }
        }
        __syncthreads();
    }

    // Write results back to global memory
    for (int i = tid; i < n_spins; i += blockDim.x) {
        my_spins_global[i] = shared_spins[i];
        my_delta_e_global[i] = shared_delta_e[i];
    }

    // Write back running energy
    if (tid == 0 && energies != nullptr) {
        energies[physical] = *shared_running_energy;
    }

    // Write back persistent dynamic offset
    if (tid == 0 && offsets != nullptr) {
        offsets[physical] = *shared_offset;
    }
}

/**
 * @brief Compute energy for all replicas using CSR format.
 *
 * dimod convention: E = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j
 * Each thread accumulates partial sums; results are reduced via a
 * power-of-two tree reduction in shared memory.
 *
 * Grid:  (n_replicas, 1, 1)
 * Block: power-of-two threads ≤ n_spins.
 * Shared memory: blockDim.x × sizeof(FloatT) for the reduction tree.
 *
 * @tparam FloatT        float or double
 * @param spins          Spin configurations [n_replicas × n_spins] (int8, ±1).
 * @param h              Linear biases [n_spins].
 * @param row_offsets    CSR row pointers [n_spins + 1].
 * @param col_indices    CSR column indices [nnz].
 * @param values         CSR non-zero coupling values [nnz].
 * @param energies       Output energies [n_replicas].
 * @param n_spins        Number of spin variables.
 * @param n_replicas     Number of replicas (grid size).
 */
template <typename FloatT>
__global__ void computeEnergySparseKernel(
    const int8_t* __restrict__ spins, const FloatT* __restrict__ h,
    const int* __restrict__ row_offsets, const int* __restrict__ col_indices,
    const FloatT* __restrict__ values, FloatT* __restrict__ energies, int n_spins, int n_replicas) {
    int replica = blockIdx.x;
    if (replica >= n_replicas)
        return;

    const int8_t* my_spins = spins + replica * n_spins;
    extern __shared__ char shared_energy_raw[];
    FloatT* shared_energy = reinterpret_cast<FloatT*>(shared_energy_raw);

    FloatT local_energy = static_cast<FloatT>(0);

    for (int i = threadIdx.x; i < n_spins; i += blockDim.x) {
        local_energy += h[i] * static_cast<FloatT>(my_spins[i]);
    }

    for (int i = threadIdx.x; i < n_spins; i += blockDim.x) {
        int row_start = __ldg(&row_offsets[i]);
        int row_end = __ldg(&row_offsets[i + 1]);
        for (int idx = row_start; idx < row_end; idx++) {
            int j = __ldg(&col_indices[idx]);
            if (i < j) {
                local_energy += __ldg(&values[idx]) * static_cast<FloatT>(my_spins[i]) *
                                static_cast<FloatT>(my_spins[j]);
            }
        }
    }

    shared_energy[threadIdx.x] = local_energy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_energy[threadIdx.x] += shared_energy[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        energies[replica] = shared_energy[0];
    }
}

/**
 * @brief Check if problem fits in shared memory for SMR mode.
 *
 * Computes the total dynamic shared memory required by daUpdateSparseKernel
 * for a problem of @p n_spins spins.  The caller compares this against the
 * device's per-block shared-memory limit to decide whether sparse mode is
 * feasible.
 *
 * Layout:
 *   shared_spins          [n_spins × int8_t]   (aligned to FloatT)
 *   shared_delta_e        [n_spins × FloatT]   (aligned to FloatT)
 *   shared_thresholds     [n_spins × FloatT]   (aligned to FloatT)
 *   shared_count          [1 × int]            (aligned to int)
 *   shared_selected       [1 × int]            (aligned to int)
 *   shared_offset         [1 × FloatT]         (aligned to FloatT)
 *   shared_running_energy [1 × FloatT]         (aligned to FloatT)
 *
 * @tparam FloatT  float or double
 * @param n_spins  Number of spin variables.
 * @return         Required dynamic shared memory in bytes.
 */
template <typename FloatT>
__host__ size_t getSparseSharedMemSize(int n_spins) {
    size_t smem_size = 0;

    smem_size += static_cast<size_t>(n_spins) * sizeof(int8_t);  // shared_spins

    smem_size = alignOffset(smem_size, alignof(FloatT));
    smem_size += static_cast<size_t>(n_spins) * sizeof(FloatT);  // shared_delta_e

    smem_size = alignOffset(smem_size, alignof(FloatT));
    smem_size += static_cast<size_t>(n_spins) * sizeof(FloatT);  // shared_thresholds

    smem_size = alignOffset(smem_size, alignof(int));
    smem_size += sizeof(int);  // shared_count

    smem_size = alignOffset(smem_size, alignof(int));
    smem_size += sizeof(int);  // shared_selected

    smem_size = alignOffset(smem_size, alignof(FloatT));
    smem_size += sizeof(FloatT);  // shared_offset

    smem_size = alignOffset(smem_size, alignof(FloatT));
    smem_size += sizeof(FloatT);  // shared_running_energy

    return smem_size;
}

// Explicit instantiations
template __global__ void computeInitialDeltaESparseKernel<float>(const int8_t*, const float*,
                                                                 const int*, const int*,
                                                                 const float*, float*, int, int);
template __global__ void computeInitialDeltaESparseKernel<double>(const int8_t*, const double*,
                                                                  const int*, const int*,
                                                                  const double*, double*, int, int);

template __global__ void daUpdateSparseKernel<float>(int8_t*, float*, curandState*, const float*,
                                                     const int*, const int*, const float*,
                                                     const float*, float, int, int, bool,
                                                     const int*, uint64_t, float*, unsigned int,
                                                     float*);
template __global__ void daUpdateSparseKernel<double>(int8_t*, double*, curandState*, const double*,
                                                      const int*, const int*, const double*,
                                                      const double*, double, int, int, bool,
                                                      const int*, uint64_t, double*, unsigned int,
                                                      double*);

template __global__ void computeEnergySparseKernel<float>(const int8_t*, const float*, const int*,
                                                          const int*, const float*, float*, int,
                                                          int);
template __global__ void computeEnergySparseKernel<double>(const int8_t*, const double*, const int*,
                                                           const int*, const double*, double*, int,
                                                           int);

template size_t getSparseSharedMemSize<float>(int);
template size_t getSparseSharedMemSize<double>(int);

}  // namespace digitalannealer
