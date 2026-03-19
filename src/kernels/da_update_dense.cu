/**
 * @file da_update_dense.cu
 * @brief Dense mode Digital Annealer update kernel (optimised)
 *
 * Optimisations over the original implementation:
 *
 * 1. **Philox stateless RNG** – eliminates the per-spin curandState array
 *    (48 bytes × M × N in global memory, read/written every sweep).
 *    Random numbers are generated on-the-fly from (seed, replica, spin, sweep).
 *
 * 2. **Shared Memory Resident (SMR) path** – when the problem fits in smem
 *    (N ≤ ~8 K on A100/A800), spins and delta_E live entirely in shared memory
 *    during the sweep loop, eliminating global memory traffic.  Falls back
 *    to the original global-memory path for larger N.
 *
 * 3. **Warp-shuffle thread-level prefix sum + Philox replay** – replaces the
 *    O(N)-element Hillis-Steele scan (which caused massive register spill via
 *    pf_tmp[] arrays) with a O(blockDim)-element prefix sum using warp shuffle
 *    intrinsics (zero local memory, only 2 __syncthreads).  Thread 0 replays
 *    Philox for at most ceil(N/blockDim) elements to pinpoint the selected spin.
 *
 * 4. **Micro-optimisations** – `__ldg()` for read-only J access,
 *    `__restrict__` pointer hints, `__launch_bounds__`, FMA usage.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils/cuda_check.cuh"
#include "../utils/rng.cuh"

namespace digitalannealer {

// ============================================================================
// Initialisation kernels (unchanged API, kept for backward compat)
// ============================================================================

/**
 * @brief Initialize cuRAND states for per-replica selection/exchange RNG.
 *
 * Each thread initializes one curandState with the given @p seed and its
 * global thread index as the sequence offset.  Only used for the small
 * per-replica selection RNG (num_reads or num_replicas states); per-spin
 * RNG states are no longer needed because Philox is stateless.
 *
 * Grid/Block: caller-determined; must collectively cover [0, n_states).
 *
 * @param states    Device array of curandState to initialize [n_states].
 * @param n_states  Number of states to initialize.
 * @param seed      64-bit base seed; each state's stream is selected by its
 *                  global thread index used as the cuRAND sequence number.
 */
__global__ void initRngStates(curandState* states, int n_states, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * @brief Initialize spins randomly to +1 or -1 using Philox RNG.
 *
 * No per-spin curandState needed — uses stateless Philox generation.
 */
__global__ void initSpinsKernel(int8_t* __restrict__ spins, int n_spins, int n_replicas,
                                uint64_t seed) {
    int replica = blockIdx.x;
    if (replica >= n_replicas)
        return;

    int8_t* my_spins = spins + replica * n_spins;

    for (int i = threadIdx.x; i < n_spins; i += blockDim.x) {
        float r = philox_uniform<float>(seed, replica, i, 0);
        my_spins[i] = (r < 0.5f) ? static_cast<int8_t>(-1) : static_cast<int8_t>(1);
    }
}

/**
 * @brief Compute initial delta_E for all spins (dense mode)
 *
 * dimod convention: E = Σ h_i s_i + Σ_{i<j} J_ij s_i s_j
 * delta_E_i = −2 s_i (h_i + Σ_{j≠i} J_ij s_j)
 */
template <typename FloatT>
__global__ void computeInitialDeltaEKernel(const int8_t* __restrict__ spins,
                                           const FloatT* __restrict__ h,
                                           const FloatT* __restrict__ J,
                                           FloatT* __restrict__ delta_e, int n_spins,
                                           int n_replicas) {
    int replica = blockIdx.x;
    if (replica >= n_replicas)
        return;

    const int8_t* my_spins = spins + replica * n_spins;
    FloatT* my_delta_e = delta_e + replica * n_spins;

    for (int i = threadIdx.x; i < n_spins; i += blockDim.x) {
        FloatT local_field = h[i];
        const size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(n_spins);
        for (int j = 0; j < n_spins; j++) {
            if (j != i) {
                local_field += __ldg(&J[row_offset + static_cast<size_t>(j)]) *
                               static_cast<FloatT>(my_spins[j]);
            }
        }
        my_delta_e[i] = static_cast<FloatT>(-2.0) * static_cast<FloatT>(my_spins[i]) * local_field;
    }
}

// ============================================================================
// Helper: block-wide inclusive prefix sum via warp shuffle
// ============================================================================

/**
 * @brief Block-wide inclusive prefix sum — 1 element per thread, zero spill.
 *
 * Uses __shfl_up_sync for intra-warp scan and a small shared-memory buffer
 * (32 ints) for inter-warp propagation.  Only 2 __syncthreads barriers,
 * no register-pressure arrays, no local memory spill.
 *
 * @param val               The value this thread contributes.
 * @param shared_warp_sums  Shared buffer of at least (blockDim.x / 32) ints.
 * @return                  Inclusive prefix sum for this thread's position.
 *
 * Requires: blockDim.x is a multiple of 32.
 */
__device__ __forceinline__ int blockPrefixSumShuffle(int val, int* shared_warp_sums) {
    const unsigned FULL = 0xFFFFFFFFu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = static_cast<int>(blockDim.x) >> 5;

// Phase 1: intra-warp inclusive scan (5 shuffle steps for warp_size=32)
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int tmp = __shfl_up_sync(FULL, val, d);
        if (lane >= d)
            val += tmp;
    }

    // Phase 2: last lane of each warp stores its warp total
    if (lane == 31)
        shared_warp_sums[warp] = val;
    __syncthreads();

    // Phase 3: warp 0 scans the warp-total array (≤ 32 elements)
    if (warp == 0) {
        int ws = (lane < num_warps) ? shared_warp_sums[lane] : 0;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int tmp = __shfl_up_sync(FULL, ws, d);
            if (lane >= d)
                ws += tmp;
        }
        if (lane < num_warps)
            shared_warp_sums[lane] = ws;
    }
    __syncthreads();

    // Phase 4: add previous warp's cumulative total
    if (warp > 0)
        val += shared_warp_sums[warp - 1];

    return val;
}

// ============================================================================
// Main dense update kernel — optimised
// ============================================================================

/**
 * @brief Main DA update kernel — dense mode, fully optimised.
 *
 * Implements the Parallel Trial Scheme with Metropolis-Hastings acceptance.
 *
 * Key design points:
 * - Philox stateless RNG (no d_rng_states parameter)
 * - SMR path when use_smr=true: spins/delta_E in shared memory
 * - Warp-shuffle thread-level prefix sum for candidate selection
 *   (1 element per thread, zero local-memory spill, 2 __syncthreads)
 * - Thread 0 Philox replay to pinpoint the exact candidate
 *   (at most ceil(N/blockDim) iterations — typically 1-8)
 * - __ldg() for read-only J matrix access
 * - __restrict__ hints on all pointer parameters
 *
 * Grid:  (n_replicas, 1, 1)    — one block per replica
 * Block: (n_threads, 1, 1)     — threads loop over spins; must be multiple of 32
 *
 * Shared memory layout (SMR mode, use_smr=true):
 *   shared_spins          [n_spins × int8_t]        (aligned to 4)
 *   shared_delta_e        [n_spins × float]
 *   shared_thread_prefix  [blockDim.x × int]
 *   shared_warp_sums      [32 × int]
 *   shared_count          [1 × int]
 *   shared_selected       [1 × int]
 *   shared_offset         [1 × float]
 *
 * Shared memory layout (non-SMR mode, use_smr=false):
 *   shared_thread_prefix  [blockDim.x × int]
 *   shared_warp_sums      [32 × int]
 *   shared_count          [1 × int]
 *   shared_selected       [1 × int]
 *   shared_offset         [1 × float]
 *   shared_running_energy [1 × float]   (incremental energy tracker for PT)
 */
template <typename FloatT>
__launch_bounds__(1024, 2) __global__ void daUpdateDenseKernel(
    int8_t* __restrict__ spins, FloatT* __restrict__ delta_e,
    curandState* selection_rng,  // [n_replicas] per-replica selection RNG
    const FloatT* __restrict__ h, const FloatT* __restrict__ J, const FloatT* __restrict__ betas,
    FloatT offset_delta, int n_spins, int num_sweeps, bool per_replica_schedule,
    const int* __restrict__ replica_map,
    uint64_t rng_seed,              // base seed for Philox RNG
    bool use_smr,                   // whether shared-memory-resident path is active
    FloatT* __restrict__ energies,  // [n_replicas] running energy (nullptr to skip)
    unsigned int sweep_offset,      // global sweep index offset for Philox RNG continuity
    FloatT* __restrict__ offsets    // [n_replicas] persistent dynamic offset (nullptr to reset)
) {
    extern __shared__ char shared_mem[];

    const int temp_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = static_cast<int>(blockDim.x);

    const int physical = (replica_map != nullptr) ? replica_map[temp_idx] : temp_idx;

    // ---- Parse shared memory layout ----
    int8_t* shared_spins = nullptr;
    FloatT* shared_delta_e = nullptr;
    int* shared_thread_prefix;
    int* shared_warp_sums;
    int* shared_count;
    int* shared_selected;
    FloatT* shared_offset;
    FloatT* shared_running_energy;

    {
        size_t off = 0;
        if (use_smr) {
            shared_spins = reinterpret_cast<int8_t*>(shared_mem + off);
            off += static_cast<size_t>(n_spins) * sizeof(int8_t);
            off = (off + alignof(FloatT) - 1) & ~(alignof(FloatT) - 1);
            shared_delta_e = reinterpret_cast<FloatT*>(shared_mem + off);
            off += static_cast<size_t>(n_spins) * sizeof(FloatT);
        }
        shared_thread_prefix = reinterpret_cast<int*>(shared_mem + off);
        off += static_cast<size_t>(bdim) * sizeof(int);
        shared_warp_sums = reinterpret_cast<int*>(shared_mem + off);
        off += 32u * sizeof(int);
        shared_count = reinterpret_cast<int*>(shared_mem + off);
        off += sizeof(int);
        shared_selected = reinterpret_cast<int*>(shared_mem + off);
        off += sizeof(int);
        shared_offset = reinterpret_cast<FloatT*>(shared_mem + off);
        off += sizeof(FloatT);
        shared_running_energy = reinterpret_cast<FloatT*>(shared_mem + off);
    }

    // ---- Global memory pointers ----
    int8_t* g_spins = spins + static_cast<size_t>(physical) * n_spins;
    FloatT* g_delta_e = delta_e + static_cast<size_t>(physical) * n_spins;
    curandState* my_selection_rng = selection_rng + physical;

    const FloatT* my_betas = per_replica_schedule ? (betas + temp_idx * num_sweeps) : betas;

    // ---- Initialize working pointers (shared or global) ----
    int8_t* s_spins;
    FloatT* s_delta_e;

    if (use_smr) {
        for (int i = tid; i < n_spins; i += bdim) {
            shared_spins[i] = g_spins[i];
            shared_delta_e[i] = g_delta_e[i];
        }
        s_spins = shared_spins;
        s_delta_e = shared_delta_e;
    } else {
        s_spins = g_spins;
        s_delta_e = g_delta_e;
    }

    if (tid == 0) {
        *shared_offset = (offsets != nullptr) ? offsets[physical] : static_cast<FloatT>(0);
        *shared_running_energy =
            (energies != nullptr) ? energies[physical] : static_cast<FloatT>(0);
    }
    __syncthreads();

    // =================================================================
    // Main annealing loop
    // =================================================================
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        FloatT beta = my_betas[sweep];
        if (beta <= static_cast<FloatT>(0))
            beta = static_cast<FloatT>(1.0e-10);
        const FloatT inv_beta = static_cast<FloatT>(1) / beta;
        const FloatT offset = *shared_offset;

        // ----------------------------------------------------------
        // Pass 1: Count candidates per thread
        // ----------------------------------------------------------
        if (tid == 0)
            *shared_count = 0;
        __syncthreads();

        const unsigned int global_sweep = sweep_offset + static_cast<unsigned int>(sweep);
        int my_count = 0;
        for (int i = tid; i < n_spins; i += bdim) {
            FloatT eff_de = s_delta_e[i] - offset;
            FloatT r = philox_uniform<FloatT>(rng_seed, physical, i, global_sweep);
            FloatT thr;
            if constexpr (std::is_same_v<FloatT, double>) {
                thr = -log(static_cast<double>(r)) * inv_beta;
            } else {
                thr = -__logf(r) * inv_beta;
            }
            if (eff_de <= thr)
                my_count++;
        }
        atomicAdd(shared_count, my_count);
        __syncthreads();

        const int total_candidates = *shared_count;
        int selected = -1;

        if (total_candidates > 0) {
            // ----------------------------------------------------------
            // Pass 2: Thread-level prefix sum  (warp shuffle, 2 syncs)
            //         + binary search to find owning thread
            //         + Philox replay to pinpoint exact spin
            // ----------------------------------------------------------
            int prefix = blockPrefixSumShuffle(my_count, shared_warp_sums);
            shared_thread_prefix[tid] = prefix;
            __syncthreads();

            if (tid == 0) {
                float u = curand_uniform(my_selection_rng);
                int target = min(static_cast<int>(u * total_candidates), total_candidates - 1);
                int target_1based = target + 1;

                // Binary search over thread-level prefix to find owning thread
                int lo = 0, hi = bdim - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (shared_thread_prefix[mid] < target_1based)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                int owning_tid = lo;
                int prior = (owning_tid > 0) ? shared_thread_prefix[owning_tid - 1] : 0;
                int local_target = target - prior;

                // Replay Philox for owning thread's elements (≤ ceil(N/blockDim))
                int cnt = 0;
                for (int i = owning_tid; i < n_spins; i += bdim) {
                    FloatT eff = s_delta_e[i] - offset;
                    FloatT r2 = philox_uniform<FloatT>(rng_seed, physical, i, global_sweep);
                    FloatT t2;
                    if constexpr (std::is_same_v<FloatT, double>) {
                        t2 = -log(static_cast<double>(r2)) * inv_beta;
                    } else {
                        t2 = -__logf(r2) * inv_beta;
                    }
                    if (eff <= t2) {
                        if (cnt == local_target) {
                            selected = i;
                            break;
                        }
                        cnt++;
                    }
                }

                *shared_selected = selected;
                if (selected >= 0)
                    *shared_offset = static_cast<FloatT>(0);
            }
            __syncthreads();

            selected = *shared_selected;

            if (selected >= 0) {
                // Update running energy before flip: E_new = E_old + delta_e[selected]
                if (tid == 0 && energies != nullptr) {
                    *shared_running_energy += s_delta_e[selected];
                }
                // Flip the selected spin
                if (tid == 0)
                    s_spins[selected] = -s_spins[selected];
                __syncthreads();

                // Differential update of delta_E
                const int8_t new_s_k = s_spins[selected];
                const size_t sel_row = static_cast<size_t>(selected) * static_cast<size_t>(n_spins);
                for (int i = tid; i < n_spins; i += bdim) {
                    if (i == selected) {
                        s_delta_e[i] = -s_delta_e[i];
                    } else {
                        FloatT j_val = __ldg(&J[sel_row + static_cast<size_t>(i)]);
                        s_delta_e[i] -= static_cast<FloatT>(4) * static_cast<FloatT>(s_spins[i]) *
                                        static_cast<FloatT>(new_s_k) * j_val;
                    }
                }
            } else {
                if (tid == 0)
                    *shared_offset = offset + offset_delta;
            }
        } else {
            // No candidates: increase offset
            if (tid == 0)
                *shared_offset = offset + offset_delta;
        }
        __syncthreads();
    }

    // ---- Write back from shared memory (SMR) ----
    if (use_smr) {
        for (int i = tid; i < n_spins; i += bdim) {
            g_spins[i] = shared_spins[i];
            g_delta_e[i] = shared_delta_e[i];
        }
    }

    // ---- Write back running energy ----
    if (tid == 0 && energies != nullptr) {
        energies[physical] = *shared_running_energy;
    }

    // ---- Write back persistent dynamic offset ----
    if (tid == 0 && offsets != nullptr) {
        offsets[physical] = *shared_offset;
    }
}

// Explicit instantiations for daUpdateDenseKernel
template __global__ void daUpdateDenseKernel<float>(int8_t*, float*, curandState*, const float*,
                                                    const float*, const float*, float, int, int,
                                                    bool, const int*, uint64_t, bool, float*,
                                                    unsigned int, float*);
template __global__ void daUpdateDenseKernel<double>(int8_t*, double*, curandState*, const double*,
                                                     const double*, const double*, double, int, int,
                                                     bool, const int*, uint64_t, bool, double*,
                                                     unsigned int, double*);

// Explicit instantiations for computeInitialDeltaEKernel
template __global__ void computeInitialDeltaEKernel<float>(const int8_t*, const float*,
                                                           const float*, float*, int, int);
template __global__ void computeInitialDeltaEKernel<double>(const int8_t*, const double*,
                                                            const double*, double*, int, int);

// ============================================================================
// Energy computation kernel (unchanged logic, added __restrict__ / __ldg)
// ============================================================================

/**
 * @brief Compute energy for all replicas (dense mode).
 *
 * Uses dimod convention: E = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j
 *
 * Energy is accumulated per-thread, stored in shared memory, and then
 * reduced with a power-of-two tree reduction.  Block size must be a
 * power of two (enforced by the caller via select_power_of_two_threads).
 *
 * Grid:  (n_replicas, 1, 1)  — one block per replica.
 * Block: power-of-two threads ≤ n_spins.
 * Shared memory: blockDim.x × sizeof(FloatT) for the reduction tree.
 *
 * @tparam FloatT      float or double
 * @param spins        Spin configurations [n_replicas × n_spins] (int8, ±1).
 * @param h            Linear biases [n_spins].
 * @param J            Interaction matrix [n_spins × n_spins], row-major.
 * @param energies     Output energies [n_replicas].
 * @param n_spins      Number of spin variables.
 * @param n_replicas   Number of replicas (grid size).
 */
template <typename FloatT>
__global__ void computeEnergyKernel(const int8_t* __restrict__ spins, const FloatT* __restrict__ h,
                                    const FloatT* __restrict__ J, FloatT* __restrict__ energies,
                                    int n_spins, int n_replicas) {
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
        const size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(n_spins);
        for (int j = i + 1; j < n_spins; j++) {
            local_energy += __ldg(&J[row_offset + static_cast<size_t>(j)]) *
                            static_cast<FloatT>(my_spins[i]) * static_cast<FloatT>(my_spins[j]);
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

// Explicit instantiations for computeEnergyKernel
template __global__ void computeEnergyKernel<float>(const int8_t*, const float*, const float*,
                                                    float*, int, int);
template __global__ void computeEnergyKernel<double>(const int8_t*, const double*, const double*,
                                                     double*, int, int);

}  // namespace digitalannealer
