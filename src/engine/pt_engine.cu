/**
 * @file pt_engine.cu
 * @brief Parallel Tempering Digital Annealer Engine - CUDA implementation (optimised)
 *
 * Changes from original:
 * - DeviceMemoryPool (RAII) replaces 11+ individual cudaMalloc/Free calls
 * - Philox stateless RNG eliminates d_rng_states (saves 48 × M × N bytes VRAM)
 * - MemoryConfig-driven dense SMR selection
 * - initSpinsKernel uses Philox (no rng_states param)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../utils/cuda_check.cuh"
#include "../utils/cuda_helpers.cuh"
#include "../utils/cuda_stream.cuh"
#include "../utils/device_pool.cuh"
#include "../utils/device_props_cache.cuh"
#include "../utils/kernel_forward_decl.cuh"
#include "da_engine.h"
#include "memory_strategy.h"
#include "pt_engine.h"
#include "seed_constants.h"

namespace digitalannealer {

// =============================================================================
// Internal helpers
// =============================================================================

/**
 * @brief Build a flat per-replica-per-sweep beta schedule for a single PT interval.
 *
 * Each replica keeps a constant temperature within an exchange interval, so
 * its schedule is @p sweeps copies of its fixed beta value.  The output is
 * laid out as [replica_0_beta × sweeps | replica_1_beta × sweeps | ...] and
 * is uploaded to d_pt_schedule once per interval.
 *
 * @tparam FloatT     float or double
 * @param betas       Per-replica inverse temperatures [num_replicas].
 * @param num_replicas Number of replicas.
 * @param sweeps      Number of sweeps in this interval.
 * @param schedule    Output vector, resized to (num_replicas × sweeps).
 */
template <typename FloatT>
static void buildConstantBetaSchedule(const FloatT* betas, int num_replicas, int sweeps,
                                      std::vector<FloatT>& schedule) {
    schedule.resize(static_cast<size_t>(num_replicas) * sweeps);
    for (int r = 0; r < num_replicas; r++) {
        const FloatT beta_r = betas[r];
        for (int s = 0; s < sweeps; s++) {
            schedule[r * sweeps + s] = beta_r;
        }
    }
}

/**
 * @brief Compute bounded adaptation warmup length in exchange cycles.
 *
 * Uses ~10% of available exchange cycles with a small minimum burn-in to
 * improve equilibration while preserving enough counted exchanges.
 */
static int computeAdaptWarmupExchangeCycles(int num_sweeps, int exchange_interval) {
    if (num_sweeps <= 0 || exchange_interval <= 0)
        return 0;
    const int total_exchange_cycles = std::max(0, (num_sweeps - 1) / exchange_interval);
    if (total_exchange_cycles <= 0)
        return 0;
    // If there is only a single exchange cycle, treat it as fully counted (no warmup)
    if (total_exchange_cycles == 1)
        return 0;

    constexpr int kWarmupMinExchangeCycles = 2;
    constexpr int kWarmupFractionDen = 10;  // ~10%
    const int warmup_from_fraction =
        (total_exchange_cycles + (kWarmupFractionDen - 1)) / kWarmupFractionDen;
    const int warmup_before_clamp = std::max(kWarmupMinExchangeCycles, warmup_from_fraction);
    // Ensure at least one exchange cycle remains after warmup
    const int max_warmup = total_exchange_cycles - 1;
    return std::min(max_warmup, warmup_before_clamp);
}

/**
 * @brief Launch even- and odd-parity replica exchange kernel pairs on a stream.
 *
 * Runs two back-to-back ptExchangeKernel launches (parity 0 then parity 1)
 * to cover all adjacent replica pairs without index conflicts.  Both launches
 * are enqueued on @p stream so they are ordered relative to the preceding
 * update kernel.
 *
 * @tparam FloatT              float or double
 * @param d_replica_map        Device array mapping temp_idx → physical replica [num_replicas].
 * @param d_energies           Device energies for each physical replica [num_replicas].
 * @param d_betas              Device inverse temperatures [num_replicas].
 * @param d_exchange_rng       Per-temperature curandState for exchange decisions [num_replicas].
 * @param num_replicas         Total number of replicas.
 * @param threads_per_block    Block size for the exchange kernels.
 * @param stream               CUDA stream to enqueue the launches on.
 * @param d_attempt_counts     Optional device array [num_replicas-1] for attempt stats.
 * @param d_accept_counts      Optional device array [num_replicas-1] for accept stats.
 */
template <typename FloatT>
static void launchPtExchangePairs(int* d_replica_map, const FloatT* d_energies,
                                  const FloatT* d_betas, curandState* d_exchange_rng,
                                  int num_replicas, int threads_per_block, cudaStream_t stream,
                                  int* d_attempt_counts = nullptr, int* d_accept_counts = nullptr) {
    const int even_pairs = num_replicas / 2;
    if (even_pairs > 0) {
        const int grid = (even_pairs + threads_per_block - 1) / threads_per_block;
        ptExchangeKernel<FloatT><<<grid, threads_per_block, 0, stream>>>(
            d_replica_map, d_energies, d_betas, d_exchange_rng, num_replicas, 0, d_attempt_counts,
            d_accept_counts);
        CUDA_CHECK_LAST_ERROR();
    }
    const int odd_pairs = (num_replicas - 1) / 2;
    if (odd_pairs > 0) {
        const int grid = (odd_pairs + threads_per_block - 1) / threads_per_block;
        ptExchangeKernel<FloatT><<<grid, threads_per_block, 0, stream>>>(
            d_replica_map, d_energies, d_betas, d_exchange_rng, num_replicas, 1, d_attempt_counts,
            d_accept_counts);
        CUDA_CHECK_LAST_ERROR();
    }
}

// =============================================================================
// GPU-based exchange statistics collection (dense)
// =============================================================================

/**
 * @brief Run short PT rounds on GPU and return per-pair exchange acceptance rates.
 *
 * This function is used during beta adaptation. It runs the full PT pipeline
 * on GPU (init + sweeps + exchanges) for the specified number of sweeps/reads,
 * collecting exchange attempt/accept counts, then computes acceptance rates.
 *
 * @tparam FloatT           float or double
 * @param h                 Linear biases [n_spins].
 * @param J                 Interaction matrix [n_spins × n_spins].
 * @param betas             Per-replica inverse temperatures [num_replicas].
 * @param n_spins           Number of spin variables.
 * @param num_replicas      Number of replicas.
 * @param num_sweeps        Number of sweeps per adaptation read.
 * @param num_reads         Number of reads to average over.
 * @param exchange_interval Sweeps between exchanges.
 * @param offset_delta      Dynamic offset increment.
 * @param seed              RNG seed.
 * @param out_acceptance    Output acceptance rates [num_replicas-1].
 */
template <typename FloatT>
void pt_exchange_stats_gpu(const FloatT* h, const FloatT* J, const FloatT* betas, int n_spins,
                           int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                           FloatT offset_delta, int64_t seed, FloatT* out_acceptance) {
    const cudaDeviceProp& prop = getCachedDeviceProperties();
    MemoryConfig cfg = selectDensePTStrategy<FloatT>(n_spins, num_replicas, exchange_interval,
                                                     prop.sharedMemPerBlock);

    const int num_pairs = num_replicas - 1;
    // Extra VRAM for stats arrays
    const size_t stats_bytes = 2 * static_cast<size_t>(num_pairs) * sizeof(int);
    const size_t required_pool_bytes =
        addPoolAlignmentOverhead(cfg.total_pool_bytes + stats_bytes, kDensePTStatsAllocCount);

    // VRAM pre-check (mirrors pt_sample_gpu behavior)
    validateAvailableVram(required_pool_bytes, "dense PT exchange stats");

    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    FloatT* d_J = pool.allocate<FloatT>(static_cast<size_t>(n_spins) * n_spins);
    FloatT* d_betas = pool.allocate<FloatT>(num_replicas);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_replicas) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_energies = pool.allocate<FloatT>(num_replicas);
    int* d_replica_map = pool.allocate<int>(num_replicas);
    curandState* d_exchange_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_pt_schedule =
        pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * exchange_interval);
    FloatT* d_offsets = pool.allocate<FloatT>(num_replicas);
    int* d_attempt_counts = pool.allocate<int>(num_pairs);
    int* d_accept_counts = pool.allocate<int>(num_pairs);

    CudaStream stream;

    // Upload model data
    CUDA_CHECK(cudaMemcpyAsync(d_h, h, n_spins * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_J, J, static_cast<size_t>(n_spins) * n_spins * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_betas, betas, num_replicas * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));

    // Build PT schedule
    std::vector<FloatT> pt_schedule;
    buildConstantBetaSchedule(betas, num_replicas, exchange_interval, pt_schedule);
    CUDA_CHECK(
        cudaMemcpyAsync(d_pt_schedule, pt_schedule.data(),
                        static_cast<size_t>(num_replicas) * exchange_interval * sizeof(FloatT),
                        cudaMemcpyHostToDevice, stream));

    // Zero stats
    CUDA_CHECK(cudaMemsetAsync(d_attempt_counts, 0, num_pairs * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_accept_counts, 0, num_pairs * sizeof(int), stream));

    // Kernel params
    const int threads_for_replica_kernels = choose_threads_for_1d(prop, 256);
    const DenseKernelLaunchConfig dense_launch =
        makeDenseKernelLaunchConfig<FloatT>(prop, n_spins, cfg.use_dense_smr);
    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));
    const int grid_size_replicas =
        (num_replicas + threads_for_replica_kernels - 1) / threads_for_replica_kernels;

    // Warmup: skip a bounded initial burn-in before counting exchange stats.
    const int warmup_exchange_cycles =
        computeAdaptWarmupExchangeCycles(num_sweeps, exchange_interval);

    // Run short PT reads, accumulating exchange stats
    for (int read = 0; read < num_reads; read++) {
        uint64_t actual_seed = generate_pt_actual_seed(seed, read);
        uint64_t exchange_seed = generate_pt_exchange_seed(seed, read);
        uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_exchange_rng, num_replicas, exchange_seed);
        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_selection_rng, num_replicas, selection_seed);
        CUDA_CHECK_LAST_ERROR();

        initReplicaMapKernel<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_replica_map, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        initSpinsKernel<<<num_replicas, dense_launch.init_threads, 0, stream>>>(
            d_spins, n_spins, num_replicas, actual_seed);
        CUDA_CHECK_LAST_ERROR();

        computeInitialDeltaEKernel<FloatT><<<num_replicas, dense_launch.init_threads, 0, stream>>>(
            d_spins, d_h, d_J, d_delta_e, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        computeEnergyKernel<FloatT>
            <<<num_replicas, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
                d_spins, d_h, d_J, d_energies, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(cudaMemsetAsync(d_offsets, 0, num_replicas * sizeof(FloatT), stream));

        int sweeps_done = 0;
        int exchange_cycles = 0;
        while (sweeps_done < num_sweeps) {
            const int sweeps_this_round = std::min(exchange_interval, num_sweeps - sweeps_done);

            if (sweeps_this_round < exchange_interval) {
                std::vector<FloatT> partial_schedule;
                buildConstantBetaSchedule(betas, num_replicas, sweeps_this_round, partial_schedule);
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pt_schedule, partial_schedule.data(),
                    static_cast<size_t>(num_replicas) * sweeps_this_round * sizeof(FloatT),
                    cudaMemcpyHostToDevice, stream));
            }

            daUpdateDenseKernel<FloatT>
                <<<num_replicas, dense_launch.update_threads, dense_launch.update_shared_mem_bytes,
                   stream>>>(d_spins, d_delta_e, d_selection_rng, d_h, d_J, d_pt_schedule,
                             offset_delta, n_spins, sweeps_this_round,
                             /*per_replica_schedule=*/true, d_replica_map, actual_seed,
                             cfg.use_dense_smr, d_energies, sweeps_done, d_offsets);
            CUDA_CHECK_LAST_ERROR();

            sweeps_done += sweeps_this_round;

            if (sweeps_done < num_sweeps) {
                exchange_cycles++;
                const bool past_warmup = (exchange_cycles > warmup_exchange_cycles);
                launchPtExchangePairs<FloatT>(d_replica_map, d_energies, d_betas, d_exchange_rng,
                                              num_replicas, threads_for_replica_kernels, stream,
                                              past_warmup ? d_attempt_counts : nullptr,
                                              past_warmup ? d_accept_counts : nullptr);
            }
        }
    }

    // Copy stats to host and compute acceptance rates
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> h_attempts(num_pairs);
    std::vector<int> h_accepted(num_pairs);
    CUDA_CHECK(cudaMemcpy(h_attempts.data(), d_attempt_counts, num_pairs * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accepted.data(), d_accept_counts, num_pairs * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_pairs; i++) {
        out_acceptance[i] =
            (h_attempts[i] > 0)
                ? static_cast<FloatT>(static_cast<double>(h_accepted[i]) / h_attempts[i])
                : static_cast<FloatT>(0);
    }
}

// =============================================================================
// GPU-based exchange statistics collection (sparse)
// =============================================================================

/**
 * @brief Run short PT rounds on GPU (sparse CSR) and return per-pair exchange acceptance rates.
 */
template <typename FloatT>
void pt_exchange_stats_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                                  const FloatT* values, const FloatT* betas, int n_spins, int nnz,
                                  int num_replicas, int num_sweeps, int num_reads,
                                  int exchange_interval, FloatT offset_delta, int64_t seed,
                                  FloatT* out_acceptance) {
    const cudaDeviceProp& prop = getCachedDeviceProperties();
    const int num_pairs = num_replicas - 1;

    const size_t shared_mem_size =
        validateSparseSharedMemory<FloatT>(n_spins, prop.sharedMemPerBlock, "N=");

    const size_t base_bytes =
        estimateVramSparsePT<FloatT>(n_spins, nnz, num_replicas, exchange_interval);
    const size_t stats_bytes = 2 * static_cast<size_t>(num_pairs) * sizeof(int);
    const size_t required_pool_bytes =
        addPoolAlignmentOverhead(base_bytes + stats_bytes, kSparsePTStatsAllocCount);

    // VRAM pre-check (mirrors pt_sample_sparse_gpu behavior)
    validateAvailableVram(required_pool_bytes, "sparse PT exchange stats");

    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    int* d_row_offsets = pool.allocate<int>(n_spins + 1);
    int* d_col_indices = pool.allocate<int>(nnz);
    FloatT* d_values = pool.allocate<FloatT>(nnz);
    FloatT* d_betas_dev = pool.allocate<FloatT>(num_replicas);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_replicas) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_energies = pool.allocate<FloatT>(num_replicas);
    int* d_replica_map = pool.allocate<int>(num_replicas);
    curandState* d_exchange_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_pt_schedule =
        pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * exchange_interval);
    FloatT* d_offsets = pool.allocate<FloatT>(num_replicas);
    int* d_attempt_counts = pool.allocate<int>(num_pairs);
    int* d_accept_counts = pool.allocate<int>(num_pairs);

    CudaStream stream;

    CUDA_CHECK(cudaMemcpyAsync(d_h, h, n_spins * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_row_offsets, row_offsets, (n_spins + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_col_indices, col_indices, nnz * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_values, values, nnz * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_betas_dev, betas, num_replicas * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));

    std::vector<FloatT> pt_schedule;
    buildConstantBetaSchedule(betas, num_replicas, exchange_interval, pt_schedule);
    CUDA_CHECK(
        cudaMemcpyAsync(d_pt_schedule, pt_schedule.data(),
                        static_cast<size_t>(num_replicas) * exchange_interval * sizeof(FloatT),
                        cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(d_attempt_counts, 0, num_pairs * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_accept_counts, 0, num_pairs * sizeof(int), stream));

    const int threads_for_sparse = choose_threads_for_spin_kernels(prop, n_spins);
    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));
    const int threads_for_replica_kernels = choose_threads_for_1d(prop, 256);
    const int grid_size_replicas =
        (num_replicas + threads_for_replica_kernels - 1) / threads_for_replica_kernels;

    // Warmup: skip a bounded initial burn-in before counting exchange stats.
    const int warmup_exchange_cycles =
        computeAdaptWarmupExchangeCycles(num_sweeps, exchange_interval);

    for (int read = 0; read < num_reads; read++) {
        uint64_t actual_seed = generate_pt_actual_seed(seed, read);
        uint64_t exchange_seed = generate_pt_exchange_seed(seed, read);
        uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_exchange_rng, num_replicas, exchange_seed);
        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_selection_rng, num_replicas, selection_seed);
        CUDA_CHECK_LAST_ERROR();

        initReplicaMapKernel<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_replica_map, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        initSpinsKernel<<<num_replicas, threads_for_sparse, 0, stream>>>(d_spins, n_spins,
                                                                         num_replicas, actual_seed);
        CUDA_CHECK_LAST_ERROR();

        computeInitialDeltaESparseKernel<FloatT><<<num_replicas, threads_for_sparse, 0, stream>>>(
            d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_delta_e, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        computeEnergySparseKernel<FloatT>
            <<<num_replicas, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
                d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_energies, n_spins,
                num_replicas);
        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(cudaMemsetAsync(d_offsets, 0, num_replicas * sizeof(FloatT), stream));

        int sweeps_done = 0;
        int exchange_cycles = 0;
        while (sweeps_done < num_sweeps) {
            const int sweeps_this_round = std::min(exchange_interval, num_sweeps - sweeps_done);

            if (sweeps_this_round < exchange_interval) {
                std::vector<FloatT> partial_schedule;
                buildConstantBetaSchedule(betas, num_replicas, sweeps_this_round, partial_schedule);
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pt_schedule, partial_schedule.data(),
                    static_cast<size_t>(num_replicas) * sweeps_this_round * sizeof(FloatT),
                    cudaMemcpyHostToDevice, stream));
            }

            daUpdateSparseKernel<FloatT>
                <<<num_replicas, threads_for_sparse, shared_mem_size, stream>>>(
                    d_spins, d_delta_e, d_selection_rng, d_h, d_row_offsets, d_col_indices,
                    d_values, d_pt_schedule, offset_delta, n_spins, sweeps_this_round,
                    /*per_replica_schedule=*/true, d_replica_map, actual_seed, d_energies,
                    sweeps_done, d_offsets);
            CUDA_CHECK_LAST_ERROR();

            sweeps_done += sweeps_this_round;

            if (sweeps_done < num_sweeps) {
                exchange_cycles++;
                const bool past_warmup = (exchange_cycles > warmup_exchange_cycles);
                launchPtExchangePairs<FloatT>(
                    d_replica_map, d_energies, d_betas_dev, d_exchange_rng, num_replicas,
                    threads_for_replica_kernels, stream, past_warmup ? d_attempt_counts : nullptr,
                    past_warmup ? d_accept_counts : nullptr);
            }
        }
    }

    // Copy stats to host and compute acceptance rates
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> h_attempts(num_pairs);
    std::vector<int> h_accepted(num_pairs);
    CUDA_CHECK(cudaMemcpy(h_attempts.data(), d_attempt_counts, num_pairs * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accepted.data(), d_accept_counts, num_pairs * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_pairs; i++) {
        out_acceptance[i] =
            (h_attempts[i] > 0)
                ? static_cast<FloatT>(static_cast<double>(h_accepted[i]) / h_attempts[i])
                : static_cast<FloatT>(0);
    }
}

// =============================================================================
// Dense PT public API
// =============================================================================

/**
 * @brief Run Parallel Tempering Digital Annealer sampling on GPU (dense mode).
 *
 * For each read, all replicas are initialized with independent random spin
 * configurations, then swept concurrently.  Every @p exchange_interval sweeps,
 * replica exchange is attempted between adjacent temperature pairs using the
 * Metropolis criterion.  The best solution across all replicas is stored.
 *
 * Pipeline per read (all ops on a single CUDA stream):
 *  1. RNG and replica-map initialisation.
 *  2. Spin and initial ΔE initialisation.
 *  3. Initial energy computation (for incremental tracking).
 *  4. Sweep loop: daUpdateDenseKernel → ptExchangeKernel (every exchange_interval).
 *  5. Host-side best-replica selection and result copy.
 *
 * @tparam FloatT            float or double
 * @param h                  Linear biases [n_spins].
 * @param J                  Interaction matrix [n_spins × n_spins], row-major.
 * @param betas              Per-replica inverse temperatures [num_replicas].
 * @param n_spins            Number of spin variables.
 * @param num_replicas       Number of temperature replicas.
 * @param num_sweeps         Total number of annealing sweeps.
 * @param num_reads          Number of independent PT runs to perform.
 * @param exchange_interval  Sweeps between replica exchange attempts.
 * @param offset_delta       Dynamic offset increment on rejection.
 * @param seed               Random seed (-1 for time-based entropy).
 * @param out_samples        Output spin configurations [num_reads × n_spins].
 * @param out_energies       Output energies [num_reads].
 *
 * @throws std::runtime_error if VRAM or shared memory is insufficient.
 */
template <typename FloatT>
void pt_sample_gpu(const FloatT* h, const FloatT* J, const FloatT* betas, int n_spins,
                   int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                   FloatT offset_delta, int64_t seed, int8_t* out_samples, FloatT* out_energies) {
    // ---- Device properties (cached) ----
    const cudaDeviceProp& prop = getCachedDeviceProperties();

    // ---- Memory strategy ----
    MemoryConfig cfg = selectDensePTStrategy<FloatT>(n_spins, num_replicas, exchange_interval,
                                                     prop.sharedMemPerBlock);
    const size_t required_pool_bytes =
        addPoolAlignmentOverhead(cfg.total_pool_bytes, kDensePTAllocCount);

    // ---- VRAM pre-check (Philox-aware) ----
    validateAvailableVram(required_pool_bytes, "dense PT mode (Philox)");

    // ---- Pool allocation (RAII — all freed on scope exit) ----
    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    FloatT* d_J = pool.allocate<FloatT>(static_cast<size_t>(n_spins) * n_spins);
    FloatT* d_betas = pool.allocate<FloatT>(num_replicas);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_replicas) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_energies = pool.allocate<FloatT>(num_replicas);
    int* d_replica_map = pool.allocate<int>(num_replicas);
    curandState* d_exchange_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_pt_schedule =
        pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * exchange_interval);
    FloatT* d_offsets = pool.allocate<FloatT>(num_replicas);  // persistent dynamic offset

    // ---- Create CUDA stream for pipelining (RAII) ----
    CudaStream stream;

    // ---- Async copy constant model data ----
    CUDA_CHECK(cudaMemcpyAsync(d_h, h, n_spins * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_J, J, static_cast<size_t>(n_spins) * n_spins * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_betas, betas, num_replicas * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));

    // Build & upload PT schedule once
    std::vector<FloatT> pt_schedule;
    buildConstantBetaSchedule(betas, num_replicas, exchange_interval, pt_schedule);
    CUDA_CHECK(
        cudaMemcpyAsync(d_pt_schedule, pt_schedule.data(),
                        static_cast<size_t>(num_replicas) * exchange_interval * sizeof(FloatT),
                        cudaMemcpyHostToDevice, stream));

    // ---- Pre-compute kernel params ----
    const int threads_for_replica_kernels = choose_threads_for_1d(prop, 256);
    const DenseKernelLaunchConfig dense_launch =
        makeDenseKernelLaunchConfig<FloatT>(prop, n_spins, cfg.use_dense_smr);

    // Validate shared memory against the actual warp-aligned launch configuration
    validateDenseSharedMemory(dense_launch.update_shared_mem_bytes, prop.sharedMemPerBlock);

    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));
    const int grid_size_replicas =
        (num_replicas + threads_for_replica_kernels - 1) / threads_for_replica_kernels;

    // ---- Multi-read loop ----
    for (int read = 0; read < num_reads; read++) {
        uint64_t actual_seed = generate_pt_actual_seed(seed, read);
        uint64_t exchange_seed = generate_pt_exchange_seed(seed, read);
        uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

        // Init exchange & selection RNG (small: num_replicas each)
        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_exchange_rng, num_replicas, exchange_seed);
        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_selection_rng, num_replicas, selection_seed);
        CUDA_CHECK_LAST_ERROR();

        // Init replica map
        initReplicaMapKernel<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_replica_map, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        // Init spins (Philox — no rng_states)
        initSpinsKernel<<<num_replicas, dense_launch.init_threads, 0, stream>>>(
            d_spins, n_spins, num_replicas, actual_seed);
        CUDA_CHECK_LAST_ERROR();

        // Compute initial delta_E
        computeInitialDeltaEKernel<FloatT><<<num_replicas, dense_launch.init_threads, 0, stream>>>(
            d_spins, d_h, d_J, d_delta_e, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        // Compute initial energies (one-time O(N²) — needed for incremental tracking)
        computeEnergyKernel<FloatT>
            <<<num_replicas, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
                d_spins, d_h, d_J, d_energies, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        // Zero-init persistent dynamic offsets for this read
        CUDA_CHECK(cudaMemsetAsync(d_offsets, 0, num_replicas * sizeof(FloatT), stream));

        // ---- PT sweep loop with periodic exchanges ----
        int sweeps_done = 0;
        while (sweeps_done < num_sweeps) {
            const int sweeps_this_round = std::min(exchange_interval, num_sweeps - sweeps_done);

            // Partial last interval: rebuild schedule
            if (sweeps_this_round < exchange_interval) {
                std::vector<FloatT> partial_schedule;
                buildConstantBetaSchedule(betas, num_replicas, sweeps_this_round, partial_schedule);
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pt_schedule, partial_schedule.data(),
                    static_cast<size_t>(num_replicas) * sweeps_this_round * sizeof(FloatT),
                    cudaMemcpyHostToDevice, stream));
            }

            daUpdateDenseKernel<FloatT>
                <<<num_replicas, dense_launch.update_threads, dense_launch.update_shared_mem_bytes,
                   stream>>>(d_spins, d_delta_e, d_selection_rng, d_h, d_J, d_pt_schedule,
                             offset_delta, n_spins, sweeps_this_round,
                             /*per_replica_schedule=*/true, d_replica_map, actual_seed,
                             cfg.use_dense_smr, d_energies, sweeps_done, d_offsets);
            CUDA_CHECK_LAST_ERROR();

            sweeps_done += sweeps_this_round;

            // Replica exchanges (energies already up-to-date via incremental tracking)
            if (sweeps_done < num_sweeps) {
                launchPtExchangePairs<FloatT>(d_replica_map, d_energies, d_betas, d_exchange_rng,
                                              num_replicas, threads_for_replica_kernels, stream);
            }
        }

        // ---- End-of-run: find best solution across all replicas ----
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            std::vector<FloatT> h_energies(num_replicas);
            CUDA_CHECK(cudaMemcpy(h_energies.data(), d_energies, num_replicas * sizeof(FloatT),
                                  cudaMemcpyDeviceToHost));
            int best_replica = 0;
            FloatT best_energy = h_energies[0];
            for (int r = 1; r < num_replicas; r++) {
                if (h_energies[r] < best_energy) {
                    best_energy = h_energies[r];
                    best_replica = r;
                }
            }
            CUDA_CHECK(cudaMemcpy(out_samples + static_cast<size_t>(read) * n_spins,
                                  d_spins + static_cast<size_t>(best_replica) * n_spins,
                                  n_spins * sizeof(int8_t), cudaMemcpyDeviceToHost));
            out_energies[read] = best_energy;
        }
    }

    // Pool destructor frees all device memory (RAII).
}

// =============================================================================
// Sparse PT implementation
// =============================================================================

/**
 * @brief Run Parallel Tempering Digital Annealer sampling on GPU (sparse CSR mode).
 *
 * Identical control flow to pt_sample_gpu (dense) but uses CSR input for J
 * and the sparse SMR kernel (daUpdateSparseKernel) where spins and ΔE reside
 * in shared memory.  Problem size is limited by shared memory capacity.
 *
 * @tparam FloatT            float or double
 * @param h                  Linear biases [n_spins].
 * @param row_offsets        CSR row pointers [n_spins + 1].
 * @param col_indices        CSR column indices [nnz].
 * @param values             CSR non-zero coupling values [nnz].
 * @param betas              Per-replica inverse temperatures [num_replicas].
 * @param n_spins            Number of spin variables.
 * @param nnz                Number of non-zero entries in the interaction matrix.
 * @param num_replicas       Number of temperature replicas.
 * @param num_sweeps         Total number of annealing sweeps.
 * @param num_reads          Number of independent PT runs.
 * @param exchange_interval  Sweeps between replica exchange attempts.
 * @param offset_delta       Dynamic offset increment on rejection.
 * @param seed               Random seed (-1 for time-based entropy).
 * @param out_samples        Output spin configurations [num_reads × n_spins].
 * @param out_energies       Output energies [num_reads].
 *
 * @throws std::runtime_error if VRAM is insufficient or problem exceeds
 *         shared memory capacity.
 */
template <typename FloatT>
void pt_sample_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                          const FloatT* values, const FloatT* betas, int n_spins, int nnz,
                          int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                          FloatT offset_delta, int64_t seed, int8_t* out_samples,
                          FloatT* out_energies) {
    const cudaDeviceProp& prop = getCachedDeviceProperties();
    const size_t pool_bytes =
        estimateVramSparsePT<FloatT>(n_spins, nnz, num_replicas, exchange_interval);
    const size_t required_pool_bytes = addPoolAlignmentOverhead(pool_bytes, kSparsePTAllocCount);

    // VRAM check (Philox-aware)
    validateAvailableVram(required_pool_bytes, "sparse PT mode (Philox)");

    // Shared memory for sparse SMR
    const size_t shared_mem_size =
        validateSparseSharedMemory<FloatT>(n_spins, prop.sharedMemPerBlock, "N=");

    const int threads_for_sparse = choose_threads_for_spin_kernels(prop, n_spins);
    const int threads_for_energy =
        select_power_of_two_threads(n_spins, choose_threads_for_1d(prop, 256));

    // Pool allocation
    DeviceMemoryPool pool(required_pool_bytes);

    FloatT* d_h = pool.allocate<FloatT>(n_spins);
    int* d_row_offsets = pool.allocate<int>(n_spins + 1);
    int* d_col_indices = pool.allocate<int>(nnz);
    FloatT* d_values = pool.allocate<FloatT>(nnz);
    FloatT* d_betas_dev = pool.allocate<FloatT>(num_replicas);
    int8_t* d_spins = pool.allocate<int8_t>(static_cast<size_t>(num_replicas) * n_spins);
    FloatT* d_delta_e = pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * n_spins);
    curandState* d_selection_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_energies = pool.allocate<FloatT>(num_replicas);
    int* d_replica_map = pool.allocate<int>(num_replicas);
    curandState* d_exchange_rng = pool.allocate<curandState>(num_replicas);
    FloatT* d_pt_schedule =
        pool.allocate<FloatT>(static_cast<size_t>(num_replicas) * exchange_interval);
    FloatT* d_offsets = pool.allocate<FloatT>(num_replicas);  // persistent dynamic offset

    // ---- Create CUDA stream for pipelining (RAII) ----
    CudaStream stream;

    // Async copy model
    CUDA_CHECK(cudaMemcpyAsync(d_h, h, n_spins * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_row_offsets, row_offsets, (n_spins + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_col_indices, col_indices, nnz * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_values, values, nnz * sizeof(FloatT), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_betas_dev, betas, num_replicas * sizeof(FloatT),
                               cudaMemcpyHostToDevice, stream));

    // Build PT schedule
    std::vector<FloatT> pt_schedule;
    buildConstantBetaSchedule(betas, num_replicas, exchange_interval, pt_schedule);
    CUDA_CHECK(
        cudaMemcpyAsync(d_pt_schedule, pt_schedule.data(),
                        static_cast<size_t>(num_replicas) * exchange_interval * sizeof(FloatT),
                        cudaMemcpyHostToDevice, stream));

    // Kernel params
    const int threads_for_replica_kernels = choose_threads_for_1d(prop, 256);
    const int grid_size_replicas =
        (num_replicas + threads_for_replica_kernels - 1) / threads_for_replica_kernels;

    for (int read = 0; read < num_reads; read++) {
        uint64_t actual_seed = generate_pt_actual_seed(seed, read);
        uint64_t exchange_seed = generate_pt_exchange_seed(seed, read);
        uint64_t selection_seed = actual_seed ^ kSelectionSeedSalt;

        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_exchange_rng, num_replicas, exchange_seed);
        initRngStates<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_selection_rng, num_replicas, selection_seed);
        CUDA_CHECK_LAST_ERROR();

        initReplicaMapKernel<<<grid_size_replicas, threads_for_replica_kernels, 0, stream>>>(
            d_replica_map, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        initSpinsKernel<<<num_replicas, threads_for_sparse, 0, stream>>>(d_spins, n_spins,
                                                                         num_replicas, actual_seed);
        CUDA_CHECK_LAST_ERROR();

        computeInitialDeltaESparseKernel<FloatT><<<num_replicas, threads_for_sparse, 0, stream>>>(
            d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_delta_e, n_spins, num_replicas);
        CUDA_CHECK_LAST_ERROR();

        // Compute initial energies (one-time — needed for incremental tracking)
        computeEnergySparseKernel<FloatT>
            <<<num_replicas, threads_for_energy, threads_for_energy * sizeof(FloatT), stream>>>(
                d_spins, d_h, d_row_offsets, d_col_indices, d_values, d_energies, n_spins,
                num_replicas);
        CUDA_CHECK_LAST_ERROR();

        // Zero-init persistent dynamic offsets for this read
        CUDA_CHECK(cudaMemsetAsync(d_offsets, 0, num_replicas * sizeof(FloatT), stream));

        int sweeps_done = 0;
        while (sweeps_done < num_sweeps) {
            const int sweeps_this_round = std::min(exchange_interval, num_sweeps - sweeps_done);

            if (sweeps_this_round < exchange_interval) {
                std::vector<FloatT> partial_schedule;
                buildConstantBetaSchedule(betas, num_replicas, sweeps_this_round, partial_schedule);
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pt_schedule, partial_schedule.data(),
                    static_cast<size_t>(num_replicas) * sweeps_this_round * sizeof(FloatT),
                    cudaMemcpyHostToDevice, stream));
            }

            daUpdateSparseKernel<FloatT>
                <<<num_replicas, threads_for_sparse, shared_mem_size, stream>>>(
                    d_spins, d_delta_e, d_selection_rng, d_h, d_row_offsets, d_col_indices,
                    d_values, d_pt_schedule, offset_delta, n_spins, sweeps_this_round,
                    /*per_replica_schedule=*/true, d_replica_map, actual_seed, d_energies,
                    sweeps_done, d_offsets);
            CUDA_CHECK_LAST_ERROR();

            sweeps_done += sweeps_this_round;

            // Replica exchanges (energies already up-to-date via incremental tracking)
            if (sweeps_done < num_sweeps) {
                launchPtExchangePairs<FloatT>(d_replica_map, d_energies, d_betas_dev,
                                              d_exchange_rng, num_replicas,
                                              threads_for_replica_kernels, stream);
            }
        }

        // ---- End-of-run: find best solution across all replicas ----
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            std::vector<FloatT> h_energies(num_replicas);
            CUDA_CHECK(cudaMemcpy(h_energies.data(), d_energies, num_replicas * sizeof(FloatT),
                                  cudaMemcpyDeviceToHost));
            int best_replica = 0;
            FloatT best_energy = h_energies[0];
            for (int r = 1; r < num_replicas; r++) {
                if (h_energies[r] < best_energy) {
                    best_energy = h_energies[r];
                    best_replica = r;
                }
            }
            CUDA_CHECK(cudaMemcpy(out_samples + static_cast<size_t>(read) * n_spins,
                                  d_spins + static_cast<size_t>(best_replica) * n_spins,
                                  n_spins * sizeof(int8_t), cudaMemcpyDeviceToHost));
            out_energies[read] = best_energy;
        }
    }

    // Pool and stream destructors free all device memory (RAII).
}

// =============================================================================
// Explicit template instantiations
// =============================================================================

template void pt_sample_gpu<float>(const float*, const float*, const float*, int, int, int, int,
                                   int, float, int64_t, int8_t*, float*);
template void pt_sample_gpu<double>(const double*, const double*, const double*, int, int, int, int,
                                    int, double, int64_t, int8_t*, double*);

template void pt_sample_sparse_gpu<float>(const float*, const int*, const int*, const float*,
                                          const float*, int, int, int, int, int, int, float,
                                          int64_t, int8_t*, float*);
template void pt_sample_sparse_gpu<double>(const double*, const int*, const int*, const double*,
                                           const double*, int, int, int, int, int, int, double,
                                           int64_t, int8_t*, double*);

template void pt_exchange_stats_gpu<float>(const float*, const float*, const float*, int, int, int,
                                           int, int, float, int64_t, float*);
template void pt_exchange_stats_gpu<double>(const double*, const double*, const double*, int, int,
                                            int, int, int, double, int64_t, double*);

template void pt_exchange_stats_sparse_gpu<float>(const float*, const int*, const int*,
                                                  const float*, const float*, int, int, int, int,
                                                  int, int, float, int64_t, float*);
template void pt_exchange_stats_sparse_gpu<double>(const double*, const int*, const int*,
                                                   const double*, const double*, int, int, int, int,
                                                   int, int, double, int64_t, double*);

}  // namespace digitalannealer
