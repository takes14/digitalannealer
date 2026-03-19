#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <set>

#include "engine/da_engine.h"
#include "engine/memory_strategy.h"
#include "engine/pt_engine.h"
#include "engine/seed_constants.h"
#include "utils/cuda_helpers.cuh"

namespace digitalannealer {
template <typename FloatT>
__global__ void ptExchangeKernel(int* replica_map, const FloatT* energies, const FloatT* betas,
                                 curandState* rng_state, int num_replicas, int exchange_parity,
                                 int* attempt_counts = nullptr, int* accept_counts = nullptr);
__global__ void initRngStates(curandState* states, int num_states, uint64_t seed);
__global__ void initSpinsKernel(int8_t* spins, int n_spins, int n_replicas, uint64_t seed);
template <typename FloatT>
__global__ void computeInitialDeltaEKernel(const int8_t* spins, const FloatT* h, const FloatT* J,
                                           FloatT* delta_e, int n_spins, int n_replicas);
template <typename FloatT>
__global__ void daUpdateDenseKernel(int8_t* spins, FloatT* delta_e, curandState* selection_states,
                                    const FloatT* h, const FloatT* J, const FloatT* betas,
                                    FloatT offset_delta, int n_spins, int num_sweeps,
                                    bool per_replica_schedule, const int* replica_map,
                                    uint64_t rng_seed, bool use_smr, FloatT* energies,
                                    unsigned int sweep_offset = 0, FloatT* offsets = nullptr);
}  // namespace digitalannealer

namespace {

using digitalannealer::computeInitialDeltaEKernel;
using digitalannealer::daUpdateDenseKernel;
using digitalannealer::getDenseSharedMemSize;
using digitalannealer::initRngStates;
using digitalannealer::initSpinsKernel;

// Advance a specific RNG state index (single-thread) to mirror expected consumption patterns in
// tests
__global__ void consumeUniformIdx(curandState* states, int idx, int draws) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    curandState local = states[idx];
    for (int i = 0; i < draws; ++i) {
        curand_uniform(&local);
    }
    states[idx] = local;
}

// Fetch the next uniform draw from a specific RNG state (single-thread) for test comparison
__global__ void nextUniformIdx(const curandState* states, float* outputs, int state_idx,
                               int output_idx) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    curandState local = states[state_idx];
    outputs[output_idx] = curand_uniform(&local);
}

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

TEST(DAEngineTest, SampleReturnsSpinsAndEnergy) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    float h[2] = {0.0f, 0.0f};
    float J[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float beta_schedule[1] = {1.0f};
    int8_t samples[2] = {0, 0};
    float energies[1] = {0.0f};

    ASSERT_NO_THROW(digitalannealer::da_sample_gpu<float>(
        h, J, beta_schedule,
        /*n_spins=*/2, /*num_sweeps=*/1, /*num_reads=*/1,
        /*offset_delta=*/0.0f, /*seed=*/1234, samples, energies));

    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
    EXPECT_TRUE(samples[1] == 1 || samples[1] == -1);
    EXPECT_FLOAT_EQ(0.0f, energies[0]);
}

TEST(DAEngineTest, VRAMEstimateExceedsAvailableForLargeProblem) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // 200,000 spins with dense matrix would require ~149GB for J alone (exceeds any single GPU)
    int n_spins = 200000;
    int num_reads = 1;
    int num_sweeps = 1000;

    size_t required = digitalannealer::estimateVramDenseDA<float>(n_spins, num_reads, num_sweeps);
    size_t free_mem, total_mem;
    digitalannealer::getCudaMemInfoChecked(&free_mem, &total_mem);
    EXPECT_GT(required, free_mem);
}

TEST(DAEngineTest, VRAMEstimateAccountsForScheduleSize) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // Use a problem size that is guaranteed to exceed any practical GPU VRAM.
    int n_spins = 1000000;
    int num_reads = 1;
    int num_sweeps_large = 1000000;

    size_t required =
        digitalannealer::estimateVramDenseDA<float>(n_spins, num_reads, num_sweeps_large);
    size_t free_mem, total_mem;
    digitalannealer::getCudaMemInfoChecked(&free_mem, &total_mem);
    // J alone is ~4TB, should always exceed available VRAM.
    EXPECT_GT(required, free_mem);
}

TEST(DAEngineTest, SparseVRAMEstimateExceedsAvailableForLargeProblem) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // Use a sparse configuration with massive per-replica buffers.
    int n_spins = 5000000;
    int nnz = 2000000000;  // 2B non-zeros → ~16GB for col_indices + values alone
    int num_reads = 2048;
    int num_sweeps = 1000000;

    size_t required =
        digitalannealer::estimateVramSparseDA<float>(n_spins, nnz, num_reads, num_sweeps);
    size_t free_mem, total_mem;
    digitalannealer::getCudaMemInfoChecked(&free_mem, &total_mem);
    EXPECT_GT(required, free_mem);
}

TEST(DAEngineTest, CandidateSelectionUsesIndependentRngState) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // With Philox, acceptance thresholds are stateless (derived from seed + counters),
    // so the selection RNG (curandState) is architecturally independent.
    // This test verifies deterministic behaviour: same seed → same final spins.
    constexpr int n_spins = 4;
    constexpr int n_replicas = 1;
    constexpr int num_sweeps = 10;
    constexpr uint64_t seed = 1234;
    constexpr uint64_t selection_seed = seed ^ digitalannealer::kSelectionSeedSalt;

    curandState* d_selection_rng = nullptr;
    int8_t* d_spins = nullptr;
    float *d_delta_e = nullptr, *d_h = nullptr, *d_J = nullptr, *d_betas = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_selection_rng, n_replicas * sizeof(curandState)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_spins, n_replicas * n_spins * sizeof(int8_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_delta_e, n_replicas * n_spins * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_h, n_spins * sizeof(float)));
    ASSERT_EQ(cudaSuccess,
              cudaMalloc(&d_J, static_cast<size_t>(n_spins) * n_spins * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_betas, num_sweeps * sizeof(float)));

    // Zero fields, identity-like J, uniform betas
    ASSERT_EQ(cudaSuccess, cudaMemset(d_h, 0, n_spins * sizeof(float)));
    ASSERT_EQ(cudaSuccess,
              cudaMemset(d_J, 0, static_cast<size_t>(n_spins) * n_spins * sizeof(float)));
    std::vector<float> betas_host(num_sweeps, 1.0f);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_betas, betas_host.data(), num_sweeps * sizeof(float),
                                      cudaMemcpyHostToDevice));

    // --- Run 1 ---
    initRngStates<<<1, n_replicas>>>(d_selection_rng, n_replicas, selection_seed);
    initSpinsKernel<<<n_replicas, n_spins>>>(d_spins, n_spins, n_replicas, seed);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    computeInitialDeltaEKernel<float>
        <<<n_replicas, n_spins>>>(d_spins, d_h, d_J, d_delta_e, n_spins, n_replicas);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Use warp-aligned block size (32) for dense kernel's warp-shuffle prefix sum.
    // The kernel's i += blockDim.x loop covers all spins even when blockDim > n_spins.
    const int threads_per_block = 32;
    size_t shared_mem_size =
        getDenseSharedMemSize<float>(n_spins, /*use_smr=*/false, threads_per_block);
    daUpdateDenseKernel<float><<<n_replicas, threads_per_block, shared_mem_size>>>(
        d_spins, d_delta_e, d_selection_rng, d_h, d_J, d_betas,
        /*offset_delta=*/0.0f, n_spins, num_sweeps, /*per_replica_schedule=*/false,
        /*replica_map=*/nullptr, seed, /*use_smr=*/false, /*energies=*/nullptr,
        /*sweep_offset=*/0, /*offsets=*/nullptr);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    int8_t spins_run1[n_spins] = {};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(spins_run1, d_spins, n_spins * sizeof(int8_t), cudaMemcpyDeviceToHost));

    // --- Run 2 (same seed — must produce identical result) ---
    initRngStates<<<1, n_replicas>>>(d_selection_rng, n_replicas, selection_seed);
    initSpinsKernel<<<n_replicas, n_spins>>>(d_spins, n_spins, n_replicas, seed);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    computeInitialDeltaEKernel<float>
        <<<n_replicas, n_spins>>>(d_spins, d_h, d_J, d_delta_e, n_spins, n_replicas);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    daUpdateDenseKernel<float><<<n_replicas, threads_per_block, shared_mem_size>>>(
        d_spins, d_delta_e, d_selection_rng, d_h, d_J, d_betas,
        /*offset_delta=*/0.0f, n_spins, num_sweeps, /*per_replica_schedule=*/false,
        /*replica_map=*/nullptr, seed, /*use_smr=*/false, /*energies=*/nullptr,
        /*sweep_offset=*/0, /*offsets=*/nullptr);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    int8_t spins_run2[n_spins] = {};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(spins_run2, d_spins, n_spins * sizeof(int8_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_spins; ++i) {
        EXPECT_EQ(spins_run1[i], spins_run2[i]) << "Determinism violation at spin " << i;
    }

    cudaFree(d_selection_rng);
    cudaFree(d_spins);
    cudaFree(d_delta_e);
    cudaFree(d_h);
    cudaFree(d_J);
    cudaFree(d_betas);
}

TEST(SeedConstantsTest, ResolveSeedReturnsSeedForNonNegative) {
    EXPECT_EQ(digitalannealer::resolve_seed(0), 0u);
    EXPECT_EQ(digitalannealer::resolve_seed(42), 42u);
    EXPECT_EQ(digitalannealer::resolve_seed(1234567890LL), 1234567890u);
}

TEST(CudaHelpersTest, SelectPowerOfTwoThreadsClampsToLargestPowerOfTwoWithinMax) {
    // When max_threads is a power of two, results should be exact.
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(64, 256), 64);
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(256, 256), 256);
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(300, 256), 256);
    // When max_threads is NOT a power of two, the result must never exceed it.
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(180, 192), 128);
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(1, 192), 1);
    EXPECT_EQ(digitalannealer::select_power_of_two_threads(200, 192), 128);
}

TEST(CudaHelpersTest, ChooseThreadsFor1DClampsToDeviceLimits) {
    cudaDeviceProp prop{};
    prop.maxThreadsPerBlock = 384;
    prop.maxThreadsDim[0] = 256;

    EXPECT_EQ(digitalannealer::choose_threads_for_1d(prop, 512), 256);
    EXPECT_EQ(digitalannealer::choose_threads_for_1d(prop, 64), 64);
    EXPECT_EQ(digitalannealer::choose_threads_for_1d(prop, 0), 1);
}

TEST(CudaHelpersTest, ChooseThreadsForSpinKernelsUsesSpinCountAndDeviceLimits) {
    cudaDeviceProp prop{};
    prop.maxThreadsPerBlock = 1024;
    prop.maxThreadsDim[0] = 512;

    EXPECT_EQ(digitalannealer::choose_threads_for_spin_kernels(prop, 128), 128);
    EXPECT_EQ(digitalannealer::choose_threads_for_spin_kernels(prop, 4096), 512);
    EXPECT_EQ(digitalannealer::choose_threads_for_spin_kernels(prop, 0), 1);
}

TEST(DAEngineSizeMathTest, DenseInteractionBytesUsesSizeTPromotion) {
    const uint32_t n_spins = 50000;
    const size_t overflow_prone_bytes = n_spins * n_spins * static_cast<uint32_t>(sizeof(float));
    const size_t safe_bytes =
        static_cast<size_t>(n_spins) * static_cast<size_t>(n_spins) * sizeof(float);

    EXPECT_NE(overflow_prone_bytes, safe_bytes);
    EXPECT_EQ(safe_bytes, 10000000000ULL);
}

TEST(SeedConstantsTest, ResolveSeedAvoidsDeterministicCastForNegative) {
    // Negative seed is intended to use a non-deterministic path and must not behave like
    // the deterministic casting used for non-negative seeds. In particular, resolve_seed(-1)
    // should not be equal to static_cast<uint64_t>(-1).
    const uint64_t v1 = digitalannealer::resolve_seed(-1);
    EXPECT_NE(v1, static_cast<uint64_t>(-1));
}

TEST(PTEngineTest, NegativeSeedMixesReadIntoEntropy) {
    const uint64_t entropy = 123456789ULL;
    const auto seed_read0 = digitalannealer::generate_pt_actual_seed(-1, 0, entropy);
    const auto seed_read1 = digitalannealer::generate_pt_actual_seed(-1, 1, entropy);
    EXPECT_NE(seed_read0, seed_read1);
    EXPECT_EQ(digitalannealer::generate_pt_actual_seed(5, 0, entropy), 5u);
    EXPECT_EQ(digitalannealer::generate_pt_actual_seed(5, 2, entropy), 7u);
}

TEST(PTEngineTest, ExchangeSeedAvoidsCollisionsAcrossReads) {
    const uint64_t entropy = 987654321ULL;
    const int64_t base_seed = 10;
    const auto read0_main = digitalannealer::generate_pt_actual_seed(base_seed, 0, entropy);
    const auto read1_main = digitalannealer::generate_pt_actual_seed(base_seed, 1, entropy);
    const auto read0_exchange = digitalannealer::generate_pt_exchange_seed(base_seed, 0, entropy);
    const auto read1_exchange = digitalannealer::generate_pt_exchange_seed(base_seed, 1, entropy);

    EXPECT_NE(read0_exchange, read0_main);
    EXPECT_NE(read0_exchange, read1_main);
    EXPECT_NE(read1_exchange, read1_main);
    EXPECT_NE(read1_exchange, read0_main);

    // Negative seed path should also avoid collisions and differ across reads
    const int64_t random_seed = -1;
    const auto neg_read0_main = digitalannealer::generate_pt_actual_seed(random_seed, 0, entropy);
    const auto neg_read1_main = digitalannealer::generate_pt_actual_seed(random_seed, 1, entropy);
    const auto neg_read0_exchange =
        digitalannealer::generate_pt_exchange_seed(random_seed, 0, entropy);
    const auto neg_read1_exchange =
        digitalannealer::generate_pt_exchange_seed(random_seed, 1, entropy);

    EXPECT_NE(neg_read0_exchange, neg_read0_main);
    EXPECT_NE(neg_read0_exchange, neg_read1_main);
    EXPECT_NE(neg_read1_exchange, neg_read1_main);
    EXPECT_NE(neg_read1_exchange, neg_read0_main);
    EXPECT_NE(neg_read0_exchange, neg_read1_exchange);
}

TEST(PTEngineTest, ExchangeRngUsesTemperatureIndex) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int num_replicas = 3;
    constexpr uint64_t seed = 2025;

    curandState *d_states = nullptr, *d_expected = nullptr;
    float *d_betas = nullptr, *d_energies = nullptr;
    int* d_replica_map = nullptr;
    float* d_outputs = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_states, num_replicas * sizeof(curandState)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_expected, num_replicas * sizeof(curandState)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_betas, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_energies, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_replica_map, num_replicas * sizeof(int)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_outputs, 4 * sizeof(float)));

    float betas_host[num_replicas] = {0.0f, 0.5f, 1.0f};
    // Chosen so both parity passes must evaluate probabilistic acceptance (delta > 0),
    // ensuring rng_state[0] and rng_state[1] each consume one draw.
    float energies_host[num_replicas] = {3.0f, 2.0f, 1.0f};
    int replica_map_host[num_replicas] = {0, 1, 2};

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_betas, betas_host, num_replicas * sizeof(float),
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_energies, energies_host, num_replicas * sizeof(float),
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_replica_map, replica_map_host, num_replicas * sizeof(int),
                                      cudaMemcpyHostToDevice));

    initRngStates<<<1, num_replicas>>>(d_states, num_replicas, seed);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_expected, d_states, num_replicas * sizeof(curandState),
                                      cudaMemcpyDeviceToDevice));

    digitalannealer::ptExchangeKernel<float>
        <<<1, num_replicas / 2>>>(d_replica_map, d_energies, d_betas, d_states, num_replicas, 0);
    digitalannealer::ptExchangeKernel<float><<<1, (num_replicas - 1) / 2>>>(
        d_replica_map, d_energies, d_betas, d_states, num_replicas, 1);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    consumeUniformIdx<<<1, 1>>>(d_expected, 0, 1);
    consumeUniformIdx<<<1, 1>>>(d_expected, 1, 1);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    nextUniformIdx<<<1, 1>>>(d_states, d_outputs, 0, 0);
    nextUniformIdx<<<1, 1>>>(d_states, d_outputs, 1, 1);
    nextUniformIdx<<<1, 1>>>(d_expected, d_outputs, 0, 2);
    nextUniformIdx<<<1, 1>>>(d_expected, d_outputs, 1, 3);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    float outputs[4] = {};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(outputs, d_outputs, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(outputs[0], outputs[2]);
    EXPECT_FLOAT_EQ(outputs[1], outputs[3]);

    cudaFree(d_states);
    cudaFree(d_expected);
    cudaFree(d_betas);
    cudaFree(d_energies);
    cudaFree(d_replica_map);
    cudaFree(d_outputs);
}

TEST(PTEngineTest, SampleReturnsSpinsAndEnergy) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    float h[2] = {0.0f, 0.0f};
    float J[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float betas[2] = {0.5f, 1.0f};
    int8_t samples[2] = {0, 0};
    float energies[1] = {0.0f};

    ASSERT_NO_THROW(digitalannealer::pt_sample_gpu<float>(
        h, J, betas,
        /*n_spins=*/2, /*num_replicas=*/2, /*num_sweeps=*/2,
        /*num_reads=*/1, /*exchange_interval=*/1,
        /*offset_delta=*/0.0f, /*seed=*/4321, samples, energies));

    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
    EXPECT_TRUE(samples[1] == 1 || samples[1] == -1);
    EXPECT_FLOAT_EQ(0.0f, energies[0]);
}

TEST(PTEngineTest, PTVRAMCheckThrowsForExcessiveMemory) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // Use small dummy host buffers because the function is expected to fail
    // at the VRAM check stage before accessing the actual data.
    float h_dummy[1] = {0.0f};
    float J_dummy[1] = {0.0f};
    float betas_dummy[1] = {1.0f};
    int8_t samples_dummy[1] = {0};
    float energies_dummy[1] = {0.0f};

    // With Philox RNG (no per-spin curandState), VRAM budget is dominated by J.
    // Use n_spins=200,000 so J alone is 200,000² × 4 = 160 GB, exceeding any GPU.
    EXPECT_THROW(
        {
            digitalannealer::pt_sample_gpu<float>(h_dummy, J_dummy, betas_dummy,
                                                  /*n_spins=*/200000, /*num_replicas=*/2,
                                                  /*num_sweeps=*/10, /*num_reads=*/1,
                                                  /*exchange_interval=*/5,
                                                  /*offset_delta=*/0.0f, /*seed=*/42, samples_dummy,
                                                  energies_dummy);
        },
        std::runtime_error);
}

// =============================================================================
// Double-precision (float64) tests
// =============================================================================

TEST(DAEngineDoubleTest, SampleReturnsSpinsAndEnergy) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    double h[2] = {0.0, 0.0};
    double J[4] = {0.0, 0.0, 0.0, 0.0};
    double beta_schedule[1] = {1.0};
    int8_t samples[2] = {0, 0};
    double energies[1] = {0.0};

    ASSERT_NO_THROW(digitalannealer::da_sample_gpu<double>(
        h, J, beta_schedule,
        /*n_spins=*/2, /*num_sweeps=*/1, /*num_reads=*/1,
        /*offset_delta=*/0.0, /*seed=*/1234, samples, energies));

    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
    EXPECT_TRUE(samples[1] == 1 || samples[1] == -1);
    EXPECT_DOUBLE_EQ(0.0, energies[0]);
}

TEST(PTEngineDoubleTest, SampleReturnsSpinsAndEnergy) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    double h[2] = {0.0, 0.0};
    double J[4] = {0.0, 0.0, 0.0, 0.0};
    double betas[2] = {0.5, 1.0};
    int8_t samples[2] = {0, 0};
    double energies[1] = {0.0};

    ASSERT_NO_THROW(digitalannealer::pt_sample_gpu<double>(
        h, J, betas,
        /*n_spins=*/2, /*num_replicas=*/2, /*num_sweeps=*/2,
        /*num_reads=*/1, /*exchange_interval=*/1,
        /*offset_delta=*/0.0, /*seed=*/4321, samples, energies));

    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
    EXPECT_TRUE(samples[1] == 1 || samples[1] == -1);
    EXPECT_DOUBLE_EQ(0.0, energies[0]);
}

// =============================================================================
// Returned energy consistency — DA
// =============================================================================

TEST(DAEngineTest, ReturnedEnergyMatchesCpuRecomputation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // Small problem with non-trivial h and J
    constexpr int N = 3;
    float h[N] = {0.5f, -0.3f, 0.1f};
    float J[N * N] = {0.0f, -1.0f, 0.5f, -1.0f, 0.0f, -0.7f, 0.5f, -0.7f, 0.0f};
    float beta_schedule[10];
    for (int i = 0; i < 10; ++i)
        beta_schedule[i] = 1.0f + 0.1f * i;

    int8_t samples[N];
    float energies[1];

    digitalannealer::da_sample_gpu<float>(h, J, beta_schedule, N, /*num_sweeps=*/10,
                                          /*num_reads=*/1,
                                          /*offset_delta=*/0.0f, /*seed=*/42, samples, energies);

    // CPU recomputation: E = sum_i h_i s_i + sum_{i<j} J_{ij} s_i s_j
    float cpu_energy = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_energy += h[i] * samples[i];
        for (int j = i + 1; j < N; ++j) {
            cpu_energy += J[i * N + j] * samples[i] * samples[j];
        }
    }
    EXPECT_NEAR(cpu_energy, energies[0], 1e-4f);
}

TEST(PTEngineTest, ReturnedEnergyMatchesCpuRecomputation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 3;
    float h[N] = {0.5f, -0.3f, 0.1f};
    float J[N * N] = {0.0f, -1.0f, 0.5f, -1.0f, 0.0f, -0.7f, 0.5f, -0.7f, 0.0f};
    float betas[2] = {0.5f, 1.5f};

    int8_t samples[N];
    float energies[1];

    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, /*num_replicas=*/2, /*num_sweeps=*/10,
                                          /*num_reads=*/1, /*exchange_interval=*/3,
                                          /*offset_delta=*/0.0f, /*seed=*/42, samples, energies);

    float cpu_energy = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_energy += h[i] * samples[i];
        for (int j = i + 1; j < N; ++j) {
            cpu_energy += J[i * N + j] * samples[i] * samples[j];
        }
    }
    EXPECT_NEAR(cpu_energy, energies[0], 1e-4f);
}

// =============================================================================
// API-level reproducibility — same seed produces same result
// =============================================================================

TEST(DAEngineTest, SameSeedProducesSameResult) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 4;
    float h[N] = {0.1f, -0.2f, 0.3f, -0.4f};
    float J[N * N] = {};
    J[0 * N + 1] = -1.0f;
    J[1 * N + 0] = -1.0f;
    J[2 * N + 3] = 0.5f;
    J[3 * N + 2] = 0.5f;
    float betas[5];
    for (int i = 0; i < 5; ++i)
        betas[i] = 1.0f;

    int8_t samples1[N], samples2[N];
    float energies1[1], energies2[1];

    digitalannealer::da_sample_gpu<float>(h, J, betas, N, 5, 1, 0.0f, 777, samples1, energies1);
    digitalannealer::da_sample_gpu<float>(h, J, betas, N, 5, 1, 0.0f, 777, samples2, energies2);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(samples1[i], samples2[i]) << "DA determinism violated at spin " << i;
    }
    EXPECT_FLOAT_EQ(energies1[0], energies2[0]);
}

TEST(DAEngineTest, DifferentSeedProducesDifferentResult) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 8;
    constexpr int num_reads = 4;
    float h[N];
    for (int i = 0; i < N; ++i)
        h[i] = 0.0f;
    float J[N * N] = {};
    // Fully connected anti-ferromagnetic couplings to maximise solution diversity
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            J[i * N + j] = -1.0f;
            J[j * N + i] = -1.0f;
        }
    float betas[20];
    for (int i = 0; i < 20; ++i)
        betas[i] = 0.5f + 0.1f * i;

    int8_t samples1[num_reads * N], samples2[num_reads * N];
    float energies1[num_reads], energies2[num_reads];

    digitalannealer::da_sample_gpu<float>(h, J, betas, N, 20, num_reads, 0.0f, 100, samples1,
                                          energies1);
    digitalannealer::da_sample_gpu<float>(h, J, betas, N, 20, num_reads, 0.0f, 999, samples2,
                                          energies2);

    // At least one read should differ (overwhelmingly likely with different seeds)
    bool any_differ = false;
    for (int r = 0; r < num_reads && !any_differ; ++r) {
        for (int i = 0; i < N; ++i) {
            if (samples1[r * N + i] != samples2[r * N + i]) {
                any_differ = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_differ) << "Different seeds produced identical results across all reads";
}

TEST(PTEngineTest, SameSeedProducesSameResult) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 4;
    float h[N] = {0.1f, -0.2f, 0.3f, -0.4f};
    float J[N * N] = {};
    J[0 * N + 1] = -1.0f;
    J[1 * N + 0] = -1.0f;
    J[2 * N + 3] = 0.5f;
    J[3 * N + 2] = 0.5f;
    float betas[3] = {0.5f, 1.0f, 2.0f};

    int8_t samples1[N], samples2[N];
    float energies1[1], energies2[1];

    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 3, 6, 1, 2, 0.0f, 555, samples1,
                                          energies1);
    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 3, 6, 1, 2, 0.0f, 555, samples2,
                                          energies2);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(samples1[i], samples2[i]) << "PT determinism violated at spin " << i;
    }
    EXPECT_FLOAT_EQ(energies1[0], energies2[0]);
}

TEST(PTEngineTest, DifferentSeedProducesDifferentResult) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 8;
    constexpr int num_reads = 4;
    float h[N] = {};
    float J[N * N] = {};
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            J[i * N + j] = -1.0f;
            J[j * N + i] = -1.0f;
        }
    float betas[3] = {0.5f, 1.0f, 2.0f};

    int8_t samples1[num_reads * N], samples2[num_reads * N];
    float energies1[num_reads], energies2[num_reads];

    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 3, 20, num_reads, 5, 0.0f, 100, samples1,
                                          energies1);
    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 3, 20, num_reads, 5, 0.0f, 999, samples2,
                                          energies2);

    bool any_differ = false;
    for (int r = 0; r < num_reads && !any_differ; ++r) {
        for (int i = 0; i < N; ++i) {
            if (samples1[r * N + i] != samples2[r * N + i]) {
                any_differ = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_differ) << "Different seeds produced identical PT results across all reads";
}

// =============================================================================
// PT invariant — replica_map is always a valid permutation of 0..R-1
// =============================================================================

TEST(PTEngineTest, ReplicaMapIsPermutationAfterExchange_EvenReplicas) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int num_replicas = 4;  // even
    constexpr uint64_t seed = 12345;

    curandState* d_states = nullptr;
    float *d_betas = nullptr, *d_energies = nullptr;
    int* d_replica_map = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_states, num_replicas * sizeof(curandState)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_betas, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_energies, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_replica_map, num_replicas * sizeof(int)));

    // Monotonically decreasing energies to make exchanges likely
    float betas_host[num_replicas] = {0.1f, 0.5f, 1.0f, 2.0f};
    float energies_host[num_replicas] = {4.0f, 3.0f, 2.0f, 1.0f};
    int replica_map_host[num_replicas] = {0, 1, 2, 3};

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_betas, betas_host, sizeof(betas_host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_energies, energies_host, sizeof(energies_host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_replica_map, replica_map_host, sizeof(replica_map_host),
                                      cudaMemcpyHostToDevice));

    digitalannealer::initRngStates<<<1, num_replicas>>>(d_states, num_replicas, seed);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // Multiple exchange rounds
    for (int round = 0; round < 10; ++round) {
        digitalannealer::ptExchangeKernel<float><<<1, num_replicas / 2>>>(
            d_replica_map, d_energies, d_betas, d_states, num_replicas, 0);
        digitalannealer::ptExchangeKernel<float><<<1, (num_replicas - 1) / 2>>>(
            d_replica_map, d_energies, d_betas, d_states, num_replicas, 1);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        int map[num_replicas];
        ASSERT_EQ(cudaSuccess, cudaMemcpy(map, d_replica_map, sizeof(map), cudaMemcpyDeviceToHost));

        std::set<int> seen(map, map + num_replicas);
        EXPECT_EQ(seen.size(), static_cast<size_t>(num_replicas))
            << "replica_map has duplicates at round " << round;
        for (int r = 0; r < num_replicas; ++r) {
            EXPECT_GE(map[r], 0);
            EXPECT_LT(map[r], num_replicas);
        }
    }

    cudaFree(d_states);
    cudaFree(d_betas);
    cudaFree(d_energies);
    cudaFree(d_replica_map);
}

TEST(PTEngineTest, ReplicaMapIsPermutationAfterExchange_OddReplicas) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int num_replicas = 5;  // odd
    constexpr uint64_t seed = 54321;

    curandState* d_states = nullptr;
    float *d_betas = nullptr, *d_energies = nullptr;
    int* d_replica_map = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_states, num_replicas * sizeof(curandState)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_betas, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_energies, num_replicas * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_replica_map, num_replicas * sizeof(int)));

    float betas_host[num_replicas] = {0.1f, 0.3f, 0.6f, 1.0f, 2.0f};
    float energies_host[num_replicas] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    int replica_map_host[num_replicas] = {0, 1, 2, 3, 4};

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_betas, betas_host, sizeof(betas_host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_energies, energies_host, sizeof(energies_host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_replica_map, replica_map_host, sizeof(replica_map_host),
                                      cudaMemcpyHostToDevice));

    digitalannealer::initRngStates<<<1, num_replicas>>>(d_states, num_replicas, seed);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    for (int round = 0; round < 10; ++round) {
        digitalannealer::ptExchangeKernel<float><<<1, num_replicas / 2>>>(
            d_replica_map, d_energies, d_betas, d_states, num_replicas, 0);
        digitalannealer::ptExchangeKernel<float><<<1, (num_replicas - 1) / 2>>>(
            d_replica_map, d_energies, d_betas, d_states, num_replicas, 1);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        int map[num_replicas];
        ASSERT_EQ(cudaSuccess, cudaMemcpy(map, d_replica_map, sizeof(map), cudaMemcpyDeviceToHost));

        std::set<int> seen(map, map + num_replicas);
        EXPECT_EQ(seen.size(), static_cast<size_t>(num_replicas))
            << "replica_map has duplicates at round " << round;
        for (int r = 0; r < num_replicas; ++r) {
            EXPECT_GE(map[r], 0);
            EXPECT_LT(map[r], num_replicas);
        }
    }

    cudaFree(d_states);
    cudaFree(d_betas);
    cudaFree(d_energies);
    cudaFree(d_replica_map);
}

// =============================================================================
// Partial schedule boundary — num_sweeps not multiple of exchange_interval
// =============================================================================

TEST(PTEngineTest, PartialScheduleBoundaryDoesNotCrash) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 2;
    float h[N] = {0.0f, 0.0f};
    float J[N * N] = {0.0f, -1.0f, -1.0f, 0.0f};
    float betas[2] = {0.5f, 1.5f};

    // num_sweeps=7, exchange_interval=3 -> rounds of 3, 3, 1
    int8_t samples[N];
    float energies[1];

    ASSERT_NO_THROW(digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 2, /*num_sweeps=*/7,
                                                          /*num_reads=*/1, /*exchange_interval=*/3,
                                                          /*offset_delta=*/0.0f, /*seed=*/42,
                                                          samples, energies));

    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
    EXPECT_TRUE(samples[1] == 1 || samples[1] == -1);
}

TEST(PTEngineTest, PartialScheduleBoundaryEnergyIsConsistent) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    constexpr int N = 3;
    float h[N] = {0.2f, -0.1f, 0.3f};
    float J[N * N] = {0.0f, -1.0f, 0.5f, -1.0f, 0.0f, -0.7f, 0.5f, -0.7f, 0.0f};
    float betas[3] = {0.5f, 1.0f, 2.0f};

    // num_sweeps=11, exchange_interval=4 -> rounds of 4, 4, 3
    int8_t samples[N];
    float energies[1];

    digitalannealer::pt_sample_gpu<float>(h, J, betas, N, 3, /*num_sweeps=*/11,
                                          /*num_reads=*/1, /*exchange_interval=*/4,
                                          /*offset_delta=*/0.0f, /*seed=*/123, samples, energies);

    float cpu_energy = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_energy += h[i] * samples[i];
        for (int j = i + 1; j < N; ++j) {
            cpu_energy += J[i * N + j] * samples[i] * samples[j];
        }
    }
    EXPECT_NEAR(cpu_energy, energies[0], 1e-4f);
}

// =============================================================================
// Minimal configuration — n_spins=1
// =============================================================================

TEST(DAEngineTest, MinimalConfigurationNSpins1) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    float h[1] = {0.5f};
    float J[1] = {0.0f};
    float betas[1] = {1.0f};
    int8_t samples[1] = {0};
    float energies[1] = {0.0f};

    ASSERT_NO_THROW(digitalannealer::da_sample_gpu<float>(
        h, J, betas, /*n_spins=*/1, /*num_sweeps=*/1,
        /*num_reads=*/1, /*offset_delta=*/0.0f, /*seed=*/42, samples, energies));
    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
}

TEST(PTEngineTest, MinimalConfigurationNSpins1) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    float h[1] = {0.5f};
    float J[1] = {0.0f};
    float betas[2] = {0.5f, 1.0f};
    int8_t samples[1] = {0};
    float energies[1] = {0.0f};

    ASSERT_NO_THROW(
        digitalannealer::pt_sample_gpu<float>(h, J, betas, /*n_spins=*/1, /*num_replicas=*/2,
                                              /*num_sweeps=*/1, /*num_reads=*/1,
                                              /*exchange_interval=*/1, /*offset_delta=*/0.0f,
                                              /*seed=*/42, samples, energies));
    EXPECT_TRUE(samples[0] == 1 || samples[0] == -1);
}

// =============================================================================
// VRAM estimate monotonicity
// =============================================================================

TEST(MemoryStrategyTest, DenseDAVramMonotonicInNumSweeps) {
    const size_t v1 = digitalannealer::estimateVramDenseDA<float>(100, 4, 100);
    const size_t v2 = digitalannealer::estimateVramDenseDA<float>(100, 4, 200);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, DenseDAVramMonotonicInNumReads) {
    const size_t v1 = digitalannealer::estimateVramDenseDA<float>(100, 4, 100);
    const size_t v2 = digitalannealer::estimateVramDenseDA<float>(100, 8, 100);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, SparseDAVramMonotonicInNnz) {
    const size_t v1 = digitalannealer::estimateVramSparseDA<float>(100, 500, 4, 100);
    const size_t v2 = digitalannealer::estimateVramSparseDA<float>(100, 1000, 4, 100);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, SparseDAVramMonotonicInNumSweeps) {
    const size_t v1 = digitalannealer::estimateVramSparseDA<float>(100, 500, 4, 100);
    const size_t v2 = digitalannealer::estimateVramSparseDA<float>(100, 500, 4, 200);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, SparseDAVramMonotonicInNumReads) {
    const size_t v1 = digitalannealer::estimateVramSparseDA<float>(100, 500, 2, 100);
    const size_t v2 = digitalannealer::estimateVramSparseDA<float>(100, 500, 8, 100);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, DensePTVramMonotonicInExchangeInterval) {
    const size_t v1 = digitalannealer::estimateVramDensePT<float>(100, 4, 5);
    const size_t v2 = digitalannealer::estimateVramDensePT<float>(100, 4, 10);
    EXPECT_LE(v1, v2);
}

TEST(MemoryStrategyTest, SparsePTVramMonotonicInNnz) {
    const size_t v1 = digitalannealer::estimateVramSparsePT<float>(100, 500, 4, 5);
    const size_t v2 = digitalannealer::estimateVramSparsePT<float>(100, 1000, 4, 5);
    EXPECT_LE(v1, v2);
}

// =============================================================================
// Sparse shared memory limit exceeded
// =============================================================================

TEST(DAEngineTest, SparseSharedMemoryLimitThrowsForLargeN) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // N=100000 will certainly exceed shared memory capacity for sparse SMR.
    // Dummy arrays are intentionally undersized: the shared memory check
    // (getSparseSharedMemSize) depends only on n_spins and fires before any
    // CSR data is accessed, so the actual array contents are irrelevant.
    constexpr int N = 100000;
    constexpr int nnz = 4;
    float h_dummy[1] = {0.0f};
    int row_offsets_dummy[2] = {0, nnz};
    int col_indices_dummy[nnz] = {0, 1, 2, 3};
    float values_dummy[nnz] = {1.0f, 1.0f, 1.0f, 1.0f};
    float betas_dummy[1] = {1.0f};
    int8_t samples_dummy[1] = {0};
    float energies_dummy[1] = {0.0f};

    EXPECT_THROW(
        {
            digitalannealer::da_sample_sparse_gpu<float>(
                h_dummy, row_offsets_dummy, col_indices_dummy, values_dummy, betas_dummy, N, nnz,
                /*num_sweeps=*/1, /*num_reads=*/1,
                /*offset_delta=*/0.0f, /*seed=*/42, samples_dummy, energies_dummy);
        },
        std::runtime_error);
}

TEST(PTEngineTest, SparseSharedMemoryLimitThrowsForLargeN) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    // Same rationale as the DA test above: shared memory check fires before
    // any CSR data access, so minimal dummy arrays are sufficient.
    constexpr int N = 100000;
    constexpr int nnz = 4;
    float h_dummy[1] = {0.0f};
    int row_offsets_dummy[2] = {0, nnz};
    int col_indices_dummy[nnz] = {0, 1, 2, 3};
    float values_dummy[nnz] = {1.0f, 1.0f, 1.0f, 1.0f};
    float betas_dummy[2] = {0.5f, 1.0f};
    int8_t samples_dummy[1] = {0};
    float energies_dummy[1] = {0.0f};

    EXPECT_THROW(
        {
            digitalannealer::pt_sample_sparse_gpu<float>(
                h_dummy, row_offsets_dummy, col_indices_dummy, values_dummy, betas_dummy, N, nnz,
                /*num_replicas=*/2, /*num_sweeps=*/1,
                /*num_reads=*/1, /*exchange_interval=*/1,
                /*offset_delta=*/0.0f, /*seed=*/42, samples_dummy, energies_dummy);
        },
        std::runtime_error);
}
