/**
 * @file rng.cuh
 * @brief Stateless Philox-based random number generation for CUDA kernels.
 *
 * Replaces per-spin curandState (48 bytes each in global memory, read/written
 * every sweep) with a stateless counter-based approach that requires ZERO
 * persistent device memory.
 *
 * The Philox 4×32-10 generator produces 4 uint32s from a (counter, key) pair.
 * We encode:
 *   counter = {sweep, spin_index}
 *   key     = {seed_lo, replica_index}
 *
 * This guarantees independent, non-overlapping streams for every
 * (replica, spin, sweep) triple without any global memory traffic for RNG state.
 *
 * Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
 * Proc. 2011 Int. Conf. High Performance Computing, Networking, Storage and Analysis.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <type_traits>

namespace digitalannealer {

/**
 * @brief Generate a uniform value in (0, 1] using Philox counter-based RNG.
 *
 * The 4-round Philox generator is invoked inline — no persistent state needed.
 * This replaces the per-spin curandState load/store that previously dominated
 * global memory bandwidth in the inner sweep loop.
 *
 * Template parameter FloatT selects single (float) or double precision.
 * For float:  uses curand_uniform()        → (0, 1]
 * For double: uses curand_uniform_double() → (0, 1]
 *
 * @tparam FloatT  float or double
 * @param seed    64-bit base seed (from user or auto-generated)
 * @param replica Replica index (blockIdx.x physical mapping)
 * @param spin    Spin index within the replica
 * @param sweep   Current sweep number (unsigned to avoid signed overflow UB)
 * @return Uniform value in (0.0, 1.0]
 */
template <typename FloatT>
__device__ __forceinline__ FloatT philox_uniform(uint64_t seed, int replica, int spin,
                                                 unsigned int sweep) {
    // Set up Philox 4x32 state on the fly.
    // curand_Philox4x32_10 is available in cuRAND device API.
    // We use the curand device API's Philox type directly.
    curandStatePhilox4_32_10_t state;
    // subsequence encodes (replica, spin); offset encodes sweep.
    // This guarantees each (replica, spin, sweep) gets a unique position
    // in a 2^128-period stream.
    const unsigned long long subsequence =
        static_cast<unsigned long long>(replica) * 0x100000000ULL +
        static_cast<unsigned long long>(static_cast<unsigned int>(spin));
    curand_init(seed, subsequence, static_cast<unsigned long long>(sweep), &state);
    // curand_uniform / curand_uniform_double returns (0, 1] for Philox
    if constexpr (std::is_same_v<FloatT, double>) {
        return curand_uniform_double(&state);
    } else {
        return curand_uniform(&state);
    }
}

}  // namespace digitalannealer
