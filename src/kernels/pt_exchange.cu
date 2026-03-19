/**
 * @file pt_exchange.cu
 * @brief Parallel Tempering replica exchange kernel
 *
 * Implements zero-copy replica exchange by swapping indices in replica_map.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <type_traits>

namespace digitalannealer {

/**
 * @brief Attempt replica exchanges between adjacent temperature pairs
 *
 * Uses the Metropolis criterion for exchange acceptance:
 * Accept with probability min(1, exp(-(beta_{k+1} - beta_k) * (E_k - E_{k+1})))
 *
 * Grid: 1D grid
 * Block: One thread per valid pair for the given parity:
 *   - Even parity (0): pairs (0,1), (2,3), (4,5), ...
 *   - Odd parity (1): pairs (1,2), (3,4), (5,6), ...
 */
template <typename FloatT>
__global__ void ptExchangeKernel(
    int* replica_map,        // [num_replicas] maps temp_idx -> physical_replica_idx
    const FloatT* energies,  // [num_replicas] energy of each physical replica
    const FloatT* betas,     // [num_replicas] inverse temperatures
    curandState* rng_state,  // Per-temperature RNG states for exchange decisions
    int num_replicas,
    int exchange_parity,  // 0: even pairs (0-1, 2-3, ...), 1: odd pairs (1-2, 3-4, ...)
    int* attempt_counts,  // Optional [num_replicas-1]: per-pair attempt counter (nullptr = skip)
    int* accept_counts)   // Optional [num_replicas-1]: per-pair accept counter (nullptr = skip)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine which pair this thread handles
    int k = exchange_parity + 2 * pair_idx;

    if (k + 1 >= num_replicas)
        return;

    // Record attempt (safe: each k is unique within a parity launch)
    if (attempt_counts != nullptr) {
        attempt_counts[k] += 1;
    }

    // Get physical replica indices
    int r1 = replica_map[k];
    int r2 = replica_map[k + 1];

    // Get energies and betas
    FloatT E1 = energies[r1];
    FloatT E2 = energies[r2];
    FloatT beta1 = betas[k];
    FloatT beta2 = betas[k + 1];

    // Metropolis criterion for exchange
    FloatT delta = (beta2 - beta1) * (E1 - E2);

    bool accept = (delta <= static_cast<FloatT>(0));
    if (!accept) {
        // Equivalent to r <= exp(-delta), but avoids exp underflow for large delta.
        if constexpr (std::is_same_v<FloatT, double>) {
            double r = curand_uniform_double(&rng_state[k]);  // (0, 1]
            accept = (log(r) <= -static_cast<double>(delta));
        } else {
            float r = curand_uniform(&rng_state[k]);  // (0, 1]
            accept = (__logf(r) <= -static_cast<float>(delta));
        }
    }

    if (accept) {
        // Swap indices (zero-copy exchange)
        replica_map[k] = r2;
        replica_map[k + 1] = r1;
        if (accept_counts != nullptr) {
            accept_counts[k] += 1;
        }
    }
}

// Explicit instantiations
template __global__ void ptExchangeKernel<float>(int*, const float*, const float*, curandState*,
                                                 int, int, int*, int*);
template __global__ void ptExchangeKernel<double>(int*, const double*, const double*, curandState*,
                                                  int, int, int*, int*);

/**
 * @brief Initialize replica map to identity mapping (replica i → temperature slot i).
 *
 * Called once at the beginning of each PT read before the sweep loop.
 * Resetting to identity ensures replica states start at the temperature
 * they were initialized for and prevents stale mappings from a previous read.
 *
 * Grid/Block: 1-D, must collectively cover [0, num_replicas).
 *
 * @param replica_map   Device array of length @p num_replicas to initialize.
 * @param num_replicas  Number of replicas.
 */
__global__ void initReplicaMapKernel(int* replica_map, int num_replicas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_replicas) {
        replica_map[idx] = idx;
    }
}

}  // namespace digitalannealer
