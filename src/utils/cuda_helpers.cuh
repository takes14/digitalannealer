/**
 * @file cuda_helpers.cuh
 * @brief Device-property-aware thread-count selection helpers.
 *
 * Provides small inline functions that choose block sizes suitable for
 * different kernel types (1-D generic, spin-mapped, energy-reduction).
 * All helpers query a pre-cached cudaDeviceProp so they introduce zero
 * driver-call overhead at the call site.
 */

#pragma once

#include <cuda_runtime.h>

namespace digitalannealer {

/**
 * @brief Select the smallest power-of-two thread count that covers all spins.
 *
 * Returns the smallest power-of-two value >= n_spins, capped at the largest
 * power-of-two that does not exceed max_threads.  Used to choose block sizes
 * for energy reduction kernels that require a power-of-two thread count for
 * efficient parallel reduction.
 *
 * @param n_spins Number of spins (defines the minimum required thread count)
 * @param max_threads Maximum allowed thread count (e.g., device warp limit)
 * @return Smallest power-of-two in [1, largest-power-of-two <= max_threads] that is >= n_spins
 */
inline int select_power_of_two_threads(int n_spins, int max_threads) {
    int threads = 1;
    while (threads < n_spins && threads < max_threads) {
        threads <<= 1;
    }
    // If max_threads is not itself a power of two the last shift can overshoot
    // it; step back to keep the result within the caller-supplied limit.
    if (threads > max_threads) {
        threads >>= 1;
    }
    return threads;
}

/**
 * @brief Get device-constrained maximum thread count for a 1D block launch.
 *
 * Returns the smaller of the device's ``maxThreadsPerBlock`` and a positive
 * ``maxThreadsDim[0]`` limit (if provided).  Falls back to 1 only if the
 * resulting maximum is non-positive (e.g., both limits are reported as
 * zero or otherwise invalid; this should not happen on real devices).
 *
 * @param prop CUDA device properties struct.
 * @return Maximum number of threads per block for a 1-D kernel on this device.
 */
inline int max_threads_for_1d_block(const cudaDeviceProp& prop) {
    int max_threads = prop.maxThreadsPerBlock;
    if (prop.maxThreadsDim[0] > 0 && prop.maxThreadsDim[0] < max_threads) {
        max_threads = prop.maxThreadsDim[0];
    }
    return max_threads > 0 ? max_threads : 1;
}

/**
 * @brief Choose thread count for generic 1D kernels, clamped to device limits.
 *
 * @param prop             CUDA device properties struct.
 * @param preferred_threads Desired number of threads per block.  Values <= 0
 *                          are treated as 1.
 * @return The smaller of @p preferred_threads and the device's 1-D block limit.
 */
inline int choose_threads_for_1d(const cudaDeviceProp& prop, int preferred_threads) {
    const int device_limit = max_threads_for_1d_block(prop);
    const int requested = preferred_threads > 0 ? preferred_threads : 1;
    return requested < device_limit ? requested : device_limit;
}

/**
 * @brief Choose thread count for spin-mapped kernels, capped by problem size and device limits.
 *
 * Spins are mapped one-to-one to threads (with a grid-stride loop for
 * overflow), so the natural choice is min(n_spins, device_limit).
 *
 * @param prop    CUDA device properties struct.
 * @param n_spins Number of spin variables; used as the preferred thread count.
 * @return Thread count clamped to [1, device 1-D block limit].
 */
inline int choose_threads_for_spin_kernels(const cudaDeviceProp& prop, int n_spins) {
    const int requested = n_spins > 0 ? n_spins : 1;
    return choose_threads_for_1d(prop, requested);
}

}  // namespace digitalannealer
