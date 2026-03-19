/**
 * @file device_props_cache.cuh
 * @brief Cached CUDA device properties to avoid repeated driver queries.
 *
 * `cudaGetDeviceProperties` is a synchronous driver call (~tens of µs).
 * By caching the result per device, subsequent engine invocations skip
 * the driver query entirely.
 */

#pragma once

#include <cuda_runtime.h>

#include <mutex>

#include "cuda_check.cuh"

namespace digitalannealer {

/**
 * @brief Return cached `cudaDeviceProp` for the current CUDA device.
 *
 * The first call for a given device queries the driver and caches the result.
 * Subsequent calls return the cached copy in ~nanoseconds.
 *
 * Thread-safety: protected by a mutex so concurrent first access from
 * multiple threads (possible because Python bindings release the GIL)
 * does not cause a data race.
 */
inline const cudaDeviceProp& getCachedDeviceProperties() {
    static cudaDeviceProp cached_prop;
    static int cached_device = -1;
    static std::mutex cache_mutex;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (device != cached_device) {
        CUDA_CHECK(cudaGetDeviceProperties(&cached_prop, device));
        cached_device = device;
    }
    return cached_prop;
}

}  // namespace digitalannealer
