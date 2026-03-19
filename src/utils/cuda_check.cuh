/**
 * @file cuda_check.cuh
 * @brief CUDA error handling utilities
 */

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace digitalannealer {

/**
 * @brief Check CUDA error and throw exception if not successful
 *
 * @param result CUDA API return code
 * @param file Source file name where the check occurred
 * @param line Line number where the check occurred
 *
 * @throws std::runtime_error if result indicates a CUDA error
 */
inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::string msg = "CUDA error at " + std::string(file) + ":" + std::to_string(line) +
                          " - " + cudaGetErrorString(result);
        throw std::runtime_error(msg);
    }
}

/**
 * @brief Macro to check CUDA API calls
 *
 * Usage: CUDA_CHECK(cudaMalloc(...));
 */
#define CUDA_CHECK(call) digitalannealer::checkCuda((call), __FILE__, __LINE__)

/**
 * @brief Check last CUDA error (for kernel launches)
 *
 * @param file Source file name where the check occurred
 * @param line Line number where the check occurred
 *
 * @throws std::runtime_error if there was a CUDA error
 */
inline void checkCudaLastError(const char* file, int line) {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::string msg = "CUDA kernel error at " + std::string(file) + ":" + std::to_string(line) +
                          " - " + cudaGetErrorString(result);
        throw std::runtime_error(msg);
    }
}

/**
 * @brief Macro to check the last CUDA error after kernel launch
 *
 * Usage:
 *   myKernel<<<grid, block>>>(...);
 *   CUDA_CHECK_LAST_ERROR();
 */
#define CUDA_CHECK_LAST_ERROR() digitalannealer::checkCudaLastError(__FILE__, __LINE__)

}  // namespace digitalannealer
