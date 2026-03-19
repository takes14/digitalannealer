/**
 * @file cuda_stream.cuh
 * @brief RAII wrapper for cudaStream_t to prevent resource leaks.
 *
 * Ensures cudaStreamDestroy is always called, even when CUDA_CHECK
 * throws an exception mid-pipeline.
 */

#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

namespace digitalannealer {

/**
 * @brief RAII wrapper around cudaStream_t.
 *
 * Creates a CUDA stream on construction and destroys it on destruction,
 * guaranteeing cleanup even when exceptions are thrown.
 */
class CudaStream {
public:
    CudaStream() : stream_(nullptr) { CUDA_CHECK(cudaStreamCreate(&stream_)); }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Movable
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_)
                cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    /** @brief Implicit conversion to cudaStream_t for use in CUDA API calls. */
    operator cudaStream_t() const { return stream_; }

    /** @brief Get the underlying stream handle. */
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_;
};

}  // namespace digitalannealer
