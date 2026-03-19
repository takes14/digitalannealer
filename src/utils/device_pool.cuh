/**
 * @file device_pool.cuh
 * @brief RAII arena allocator for contiguous GPU memory allocation.
 *
 * Implements the "Global Memory Pool (Allocated at Startup)" pattern described
 * in detailed_design_document.md §3.1.  A single cudaMalloc acquires all
 * required VRAM, and sub-allocations are handed out by bumping an offset
 * pointer with proper alignment.
 *
 * Benefits over individual cudaMalloc calls:
 * - Eliminates CUDA driver overhead of multiple alloc/free roundtrips
 * - Prevents VRAM fragmentation across 8-11 separate allocations
 * - RAII destructor guarantees cleanup even on exceptions
 * - Enables the engine to pre-check exactly one allocation against free VRAM
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_check.cuh"

namespace digitalannealer {

/// Default sub-allocation alignment (256 bytes) to satisfy CUDA coalescing
/// requirements and avoid misaligned access penalties.
constexpr size_t kPoolAlignment = 256;

/**
 * @brief RAII arena allocator backed by a single contiguous cudaMalloc.
 *
 * Usage:
 * @code
 *   DeviceMemoryPool pool(total_bytes);
 *   float* d_h = pool.allocate<float>(n_spins);
 *   float* d_J = pool.allocate<float>(n_spins * n_spins);
 *   // ... use d_h, d_J ...
 *   // pool destructor frees all memory at once
 * @endcode
 *
 * The pool must not outlive any CUDA kernels that reference its memory.
 * Typical usage: construct at the top of an engine function, launch kernels,
 * synchronize, then let the pool go out of scope (or call reset()).
 */
class DeviceMemoryPool {
public:
    /**
     * @brief Construct a pool with the given capacity.
     * @param total_bytes Total bytes to allocate on the device.
     * @throws std::runtime_error if cudaMalloc fails.
     */
    explicit DeviceMemoryPool(size_t total_bytes)
        : capacity_(total_bytes), offset_(0), base_(nullptr) {
        if (total_bytes == 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(&base_, total_bytes));
    }

    /// Non-copyable
    DeviceMemoryPool(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;

    /// Movable
    DeviceMemoryPool(DeviceMemoryPool&& other) noexcept
        : capacity_(other.capacity_), offset_(other.offset_), base_(other.base_) {
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    DeviceMemoryPool& operator=(DeviceMemoryPool&& other) noexcept {
        if (this != &other) {
            free();
            capacity_ = other.capacity_;
            offset_ = other.offset_;
            base_ = other.base_;
            other.base_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    ~DeviceMemoryPool() { free(); }

    /**
     * @brief Sub-allocate count elements of type T from the pool.
     *
     * The returned pointer is aligned to kPoolAlignment (256 bytes).
     *
     * @tparam T Element type
     * @param count Number of elements
     * @return Device pointer to the allocated region
     * @throws std::runtime_error if the pool is exhausted
     */
    template <typename T>
    T* allocate(size_t count) {
        const size_t bytes = count * sizeof(T);
        const size_t aligned_offset = align(offset_, kPoolAlignment);
        if (aligned_offset + bytes > capacity_) {
            throw std::runtime_error("DeviceMemoryPool exhausted: requested " +
                                     std::to_string(bytes) + " bytes at offset " +
                                     std::to_string(aligned_offset) + " but capacity is " +
                                     std::to_string(capacity_) + " bytes");
        }
        T* ptr = reinterpret_cast<T*>(static_cast<char*>(base_) + aligned_offset);
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    /** @brief Reset the bump pointer without freeing memory (reuse the pool). */
    void reset() { offset_ = 0; }

    /** @brief Total capacity in bytes. */
    size_t capacity() const { return capacity_; }

    /** @brief Bytes consumed so far (including alignment padding). */
    size_t used() const { return offset_; }

    /** @brief Bytes remaining. */
    size_t remaining() const { return capacity_ > offset_ ? capacity_ - offset_ : 0; }

    /** @brief Whether the pool owns device memory. */
    bool valid() const { return base_ != nullptr; }

private:
    static size_t align(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    void free() {
        if (base_) {
            cudaFree(base_);
            base_ = nullptr;
        }
        capacity_ = 0;
        offset_ = 0;
    }

    size_t capacity_;
    size_t offset_;
    void* base_;
};

}  // namespace digitalannealer
