/**
 * @file seed_constants.h
 * @brief Shared RNG seed salts and mixing constants.
 *
 * These constants are used to generate independent RNG streams for different
 * purposes within the annealing algorithms, preventing correlation between:
 * - Per-spin acceptance RNG (rng_states)
 * - Per-replica uniform candidate selection RNG (selection_rng)
 * - PT replica exchange decision RNG (exchange_rng)
 */

#pragma once

#include <chrono>
#include <cstdint>

namespace digitalannealer {

/**
 * @brief Golden ratio constant for seed mixing and hashing.
 *
 * This is floor(2^64 / phi) where phi is the golden ratio (1.618...).
 * This constant (0x9e3779b97f4a7c15) is commonly used in hash functions
 * and RNG mixing for its good distribution properties, as it spreads
 * bits uniformly across the 64-bit space. Used for:
 * - XOR salt for selection RNG seeds (Issue #95 fix)
 * - Seed mixing for PT per-read entropy (Issue #93 fix)
 */
constexpr uint64_t kSeedMixConstant = 0x9e3779b97f4a7c15ULL;

/**
 * @brief Salt for generating candidate selection RNG seeds.
 *
 * Applied via XOR to the base seed to create independent streams for
 * per-replica uniform candidate selection, separate from per-spin acceptance RNG.
 *
 * This is an alias for kSeedMixConstant for backward compatibility with
 * existing code that uses this name.
 */
constexpr uint64_t kSelectionSeedSalt = kSeedMixConstant;

/**
 * @brief Large offset for PT exchange RNG seeds.
 *
 * Applied additively to ensure exchange decision RNG seeds don't collide
 * with either the main per-read RNG seed or seeds from other reads.
 */
constexpr uint64_t kPtExchangeSeedOffset = 1ULL << 32;

/**
 * @brief Resolve a user-provided seed to a 64-bit entropy value.
 *
 * - Non-negative seed: returned as-is (deterministic/reproducible).
 * - Negative seed: returns the current high-resolution clock nanoseconds (non-deterministic).
 */
inline uint64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return static_cast<uint64_t>(seed);
    }
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

}  // namespace digitalannealer

// ---- PT seed generation (moved from pt_engine.h for single-source-of-truth) ----

namespace digitalannealer {

/**
 * @brief Generate the per-read base RNG seed for PT.
 *
 * - For non-negative @p seed, returns @p seed + @p read (deterministic).
 * - For negative @p seed, mixes high-resolution clock entropy with the
 *   read index to avoid collisions across reads.
 */
inline uint64_t generate_pt_actual_seed(int64_t seed, int read, uint64_t entropy) {
    if (seed >= 0) {
        return static_cast<uint64_t>(seed + read);
    }
    uint64_t read_mix = static_cast<uint64_t>(read + 1);
    return entropy ^ (read_mix + kSeedMixConstant + (entropy << 6) + (entropy >> 2));
}

inline uint64_t generate_pt_actual_seed(int64_t seed, int read) {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    const auto entropy =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    return generate_pt_actual_seed(seed, read, entropy);
}

/**
 * @brief Generate the RNG seed for PT replica-exchange decisions.
 *
 * Uses the base per-read seed with a large fixed offset to avoid collisions.
 */
inline uint64_t generate_pt_exchange_seed(int64_t seed, int read, uint64_t entropy) {
    return generate_pt_actual_seed(seed, read, entropy ^ kSeedMixConstant) + kPtExchangeSeedOffset;
}

inline uint64_t generate_pt_exchange_seed(int64_t seed, int read) {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    const auto entropy =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    return generate_pt_exchange_seed(seed, read, entropy);
}

}  // namespace digitalannealer
