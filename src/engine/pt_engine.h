/**
 * @file pt_engine.h
 * @brief Parallel Tempering Digital Annealer Engine - C++ interface header
 */

#pragma once

#include <cstdint>

#include "seed_constants.h"

namespace digitalannealer {

/**
 * @brief Run Parallel Tempering DA sampling on GPU
 *
 * @tparam FloatT float or double
 * @param h Linear biases [n_spins]
 * @param J Interaction matrix [n_spins x n_spins], row-major
 * @param betas Inverse temperatures for each replica [num_replicas]
 * @param n_spins Number of spin variables
 * @param num_replicas Number of temperature replicas
 * @param num_sweeps Number of annealing sweeps
 * @param num_reads Number of independent runs
 * @param exchange_interval Sweeps between replica exchange attempts
 * @param offset_delta Dynamic offset increment
 * @param seed Random seed (-1 for random)
 * @param out_samples Output samples [num_reads x n_spins]
 * @param out_energies Output energies [num_reads]
 */
template <typename FloatT>
void pt_sample_gpu(const FloatT* h, const FloatT* J, const FloatT* betas, int n_spins,
                   int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                   FloatT offset_delta, int64_t seed, int8_t* out_samples, FloatT* out_energies);

/**
 * @brief Run Parallel Tempering DA sampling on GPU with sparse (CSR) input
 *
 * @tparam FloatT float or double
 * @param h Linear biases [n_spins]
 * @param row_offsets CSR row pointers [n_spins + 1]
 * @param col_indices CSR column indices [nnz]
 * @param values CSR non-zero values [nnz]
 * @param betas Inverse temperatures for each replica [num_replicas]
 * @param n_spins Number of spin variables
 * @param nnz Number of non-zero entries in interaction matrix
 * @param num_replicas Number of temperature replicas
 * @param num_sweeps Number of annealing sweeps
 * @param num_reads Number of independent runs
 * @param exchange_interval Sweeps between replica exchange attempts
 * @param offset_delta Dynamic offset increment
 * @param seed Random seed (-1 for random)
 * @param out_samples Output samples [num_reads x n_spins]
 * @param out_energies Output energies [num_reads]
 */
template <typename FloatT>
void pt_sample_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                          const FloatT* values, const FloatT* betas, int n_spins, int nnz,
                          int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                          FloatT offset_delta, int64_t seed, int8_t* out_samples,
                          FloatT* out_energies);

/**
 * @brief Run short PT rounds on GPU (dense) and return per-pair exchange acceptance rates.
 *
 * Used during beta adaptation to collect exchange statistics from the actual GPU
 * kernels, replacing the previous CPU-based adaptation that produced unreliable
 * statistics for large problem sizes.
 *
 * @tparam FloatT           float or double
 * @param h                 Linear biases [n_spins].
 * @param J                 Interaction matrix [n_spins × n_spins].
 * @param betas             Per-replica inverse temperatures [num_replicas].
 * @param n_spins           Number of spin variables.
 * @param num_replicas      Number of replicas.
 * @param num_sweeps        Number of sweeps per read.
 * @param num_reads         Number of reads to average stats over.
 * @param exchange_interval Sweeps between exchanges.
 * @param offset_delta      Dynamic offset increment.
 * @param seed              RNG seed.
 * @param out_acceptance    Output acceptance rates [num_replicas-1].
 */
template <typename FloatT>
void pt_exchange_stats_gpu(const FloatT* h, const FloatT* J, const FloatT* betas, int n_spins,
                           int num_replicas, int num_sweeps, int num_reads, int exchange_interval,
                           FloatT offset_delta, int64_t seed, FloatT* out_acceptance);

/**
 * @brief Run short PT rounds on GPU (sparse CSR) and return per-pair exchange acceptance rates.
 */
template <typename FloatT>
void pt_exchange_stats_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                                  const FloatT* values, const FloatT* betas, int n_spins, int nnz,
                                  int num_replicas, int num_sweeps, int num_reads,
                                  int exchange_interval, FloatT offset_delta, int64_t seed,
                                  FloatT* out_acceptance);

}  // namespace digitalannealer
