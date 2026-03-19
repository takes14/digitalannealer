/**
 * @file da_engine.h
 * @brief Digital Annealer Engine - C++ interface header
 */

#pragma once

#include <cstdint>

namespace digitalannealer {

/**
 * @brief Run Digital Annealer sampling on GPU
 *
 * @tparam FloatT float or double
 * @param h Linear biases [n_spins]
 * @param J Interaction matrix [n_spins x n_spins], row-major
 * @param beta_schedule Inverse temperature schedule [num_sweeps]
 * @param n_spins Number of spin variables
 * @param num_sweeps Number of annealing sweeps
 * @param num_reads Number of independent runs
 * @param offset_delta Dynamic offset increment
 * @param seed Random seed (-1 for random)
 * @param out_samples Output samples [num_reads x n_spins]
 * @param out_energies Output energies [num_reads]
 */
template <typename FloatT>
void da_sample_gpu(const FloatT* h, const FloatT* J, const FloatT* beta_schedule, int n_spins,
                   int num_sweeps, int num_reads, FloatT offset_delta, int64_t seed,
                   int8_t* out_samples, FloatT* out_energies);

/**
 * @brief Run Digital Annealer sampling on GPU with sparse (CSR) input
 *
 * @tparam FloatT float or double
 * @param h Linear biases [n_spins]
 * @param row_offsets CSR row pointers [n_spins + 1]
 * @param col_indices CSR column indices [nnz]
 * @param values CSR non-zero values [nnz]
 * @param beta_schedule Inverse temperature schedule [num_sweeps]
 * @param n_spins Number of spin variables
 * @param nnz Number of non-zero entries in interaction matrix
 * @param num_sweeps Number of annealing sweeps
 * @param num_reads Number of independent runs
 * @param offset_delta Dynamic offset increment
 * @param seed Random seed (-1 for random)
 * @param out_samples Output samples [num_reads x n_spins]
 * @param out_energies Output energies [num_reads]
 */
template <typename FloatT>
void da_sample_sparse_gpu(const FloatT* h, const int* row_offsets, const int* col_indices,
                          const FloatT* values, const FloatT* beta_schedule, int n_spins, int nnz,
                          int num_sweeps, int num_reads, FloatT offset_delta, int64_t seed,
                          int8_t* out_samples, FloatT* out_energies);

}  // namespace digitalannealer
