# Requirements Definition Document (RDD)

**Project Name:** `digitalannealer`

## 1. Project Overview

### 1.1 Background and Objective

`digitalannealer` is a Python library for Ising/QUBO combinatorial optimization with a C++/CUDA backend.
The primary execution path runs annealing and (for PT) replica exchange on GPU. When the CUDA extension is unavailable, Python samplers provide CPU fallback behavior for development/testing.

### 1.2 Core Concepts

1. **GPU-first execution:** DA and PT GPU paths keep model/state buffers in VRAM and execute sweep logic in CUDA kernels.
2. **Schedule-driven updates:** Host code prepares beta schedules and uploads them; kernels iterate over sweeps internally.
3. **Parallel-trial + Dynamic Offset:** Each sweep evaluates all spins, selects one accepted candidate uniformly, and updates offset on rejection.
4. **Replica-map exchange:** PT exchanges are zero-copy swaps of replica indices (`replica_map`), not spin-buffer copies.

---

## 2. Functional Requirements

### 2.1 Algorithmic Logic

#### 2.1.1 Digital Annealer Mode (`DASampler`)

- **Accepted inputs:** Ising `(h, J)` as dicts or NumPy arrays.
- **Schedule:** `beta_schedule = geomspace(beta_min, beta_max, num_sweeps)`.
- **Update rule (dense/sparse GPU kernels):**
    - Compute per-spin $\Delta E_i$.
    - Draw $r_i \in (0, 1]$ and compute threshold $-\log(r_i)/\beta$.
    - Spin $i$ is a candidate if $\Delta E_i - E_{off} \le -\log(r_i)/\beta$.
    - Select exactly one candidate uniformly and flip it.
- **Dynamic Offset:**
    - On successful flip: reset offset to `0`.
    - On rejection (or no valid selected candidate): increment by `offset_delta`.
- **Energy convention:** dimod-style Ising energy (`sum_i h_i s_i + sum_{i<j} J_ij s_i s_j`); diagonal terms of `J` are ignored in energy/delta-E logic.
- **Execution granularity (GPU path):** one DA update kernel launch covers the full `num_sweeps` loop (no Python per-sweep kernel launch loop).

#### 2.1.2 Parallel Tempering Mode (`PTDASampler`)

- **Replica requirements:** `num_replicas >= 2`.
- **Base schedule:** linear beta ladder from `beta_min` to `beta_max`.
- **Beta adaptation (optional):** adjusts intermediate betas in linear-β space while keeping endpoints fixed (`beta_min`/`beta_max`).
- **Exchange logic:**
    - Exchange attempts every `exchange_interval` sweeps.
    - Adjacent-pair even/odd parity exchange kernels.
    - Metropolis exchange criterion on adjacent temperatures.
    - Zero-copy swap via `replica_map`.
- **State continuity across exchange intervals (GPU path):**
    - DA update kernels are relaunched per interval in C++.
    - RNG sweep continuity uses `sweep_offset`.
    - Dynamic offsets persist across intervals (`offsets` buffer).
- **Returned sample per read:**
    - **GPU path:** choose final lowest-energy replica among all replicas.
    - **CPU fallback path:** returns the final coldest-replica state.

#### 2.1.3 Beta Adaptation (`PTDASampler`)

- **Default:** `adapt_betas=True`.
- **Activation:** adaptation runs only when `adapt_betas=True` and `num_replicas > 2`.
- **Rounds:** `adapt_rounds >= 1` when enabled.
- **Adapt sweeps:** `adapt_sweeps` defaults to `min(num_sweeps, max(1000, n_spins*5))`.
- **Stats source:**
    - GPU stats via `_core.pt_exchange_stats` / `_core.pt_exchange_stats_sparse` when the extension import succeeds.
    - CPU fallback stats only when the CUDA extension import is unavailable.
- **Metadata:** `SampleSet.info` includes `betas`, `beta_adaptation_used`, and optional `beta_adaptation_history` when `log_adaptation=True`.

### 2.2 Input and Model Support

- **Ising input validation (high-level API):**
    - Dict path: `h` keys must be integers; `J` keys must be `(int, int)` tuples; values must be numeric and finite.
    - Array path: `h` must be 1D; `J` must be square 2D; dimensions must match; values must be finite.
    - Mixed dict/array types are rejected.
- **QUBO support:**
    - `Q` accepted as dict or square NumPy array.
    - Empty or non-finite `Q` is rejected.
    - `sample_qubo` converts QUBO→Ising, calls `sample_ising`, then returns `BINARY` `SampleSet` with energy offset applied.
- **Dense normalization:**
    - Array `J` is symmetrized as `(J + J.T) / 2`.
    - Dict variables are remapped to dense indices; original labels are preserved in `SampleSet` variables.
- **Sparse normalization (CSR):**
    - Produces symmetric CSR with no diagonal entries.
    - Python utility `validate_csr_structure` validates CSR invariants.
    - Low-level `_core` bindings perform additional CSR structural validation and reject malformed CSR with clear exceptions.
- **Sparse mode behavior:**
    - `sparse=True` uses sparse GPU kernels when CUDA backend exists.
    - If sparse shared-memory requirement exceeds device capacity, sampling raises `RuntimeError` (no automatic dense GPU fallback).
    - If CUDA backend is unavailable, sparse sampling warns and falls back to dense CPU simulation.

### 2.3 User Interface (API)

- **Python requirement:** `>=3.10`.
- **Public API:** `DASampler`, `PTDASampler`, `qubo_to_ising`, `ising_to_qubo`.
- **DA parameters:** `num_sweeps`, `num_reads`, `beta_min`, `beta_max`, `offset_delta`, `seed`, `sparse`, `dtype`.
- **PT parameters:** DA parameters plus `num_replicas`, `exchange_interval`, `adapt_betas`, `adapt_rounds`, `adapt_sweeps`, `log_adaptation`.
- **`dtype` support:** `float32` and `float64` (other dtypes rejected).
- **Unknown kwargs:** accepted for forward compatibility, ignored with `UserWarning`.
- **Output format:** `dimod.SampleSet`.
    - DA info: includes `dtype`.
    - PT info: includes `dtype`, `betas`, `beta_adaptation_used`, optional adaptation history.

---

## 3. Non-Functional Requirements

- **Host/device control model:**
    - No Python-level per-sweep kernel launch loops in sampler APIs.
    - PT interval orchestration is in C++ engine code.
- **Memory safety checks:**
    - VRAM pre-checks use `estimateVram*()` plus pool-alignment overhead and a 90% free-memory safety margin.
    - Insufficient memory raises `RuntimeError` before heavy kernel execution.
- **Shared-memory constraints:**
    - Sparse kernels require SMR shared memory to fit per block; otherwise raise `RuntimeError`.
    - Dense kernel may switch internally between SMR and global-memory paths based on device limits.
- **Determinism:**
    - Fixed seeds produce deterministic runs for both DA and PT paths.
    - `seed=None` uses non-deterministic seed resolution.
- **Typing support:**
    - Package includes `py.typed` and `_core.pyi` stubs (PEP 561).

## 4. Testing Strategy

- **C++ backend tests (`tests/cpp/engine_tests.cu`):**
    - DA/PT basic execution, energy consistency, seed behavior, exchange-map invariants, VRAM estimates, partial-interval PT boundaries, sparse shared-memory limit checks.
- **Python tests (`tests/python/`):**
    - Sampler API behavior, metadata, dtype handling, parameter validation, QUBO/Ising conversions, CSR validation, dense/sparse consistency, adaptation behavior, and error propagation.
- **Execution practice:** run targeted test cases rather than full-suite runs for iterative development.
