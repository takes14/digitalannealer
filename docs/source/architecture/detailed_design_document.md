# Detailed Design Document (DDD)

**Project Name:** `digitalannealer`  
**Document Scope:** `digitalannealer/` and `src/`

## 1. End-to-End Architecture (Current)

The project is a Python API + PyBind11 + C++/CUDA backend stack:

1. **Python samplers** (`digitalannealer/samplers/*.py`) validate input, normalize data layout, build schedules, optionally adapt PT betas, and call the C++ extension.
2. **PyBind11 bindings** (`src/bindings/main.cpp`) perform shape/CSR checks, dtype-based template dispatch (`float32`/`float64`), allocate output arrays, release the GIL, and invoke engine entry points.
3. **C++ engines** (`src/engine/*.cu`) manage VRAM pre-checks, memory-pool allocations, async copies, kernel launch sequencing, and host-side finalization.
4. **CUDA kernels** (`src/kernels/*.cu`) perform spin initialization, ΔE setup, annealing updates, PT exchanges, and energy calculations.

---

## 2. Python Layer Responsibilities

## 2.1 Input validation and normalization

- `BaseSampler._validate_and_prepare_inputs()` uses:
    - `validate_ising_input()` for structural/value checks.
    - `validate_dtype()` (supports `np.float32` and `np.float64`).
    - per-call seed resolution.
- Normalization utilities:
    - `normalize_ising_to_dense()` returns dense symmetric `h`, `J`, and `var_labels`.
    - `normalize_ising_to_sparse()` returns `h`, CSR (`row_offsets`, `col_indices`, `values`), and `var_labels`.
- CSR-specific Python validation helper exists (`validate_csr_structure`) and mirrors kernel-facing constraints.

## 2.2 DA schedule construction

- `DASampler.sample_ising()` builds a **geometric** beta schedule:
    - `beta_schedule = np.geomspace(beta_min, beta_max, num_sweeps, dtype=dtype)`.
- Dense vs sparse backend is selected by the `sparse` boolean argument (no automatic mode switching in Python).

## 2.3 PT schedule construction and optional adaptation

- `PTDASampler.sample_ising()` builds initial replica betas with:
    - `betas = np.linspace(beta_min, beta_max, num_replicas, dtype=dtype)`.
- If `adapt_betas=True` and `num_replicas > 2`, it runs adaptation rounds before the main PT run:
    - prefers GPU exchange-stat collection (`pt_exchange_stats` / `pt_exchange_stats_sparse`),
    - falls back to CPU only when C++ extension import is unavailable.
- Adaptation updates run in linear-β space, scaling each pair gap by acceptance/mean-acceptance (`p_m/c` style) and then renormalizing deltas so the endpoints stay fixed (`beta_min`/`beta_max`).

## 2.4 Backend invocation contract

- Python passes backend kwargs through `_build_backend_kwargs()`:
    - `seed=None` is converted to backend sentinel `-1`.
    - scalar values are coerced to backend dtype.
- On successful backend run, samplers return a `dimod.SampleSet` (SPIN vartype) with metadata (`dtype`; PT also includes `betas`, `beta_adaptation_used`, and optional `beta_adaptation_history`).
- If `_core` cannot be imported, CPU fallback implementations are used for development/testing.

---

## 3. Binding Layer (`src/bindings/main.cpp`)

## 3.1 Dense validation and dispatch

- Dense calls validate:
    - `h` is 1D,
    - `J` is 2D,
    - `J.shape == (n_spins, n_spins)`.
- Template dispatch uses `h.dtype()`:
    - `double` path when `h` is float64,
    - otherwise float32 path.

## 3.2 Sparse CSR validation

Before sparse GPU entry, bindings enforce:

- `row_offsets[0] == 0`,
- `row_offsets[n_spins] == nnz`, monotonic row offsets,
- column index bounds,
- no diagonal entries,
- no duplicate columns per row,
- strict symmetry: every `(i,j,v)` must have `(j,i,v)` with equal `v`.

This is a hard precondition for sparse kernel correctness.

## 3.3 Exposed backend API

Bindings export:

- `da_sample_ising`, `da_sample_ising_sparse`,
- `pt_sample_ising`, `pt_sample_ising_sparse`,
- `pt_exchange_stats`, `pt_exchange_stats_sparse`,
- `check_vram_dense_da` (dense DA estimator check only).

All heavy calls run under `py::gil_scoped_release`.

---

## 4. Engine Data Flow and Launch Model

## 4.1 DA dense engine (`da_sample_gpu`)

Per call:

1. Cache device properties.
2. Select dense strategy (`selectDenseDAStrategy`): SMR eligibility + pool size estimate.
3. Apply alignment overhead (`addPoolAlignmentOverhead`) and validate VRAM (`validateAvailableVram`).
4. Allocate a single `DeviceMemoryPool`.
5. Async-copy `h`, `J`, `beta_schedule` to device.
6. Initialize per-replica selection RNG (`initRngStates`).
7. Initialize spins (`initSpinsKernel`, Philox).
8. Compute initial `delta_e` (`computeInitialDeltaEKernel`).
9. Launch **one** `daUpdateDenseKernel` for the full DA schedule (`num_sweeps`).
10. Compute final energies (`computeEnergyKernel`).
11. Async-copy samples/energies back, then one stream sync.

## 4.2 DA sparse engine (`da_sample_sparse_gpu`)

Flow mirrors dense DA but uses CSR arrays and sparse kernels:

- validates sparse shared-memory fit (`validateSparseSharedMemory`) before launch,
- launches **one** `daUpdateSparseKernel` over full `num_sweeps`,
- computes energies with `computeEnergySparseKernel`.

## 4.3 PT engines (`pt_sample_gpu`, `pt_sample_sparse_gpu`)

PT is host-orchestrated in rounds (not a single full-run kernel):

1. Allocate model + replica state buffers (`spins`, `delta_e`, `energies`, `replica_map`, RNG states, `d_pt_schedule`, persistent `d_offsets`).
2. Build and upload constant schedule block for one interval: shape `[num_replicas * exchange_interval]`.
3. For each read:
    - initialize exchange/selection RNG states,
    - reset `replica_map` to identity,
    - initialize spins and initial `delta_e`,
    - compute initial energies once,
    - zero persistent offsets.
4. While `sweeps_done < num_sweeps`:
    - run DA update kernel for `sweeps_this_round = min(exchange_interval, remaining)`,
    - if final partial interval is shorter, rebuild/upload partial PT schedule,
    - attempt exchanges (even parity then odd parity) if more sweeps remain.
5. End of read (GPU path): sync, copy replica energies to host, pick lowest-energy replica, copy only that sample to output. CPU fallback PT returns the final coldest-replica state.

## 4.4 PT exchange statistics engines

`pt_exchange_stats_gpu` and `pt_exchange_stats_sparse_gpu` run short PT rounds and collect per-pair attempt/accept counts on device.

- Warmup exchange cycles are skipped from counting (`computeAdaptWarmupExchangeCycles`).
- Final output is acceptance rate per adjacent pair: `accepts / attempts`.

---

## 5. Kernel-Level Algorithm Details

## 5.1 Common initialization kernels

- `initRngStates`: initializes small per-replica `curandState` arrays (selection/exchange RNG).
- `initSpinsKernel`: stateless Philox-based spin initialization (`±1`).

## 5.2 Dense DA update (`daUpdateDenseKernel`)

- Grid mapping: one block per temperature index (`blockIdx.x`), threads sweep over spins (`threadIdx.x` stride).
- Supports two data paths:
    - **SMR path**: spins + `delta_e` in shared memory,
    - **non-SMR path**: spins + `delta_e` in global memory.
- Per sweep:
    1. Compute candidates where `delta_e - offset <= -log(r)/beta` (`r` from Philox).
    2. Count candidates in parallel.
    3. If candidates exist, select one uniformly:
        - thread-count prefix sum via warp-shuffle helper,
        - thread 0 picks target with `curand_uniform`,
        - thread 0 replays Philox for owning thread’s stripe to locate exact spin index.
    4. On accepted flip:
        - reset offset to 0,
        - update running energy when tracking is enabled,
        - flip spin and apply differential `delta_e` update.
    5. If no flip, increase offset by `offset_delta`.
- Optional PT-specific state:
    - `replica_map` maps temperature slot to physical replica,
    - `sweep_offset` preserves Philox counter continuity across relaunches,
    - `offsets` persists dynamic offsets across relaunches,
    - `energies` tracks running energies incrementally.

## 5.3 Sparse DA update (`daUpdateSparseKernel`)

- Always shared-memory-resident for spins/`delta_e` in the update loop.
- Uses CSR neighborhood traversal for differential updates.
- Candidate selection uses thread 0 linear scan over accepted spins (uniform by index among accepted set).
- Same PT extension points as dense (`replica_map`, `sweep_offset`, persistent `offsets`, incremental `energies`).

## 5.4 PT exchange kernel (`ptExchangeKernel`)

- Uses `replica_map` indirection (zero-copy exchange).
- For adjacent pair `(k, k+1)`:
    - physical replicas `r1=replica_map[k]`, `r2=replica_map[k+1]`,
    - acceptance with `delta=(beta[k+1]-beta[k])*(E[r1]-E[r2])`,
    - accept if `delta <= 0` or by an equivalent log-domain test `log(u) <= -delta` (`u∈(0,1]`), i.e. Metropolis `min(1, exp(-delta))`.
- Accepting exchange swaps only `replica_map` entries.
- Stats arrays (`attempt_counts`, `accept_counts`) are optional and used during adaptation stat collection.

## 5.5 Parallelism mapping

- **Replica/Read axis**: `blockIdx.x`.
- **Spin axis inside replica**: `threadIdx.x` with strided loops.
- **Time axis**:
    - DA: inside one kernel launch.
    - PT: partitioned into host-controlled rounds of up to `exchange_interval` sweeps.

---

## 6. Dense/Sparse Handling and Limits

- Python chooses dense vs sparse via `sparse` argument.
- Dense kernels internally auto-select SMR eligibility (`canUseDenseSMR`) per device shared-memory limit.
- Sparse kernels require shared-memory fit from `validateSparseSharedMemory`.
    - If not fit, engine throws runtime error.
    - There is **no automatic GPU dense fallback** when sparse mode exceeds shared-memory capacity.
- Sparse CSR input must satisfy strict symmetry and structural invariants at binding boundary.

---

## 7. Memory, Streams, Seeds, and Error Handling

## 7.1 Device memory strategy

- `DeviceMemoryPool`: one contiguous `cudaMalloc`, sub-allocations aligned to 256 bytes.
- VRAM estimators in `memory_strategy.h`:
    - `estimateVramDenseDA`, `estimateVramSparseDA`,
    - `estimateVramDensePT`, `estimateVramSparsePT`.
- Alignment overhead is added conservatively with per-mode allocation-count constants.
- Safety check uses `kVramSafetyMargin = 0.9` against free VRAM.

## 7.2 Stream model

- Engines use one RAII `CudaStream` per call.
- H2D copies, kernel launches, and D2H copies are issued asynchronously on that stream and execute in stream order.
- A single `cudaStreamSynchronize` is used at completion boundaries.

## 7.3 RNG seed handling

- `resolve_seed(seed)`:
    - non-negative seed: deterministic,
    - negative seed: high-resolution time entropy.
- PT per-read seeds:
    - `generate_pt_actual_seed(seed, read, ...)`,
    - `generate_pt_exchange_seed(seed, read, ...)`.
- Selection RNG stream uses `actual_seed ^ kSelectionSeedSalt`.

## 7.4 CUDA error propagation

- `CUDA_CHECK(...)` wraps API calls and throws `std::runtime_error` on failure.
- `CUDA_CHECK_LAST_ERROR()` validates kernel launch/runtime status.
- `getCudaMemInfoChecked()` clears stale async errors before `cudaMemGetInfo`.

---

## 8. Current Module Map

| Layer                   | Files                                                                              | Current responsibility                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Python samplers         | `digitalannealer/samplers/da_sampler.py`, `ptda_sampler.py`, `base_sampler.py`     | User API, validation invocation, schedule construction, beta adaptation orchestration, backend call, SampleSet packaging  |
| Python model/validation | `digitalannealer/models/converters.py`, `digitalannealer/utils/*.py`               | QUBO/Ising conversion, dense/CSR normalization, dtype/input checks, CSR validation, CPU fallback energy utilities         |
| Bindings                | `src/bindings/main.cpp`                                                            | Array-shape/CSR validation, dtype dispatch, GIL release, Python ↔ C++ bridge                                              |
| Engines                 | `src/engine/da_engine.cu`, `pt_engine.cu`, `memory_strategy.h`, `seed_constants.h` | Memory planning, stream/memory orchestration, schedule upload policy, launch sequencing, host-side best-replica selection |
| Kernels                 | `src/kernels/da_update_dense.cu`, `da_update_sparse.cu`, `pt_exchange.cu`          | Spin init, ΔE init, annealing updates, exchange, energy kernels                                                           |
| C++ utils               | `src/utils/*.cuh`                                                                  | CUDA checks, stream RAII, device pool, cached device props, launch helper declarations, Philox helper                     |
