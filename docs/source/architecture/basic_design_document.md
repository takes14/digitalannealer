# Basic Design Document (BDD)

**Project Name:** `digitalannealer`

## 1. System Architecture

The system adopts a modular **"Python API + PyBind11 + C++/CUDA Engine"** architecture.
The Python layer provides dimod-compatible interfaces and input normalization, while C++/CUDA handles GPU execution, memory planning, and kernel orchestration.

### 1.1 Directory Structure

The repository is organized around a Python frontend and a CUDA backend:

```text
digital_annealer/
├── benchmarks/              # Performance benchmarking scripts
├── digitalannealer/         # Python package (public API)
│   ├── __init__.py
│   ├── _core.pyi            # Type stubs for PyBind11 extension API
│   ├── models/
│   │   └── converters.py    # QUBO <-> Ising conversion
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base_sampler.py  # Shared sampler utilities
│   │   ├── da_sampler.py    # DA sampler API + fallback logic
│   │   ├── ptda_sampler.py  # PT sampler API + beta adaptation
│   │   └── ptda_cpu.py      # CPU PT fallback helpers
│   └── utils/
│       ├── energy.py           # CPU energy / ΔE helpers
│       ├── validation.py       # Compatibility facade
│       ├── validation_input.py # Input + dtype validation
│       ├── normalization.py    # Dense/CSR normalization
│       └── csr_validation.py   # CSR contract checks
├── src/                     # C++/CUDA backend
│   ├── bindings/
│   │   └── main.cpp         # PyBind11 entry points and validation
│   ├── engine/
│   │   ├── da_engine.h / da_engine.cu
│   │   ├── pt_engine.h / pt_engine.cu
│   │   ├── memory_strategy.h  # VRAM estimation + dense/sparse strategy
│   │   └── seed_constants.h   # RNG seed resolution and PT seed splitting
│   ├── kernels/
│   │   ├── da_update_dense.cu
│   │   ├── da_update_sparse.cu
│   │   └── pt_exchange.cu
│   └── utils/
│       ├── cuda_stream.cuh
│       ├── device_pool.cuh
│       ├── device_props_cache.cuh
│       ├── kernel_forward_decl.cuh
│       └── rng.cuh
├── tests/
│   ├── python/              # pytest integration/behavior tests
│   └── cpp/                 # GoogleTest engine/kernel tests
└── README.md
```

---

## 2. Component Design

### 2.1 Python Layer (Frontend)

The Python layer provides dimod-compatible user APIs and delegates compute-heavy paths to the backend extension (`digitalannealer._core`).

#### 2.1.1 Sampler Modules (`digitalannealer.samplers`)

- **`BaseSampler` (base_sampler.py):**
    - Defines the shared sampler interface.
    - Handles shared validation (`_validate_and_prepare_inputs`, `_validate_sampling_params`), dtype/seed handling, and backend argument coercion.
    - Builds `dimod.SampleSet` via `_build_sample_set()` with metadata (`dtype`, PT adaptation info, etc.).
    - Provides `_da_single_sweep()` used by CPU fallback paths.

- **`DASampler` (da_sampler.py):**
    - Public API: `sample_ising()` / `sample_qubo()`.
    - Builds a geometric beta schedule and dispatches to:
        - Dense backend: `da_sample_ising`
        - Sparse backend: `da_sample_ising_sparse`
    - Supports both `float32` and `float64` dtypes.
    - Implements CPU fallback if the extension is unavailable.

- **`PTDASampler` (ptda_sampler.py):**
    - Public API: `sample_ising()` / `sample_qubo()`.
    - Builds linearly spaced replica betas in beta-space (`num_replicas`) and runs PT sampling via:
        - Dense backend: `pt_sample_ising`
        - Sparse backend: `pt_sample_ising_sparse`
    - Supports optional beta adaptation (`adapt_betas=True` by default for `num_replicas > 2`), adjusting intermediate betas in linear-β space while keeping endpoints fixed (`beta_min`/`beta_max`).
    - Uses GPU exchange-stat collection (`pt_exchange_stats`, `pt_exchange_stats_sparse`) when the extension is available; falls back to CPU stats only when extension import fails.
    - Returns PT metadata in `SampleSet.info` (`betas`, `beta_adaptation_used`, and optional adaptation history).

- **`ptda_cpu.py`:**
    - Contains PT CPU fallback helpers (`run_pt_cpu`, `attempt_exchanges_cpu`) used when CUDA bindings are unavailable.

#### 2.1.2 Model / Validation Utilities

- **`models/converters.py`:**
    - Provides `qubo_to_ising()` and `ising_to_qubo()`.
    - Supports dict and NumPy inputs with dimod-consistent conventions.

- **`utils/validation_input.py`:**
    - Validates Ising/QUBO input structure and finite values.
    - Validates supported dtypes (`float32`, `float64`).

- **`utils/normalization.py`:**
    - Normalizes Ising input to dense matrices or CSR arrays.
    - Handles variable relabeling for dict inputs.

- **`utils/csr_validation.py`:**
    - Enforces CSR invariants (index ranges, no diagonal, no duplicates, symmetry contract).

- **`utils/validation.py`:**
    - Backward-compatible facade re-exporting validation/normalization functions.

#### 2.1.3 Backend Unavailable Fallback Behavior

- If `digitalannealer._core` import fails, samplers use pure Python/NumPy CPU implementations.
- In sparse API mode under this fallback, samplers warn and run a dense CPU fallback path.
- If CUDA backend exists but sparse shared-memory limits are exceeded, the backend raises a runtime error (no implicit sparse→dense GPU fallback).

### 2.2 C++/CUDA Layer (Backend)

The backend is split into bindings, engine orchestration, and kernel compute layers.

#### 2.2.1 Bindings Layer (`src/bindings/`)

- **Purpose:** Expose backend entry points to Python via PyBind11.
- **Responsibilities:**
    - Validate dense and sparse array shapes.
    - Enforce CSR structural contract for sparse calls (including symmetry and reverse-edge value checks).
    - Dispatch float32/float64 template instantiations from NumPy dtype.
    - Release the GIL around backend execution (`py::gil_scoped_release`).
- **Exposed functions (current):**
    - Sampling: `da_sample_ising`, `da_sample_ising_sparse`, `pt_sample_ising`, `pt_sample_ising_sparse`
    - PT adaptation stats: `pt_exchange_stats`, `pt_exchange_stats_sparse`
    - Utility: `check_vram_dense_da`

#### 2.2.2 Engine Layer (`src/engine/`)

- **Purpose:** Execute DA/PT workflows independently of Python.
- **Core orchestration files:**
    - `da_engine.cu`: dense/sparse DA entry points (`da_sample_gpu`, `da_sample_sparse_gpu`)
    - `pt_engine.cu`: dense/sparse PT entry points and GPU exchange-stat collection
    - `memory_strategy.h`: VRAM estimation, dense SMR selection, sparse shared-memory validation
    - `seed_constants.h`: seed resolution and independent PT/selection/exchange seed derivation

- **Memory strategy (current behavior):**
    - Estimates required VRAM (`estimateVramDenseDA/PT`, `estimateVramSparseDA/PT`).
    - Adds pool alignment overhead for conservative checks.
    - Applies safety margin (`kVramSafetyMargin = 0.9`) before launch.
    - Dense mode decides SMR usage via `canUseDenseSMR(...)` at runtime.
    - Sparse mode requires `validateSparseSharedMemory(...)`; oversized cases fail early.

- **DA execution flow (dense/sparse):**
    1. Host-side memory strategy + VRAM validation.
    2. Single contiguous `DeviceMemoryPool` allocation.
    3. Async upload of model/schedule data.
    4. Initialize per-replica selection RNG states and initial spins.
    5. Compute initial ΔE.
    6. Launch DA update kernel; for standard DA this kernel runs the full `num_sweeps` loop.
    7. Compute final energies and async copy results to host.

- **PT execution flow (dense/sparse):**
    1. Allocate PT-specific buffers (`replica_map`, exchange RNG, per-interval schedule buffer, persistent offsets).
    2. Initialize replica map to identity and per-read RNG streams.
    3. Run sweeps in `exchange_interval` chunks:
        - Launch DA update kernel on all replicas for `sweeps_this_round`
        - Launch even/odd parity exchange kernels (`ptExchangeKernel`)
    4. Exchanges swap indices in `replica_map` (zero-copy replica swap).
    5. At read end (GPU path), copy the lowest-energy replica sample to output. In CPU fallback PT, the returned state is the final coldest replica.

#### 2.2.3 Kernel Layer (`src/kernels/`)

- **Purpose:** Raw CUDA computation.
- **`da_update_dense.cu`:**
    - Dense DA update kernel with two internal paths:
        - SMR path (when dense shared-memory footprint fits)
        - Global-memory path for larger instances
    - Uses stateless Philox RNG for per-spin acceptance thresholds.
    - Uses per-replica `curandState` for uniform candidate selection among accepted spins.
    - Maintains dynamic offset behavior (increase on rejection, reset on accepted flip).
    - Updates ΔE differentially after spin flips.

- **`da_update_sparse.cu`:**
    - Sparse DA update kernel using CSR couplings.
    - Uses shared-memory-resident spin/ΔE buffers and the same DA acceptance/candidate rules as dense mode.
    - Requires sparse shared-memory fit validated before launch.

- **`pt_exchange.cu`:**
    - Performs PT exchange attempts with Metropolis criterion.
    - Uses even/odd parity launches for adjacent pair coverage.
    - Swaps indices in `replica_map` (zero-copy exchange; spin buffers are not moved).

## 3. Development & Maintenance Workflow

To ensure high code quality and maintainability, the project enforces the following workflows:

1. **Testing Strategy:**
    - **Python tests (`tests/python`)** validate API contracts (`dimod.SampleSet`, QUBO/Ising conversion, dtype metadata), dense/sparse consistency, PT adaptation behavior, and CSR contract checks.
    - **C++ tests (`tests/cpp/engine_tests.cu`)** validate backend invariants such as VRAM estimation logic, seed behavior, exchange permutation correctness, and GPU-vs-CPU energy consistency.

2. **Linting & Formatting:**
    - **C++:** Enforced via `clang-format` (defined in `.clang-format`).
    - **Python:** Enforced via `ruff` (defined in `pyproject.toml`).
    - **Automation:** The `tools/lint.sh` script runs both.

3. **Build System:**
    - Uses `scikit-build-core` in `pyproject.toml` to seamlessly compile C++ extensions during `pip install`.
