# Digital Annealer

**A High-Performance GPU QUBO/Ising Solver with dimod-compatible API**

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-green)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

**Digital Annealer** is a next-generation Ising solver designed to maximize throughput on NVIDIA GPUs. Unlike hybrid CPU-GPU solvers, it executes the entire annealing process—including state updates and replica exchanges—solely within VRAM. This "All-Resident" architecture eliminates the PCIe bus bottleneck, delivering massive speedups for large-scale combinatorial optimization problems.

Designed with a **dimod-compatible API**, it serves as a drop-in replacement for existing annealing workflows, allowing you to unlock GPU performance by changing just a single line of code.

---

## 📚 Table of Contents

- [Digital Annealer](#digital-annealer)
    - [📚 Table of Contents](#-table-of-contents)
    - [✨ Features](#-features)
    - [🛠 Prerequisites](#-prerequisites)
    - [📦 Installation](#-installation)
        - [User Installation](#user-installation)
        - [🔧 Development Setup](#-development-setup)
    - [🚀 Quick Start](#-quick-start)
        - [Basic Usage (Digital Annealer)](#basic-usage-digital-annealer)
        - [Advanced Usage (Parallel Tempering)](#advanced-usage-parallel-tempering)
    - [⚡ Performance](#-performance)
    - [🏗 Architecture](#-architecture)
    - [🤝 Contributing](#-contributing)
    - [📚 Academic References](#-academic-references)
    - [📄 License](#-license)

---

## ✨ Features

- **All-GPU Execution:** Performs Monte Carlo updates and Parallel Tempering (Replica Exchange) entirely on the GPU.
- **Massive Parallelism:** Solves fully connected graphs with thousands of spins ($N \ge 2000$) efficiently using CUDA cores.
- **Dynamic Offset:** Implements a dynamic energy offset mechanism to escape local minima effectively.
- **OpenJij / dimod Compatible:** Standard interface supports `qubo` and `ising` models, returning `dimod.SampleSet` objects.

---

## 🛠 Prerequisites

Before installing, ensure your environment meets the following requirements:

- **OS:** Linux
- **Python:** 3.10+
- **Hardware:** NVIDIA GPU with CUDA support
- **Build Tools:**
    - **CUDA Toolkit** (`nvcc` available in your `PATH`)
    - **CMake 3.18+**
    - **C++/CUDA compiler toolchain** with C++17 support

---

## 📦 Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable package management.

### User Installation

If you just want to use the solver:

```bash
pip install git+https://github.com/takes14/digitalannealer.git

```

Get the digitalannealer sources:

```bash
# Clone repository
git clone https://github.com/takes14/digitalannealer.git
cd digitalannealer

# Create a virtual environment and install
uv venv
source .venv/bin/activate

# Install dependencies and compile C++/CUDA kernels
uv pip install .

```

> **Note:** The installation process automatically triggers `scikit-build-core` to compile the CUDA backend. Please ensure `nvcc` is available.

### 🔧 Development Setup

For contributors who want to modify the C++ engine or run the test suite:

**1. Sync Dependency Groups**

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Sync project dependencies.
# Sync core + development dependencies (lint + test)
uv sync

# Add optional groups as needed (documentation / benchmarks)
uv sync --group docs --group bench

```

**2. C++ Debug Build (Optional)**

To build the C++ backend with debug symbols and run GoogleTest:

> **Recompile after backend changes:** If you modify files under `src/` (PyBind11/C++/CUDA backend), rebuild the compiled extension before testing or running examples:
>
> ```bash
> uv pip install .
> ```

```bash
uv run --group test cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
uv run --group test cmake --build build -j$(nproc)

# Run C++ Unit Tests
uv run --group test ctest --test-dir build --output-on-failure

```

**3. Run Python Tests**

```bash
# Prefer targeted runs while iterating
uv run --group test pytest tests/python/test_sampler.py::TestDASampler::test_sample_ising_dict_input

```

**4. Build Documentation**

```bash
bash tools/docs.sh

```

**5. Preview Built Documentation**

```bash
bash tools/docs.sh --preview

```

---

## 🚀 Quick Start

The API is designed to be a drop-in replacement for OpenJij.

### Basic Usage (Digital Annealer)

Use `DASampler` for standard annealing on a single GPU.

```python
import digitalannealer
from digitalannealer import DASampler

# 1. Define a QUBO problem
#    Minimize: x0 + x1 - x0*x1
Q = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): -1.0}

# 2. Initialize and Sample
sampler = DASampler()
response = sampler.sample_qubo(Q, num_sweeps=1000)

# 3. Access results (dimod compatible SampleSet)
print(f"Sample: {response.first.sample}")
print(f"Energy: {response.first.energy}")

```

> Precision: `sample_ising`/`sample_qubo` accept a `dtype` parameter (default: `np.float32`).
> `sample_qubo` forwards the dtype to the underlying `sample_ising` call.
> Supported dtypes are `np.float32` and `np.float64`; unsupported dtypes raise `ValueError`.

### Advanced Usage (Parallel Tempering)

Use `PTDASampler` for hard combinatorial problems with complex energy landscapes. It performs **Replica Exchange** entirely in VRAM.

```python
from digitalannealer import PTDASampler

pt_sampler = PTDASampler()

# PTDA uses beta range + replica/exchange settings (not temperatures/schedule args)
response = pt_sampler.sample_qubo(
    Q,
    num_sweeps=2000,
    num_reads=4,
    num_replicas=32,
    beta_min=0.2,
    beta_max=8.0,
    exchange_interval=20,
    adapt_betas=True,
    adapt_rounds=3,
    adapt_sweeps=1000,   # Optional; None uses auto setting
    log_adaptation=False
)

```

---

## ⚡ Performance

**`digitalannealer` is optimized for large-scale, fully connected graphs.**

- **Target Speedup:** 10x - 20x faster than standard CPU implementations (e.g., OpenJij CPU) for N \ge 2000.
- **Ideal Workloads:** Large dense graphs where O(N^2) interactions dominate computation.

_(Comparisons performed on NVIDIA A100 vs AMD EPYC 7763. Lower is better.)_

> **Note:** For small problems (N < 100), CPU solvers may be faster due to the overhead of launching CUDA kernels.

---

## 🏗 Architecture

The system adopts a modular **"Core Engine + Python Binding"** architecture.

- **Frontend (Python):** Handles input validation, QUBO-to-Ising conversion, and API compliance with `dimod`.
- **Bindings (PyBind11):** Efficiently maps NumPy arrays to C++ vectors and manages the GIL.
- **Backend (C++/CUDA):**
    - **Engine:** Orchestrates the annealing loop and memory management.
    - **Kernels:** Highly optimized CUDA kernels use PT `replica_map` index swaps (temperature-slot to physical-replica mapping) for zero-copy exchange, so spin buffers stay in place.

## 🤝 Contributing

For full contributor workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

Practical pointers:

- Run **targeted tests** while iterating (for example: `uv run --group test pytest tests/python/test_sampler.py::TestDASampler::test_sample_ising_dict_input`).
- Run lint/format checks before opening a PR: `bash tools/lint.sh`.
- Review architecture docs before changing core execution flow:
    - [`docs/source/architecture/basic_design_document.md`](docs/source/architecture/basic_design_document.md)
    - [`docs/source/architecture/detailed_design_document.md`](docs/source/architecture/detailed_design_document.md)
    - [`docs/source/architecture/requirements_definition_document.md`](docs/source/architecture/requirements_definition_document.md)

## 📚 Academic References

- Aramon, M., Rosenberg, G., Valiante, E., Miyazawa, T., Tamura, H., & Katzgraber, H. G. (2019). _Physics-inspired optimization for quadratic unconstrained problems using a digital annealer_. Frontiers in Physics, 7, 48. DOI: [10.3389/fphy.2019.00048](https://doi.org/10.3389/fphy.2019.00048)
- Tsukamoto, S., Takatsu, M., Matsubara, S., & Tamura, H. (2017). _An accelerator architecture for combinatorial optimization problems_. Fujitsu Scientific & Technical Journal, 53(5), 8–13.
- Earl, D. J., & Deem, M. W. (2005). _Parallel tempering: Theory, applications, and new perspectives_. Physical Chemistry Chemical Physics, 7(23), 3910–3916. DOI: [10.1039/B509983H](https://doi.org/10.1039/B509983H)
- Hukushima, K., & Nemoto, K. (1996). _Exchange Monte Carlo method and application to spin glass simulations_. Journal of the Physical Society of Japan, 65(6), 1604–1608. DOI: [10.1143/JPSJ.65.1604](https://doi.org/10.1143/JPSJ.65.1604)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
