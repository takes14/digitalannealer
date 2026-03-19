# Contributing to Digital Annealer

Thanks for contributing to `digitalannealer`! This guide is focused on practical development workflow for this repository.

---

## 📌 Project Overview

`digitalannealer` is a GPU-first QUBO/Ising solver with:

- **Python API** compatible with OpenJij/dimod workflows.
- **PyBind11 bindings** connecting Python to C++/CUDA.
- **C++/CUDA backend** that runs annealing and PT logic on GPU.

Key source areas:

- `digitalannealer/` – Python package (`samplers/`, `models/`, `utils/`)
- `src/bindings/` – PyBind11 entry points and validation
- `src/engine/` – DA/PT orchestration and memory strategy
- `src/kernels/` – CUDA update/exchange kernels
- `tests/python/` and `tests/cpp/` – Python and C++ tests
- `docs/source/architecture/` – BDD/DDD/RDD architecture docs

---

## ⚙️ Environment Setup (uv)

Use `uv` for development installs.

```bash
# Clone repository
git clone https://github.com/takes14/digitalannealer.git
cd digitalannealer

# Create a virtual environment and install
uv venv
source .venv/bin/activate

# `uv sync` in this repository includes the `dev` group
# (lint + test dependencies via include-group).
uv sync

# Add optional groups only when needed.
uv sync --group docs --group bench

```

Notes:

- Do not use `setup.py` in this repository.
- The editable install builds the extension through `scikit-build-core`.
- If you modify files under `src/` (PyBind11/C++/CUDA backend), rebuild the compiled extension before running tests or examples: `uv pip install .`

---

## 🏗 Build (CMake Debug)

For a debug C++/CUDA build:

```bash
uv run --group test cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
uv run --group test cmake --build build -j$(nproc)

```

Run the C++ test target:

```bash
uv run --group test ctest --test-dir build -R da_engine_tests --output-on-failure

```

---

## ✅ Testing Workflow (Targeted Runs Only)

Do **not** run the full test suite during normal iteration. Run one related test file/case at a time.

Python examples:

```bash
# Single test case
uv run --group test pytest tests/python/test_sampler.py::TestDASampler::test_sample_ising_dict_input

# Focused subset
uv run --group test pytest tests/python/test_converters.py -k energy_preservation

```

C++ example:

```bash
uv run --group test ctest --test-dir build -R da_engine_tests --output-on-failure

```

If you are fixing a bug, first create a standalone reproducer and verify it fails before implementing the fix; then add/adjust tests in the appropriate test file.

For a one-shot scripted validation (build + C++ tests + Python tests), you can run:

```bash
bash tools/test.sh

```

---

## 🧹 Lint / Format

Use the repo script:

```bash
bash tools/lint.sh

```

`tools/lint.sh` runs:

1. `uv run ruff check digitalannealer/ tests/ examples/ --fix`
2. `uv run ruff format digitalannealer/ tests/ examples/`
3. `uv run ty check digitalannealer/ tests/python/ examples/`
4. `clang-format -i` over `src/**/*.cpp`, `src/**/*.cu`, `src/**/*.cuh` (if `clang-format` is installed)

---

## 🧠 Architecture Pointers

Before changing core behavior, read:

- `docs/source/architecture/basic_design_document.md`
- `docs/source/architecture/detailed_design_document.md`
- `docs/source/architecture/requirements_definition_document.md`

Important architectural expectations:

- Keep public sampler behavior dimod-compatible and return `dimod.SampleSet`.
- Keep GPU-heavy loops in C++/CUDA flow (avoid Python per-sweep hot loops).
- Preserve DA/PT execution model and data movement assumptions (GPU-resident state, PT exchange via replica mapping).

---

## 🛠 Contribution Tips & Common Pitfalls

- **Bug fixes:** repro first, then fix, then add/update tests.
- **Scope tests tightly:** run only targeted tests related to your change.
- **Avoid unsupported workflows:** no `setup.py`, no `pre-commit` usage here.
- **Respect sparse input constraints:** CSR invariants (symmetry, no duplicates/diagonal) are strictly validated.
- **Keep changes surgical:** avoid unrelated refactors in the same PR.

Suggested flow:

1. Sync branch and create a focused change.
2. Implement and run targeted tests.
3. Run `bash tools/lint.sh`.
4. Re-run impacted targeted tests.
5. Open PR with clear rationale and validation steps.
