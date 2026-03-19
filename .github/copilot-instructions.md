# Copilot Instructions for digitalannealer

## What this project is

- High-performance GPU QUBO/Ising solver; OpenJij/dimod-compatible Python API with C++/CUDA backend. All annealing (state updates + replica exchange) runs on GPU to avoid PCIe overhead (see docs/source/architecture/basic_design_document.md, docs/source/architecture/detailed_design_document.md, docs/source/architecture/requirements_definition_document.md).

## Architecture you must respect

- Core Engine + Python Binding split: Python frontend validates/normalizes input; PyBind11 bindings convert to C++ types and drop the GIL; C++ engine drives schedule; CUDA kernels do compute.
- Execution model is schedule-driven monolithic kernel: host builds beta/duration schedule once per call, uploads to VRAM, launches a single kernel for the full num_sweeps loop. Do not add Python-level per-sweep kernel launches.
- Parallel Tempering uses zero-copy swaps via replica index maps—do not move spin buffers; keep swaps inside GPU logic.
- Dense vs Sparse: dense interaction matrix for large fully-connected graphs; CSR + shared-memory resident path for sparse up to SM limits; runtime may override requested sparse mode when too large.
- Dynamic Offset: rejection increases offset; successful flip resets offset; keep this behavior in kernels/engines.

## Layout highlights

- Python API: digitalannealer/samplers (DASampler, PTDASampler), models/converters.py for QUBO→Ising, utils for helpers.
- C++/CUDA: src/bindings (PyBind11 entry, validation helpers), src/engine (DAEngine, PTEngine orchestration, memory_strategy.h VRAM estimation, seed_constants.h RNG seeds), src/kernels (da_update_dense.cu, da_update_sparse.cu, pt_exchange.cu), src/utils (cuda_check.cuh, cuda_stream.cuh, device_pool.cuh, rng.cuh, kernel_forward_decl.cuh).
- Tests: tests/python (pytest integration), tests/cpp (gtest backend). Examples in examples/, benchmarks in benchmarks/.

## Build & run

- Use uv for Python: `uv pip install -e "[dev,test,docs]"` (or `uv pip install .` for users). Do not run setup.py.
- C++ debug build: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j$(nproc)`; run `./build/tests/cpp/digitalannealer_test` for C++ unit tests.
- Python tests: prefer targeted runs, e.g., `uv pytest tests/python -k sampler` or a single case per AGENTS.md guidance.
- Lint/format via tools/lint.sh (ruff + clang-format) if available; otherwise ruff/clang-format directly.

## Coding conventions & pitfalls

- Maintain dimod-compatible API: accept dict/NumPy for h/J/Q; return dimod.SampleSet.
- Preserve GPU residency: avoid host/device thrash and Python hot loops; keep annealing/replica-exchange inside C++/CUDA flow.
- Keep VRAM pre-checks and clear MemoryError messages before launching kernels; pool allocations are contiguous.
- Respect differential update logic in dense kernels and shared-memory residency in sparse kernels; fall back gracefully when N exceeds SM capacity.
- Parallelism mapping: blockIdx.x per replica; threadIdx.x over spins; time loop inside kernel only.
- Dynamic Offset and candidate selection rules must stay intact (choose uniformly among negative ΔE candidates; offset increases on rejection).

## Workflow guardrails (from AGENTS.md)

- For bugs, first craft a standalone failing repro file; validate failure before fixes, then add/align with proper tests.
- When running tests, do not run the entire suite; run a single targeted test case.
- Do not run pre-commit; do not run setup.py; do not create summary files unless asked.
- PR prep: `git stash -u`, `git reset --hard $(cat /tmp/orig_work.txt)`, `git stash pop`, then resolve conflicts if any.

## Documentation practices

- Always consult docs/source/architecture/ (basic_design_document.md, detailed_design_document.md, requirements_definition_document.md) before making architectural decisions.
- Update docs/source/architecture/ when you introduce new patterns, change data flows, or modify core algorithms—keep the "why" up to date.
- Capture important user instructions and project-specific conventions in this copilot-instructions.md file for future reference.

## Testing strategy (t-wada TDD principles)

- Write tests first when adding features or fixing bugs; ensure tests fail before implementation.
- Test behavior, not implementation details—focus on public APIs and observable outcomes.
- Keep tests isolated, fast, and deterministic; avoid GPU-dependent tests in CI unless explicitly required.
- Use descriptive test names that document expected behavior (e.g., `test_dense_kernel_respects_dynamic_offset_on_rejection`).

## Research & implementation guidance

- Use web search when exploring CUDA optimization patterns, cuRAND best practices, or dimod API compatibility questions.
- Verify implementation approaches against authoritative sources (NVIDIA docs, academic papers on simulated annealing/replica exchange) before committing to a design.

## What to surface in reviews/changes

- Call out any new host/device crossings, per-sweep launches, or data copies that break the all-resident design.
- Note changes to schedule construction, replica mapping, or dense/sparse selection thresholds.
- Ensure examples/tests cover both dense and sparse paths and PT swaps when touched.
