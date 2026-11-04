# Repository instructions

This is the only AGENTS.md; it applies to the entire repository. Do not add nested AGENTS.md files.

## Project context

Digital Annealer is a Python 3.10+ QUBO/Ising solver with a C++17/CUDA backend and a dimod-compatible API.

- `digitalannealer/samplers/`: public `DASampler`/`PTDASampler` APIs and CPU fallback logic.
- `digitalannealer/models/` and `digitalannealer/utils/`: conversions, energy calculations, and input validation.
- `src/bindings/`, `src/engine/`, `src/kernels/`, and `src/utils/`: PyBind11 interface, GPU orchestration, CUDA kernels, and backend utilities.
- `tests/python/` and `tests/cpp/`: pytest and GoogleTest cases.

Preserve dimod `SampleSet` behavior, QUBO energy offsets, and float32/float64 support. Keep GPU annealing loops in the backend and PT exchanges based on replica mapping rather than copying spin buffers. Retain CPU fallback behavior when the CUDA extension is unavailable.

## Relevant references

Read only what the task needs:

- `README.md`: public usage and sampler parameters.
- `CONTRIBUTING.md` and `pyproject.toml`: development tools and configuration, subject to the environment limits below.
- `docs/source/architecture/basic_design_document.md`: execution model and component boundaries.
- `docs/source/architecture/detailed_design_document.md`: engine, kernel, and memory details when changing backend behavior.
- `benchmarks/README.md` and `benchmarks/PROFILING_GUIDE.md`: benchmark and profiling work.

## Changes and validation

- For a bug fix, first create a standalone reproducer and verify it fails for the expected reason. Use it to verify the fix, then add a regression test to the appropriate existing test file following its conventions.
- Run only one relevant test case at a time, never the entire suite. For example, using the existing environment: `.venv/bin/python -m pytest tests/python/test_converters.py::TestQuboToIsing::test_diagonal_only`. For GPU sampling with a working backend: `.venv/bin/python -m pytest tests/python/test_sampler.py::TestDASampler::test_sample_ising_dict_input`.
- Do not run `tools/test.sh`; it builds the backend and runs both full suites. For an already-built C++ test executable, select one GoogleTest case with `--gtest_filter=SuiteName.TestName`.
- There is no working build environment here. Do not run `setup.py`. Backend changes require rebuilding the compiled extension to validate; report this limitation if that cannot be done. CUDA tests may skip when `_core` is unavailable; a skip does not validate GPU behavior.
- Follow Ruff settings in `pyproject.toml` (Python 3.10 target, 88-column formatting) and `.clang-format` for C++/CUDA. Scope formatting to changed files; `tools/lint.sh` rewrites files across the repository.
- Do not run `pre-commit`; it is not configured.
- Do not create summary files unless explicitly asked.
- Complete the requested change and applicable focused checks before handing back; state any checks blocked by the environment.

## PR preparation

When ready to make a PR, perform exactly these steps in order:

```bash
git stash -u
git reset --hard $(cat /tmp/orig_work.txt) # Reset to the LOCAL branch; do NOT fetch.
git stash pop
```

Resolve conflicts if necessary.
