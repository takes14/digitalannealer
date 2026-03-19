# Digital Annealer Benchmarks

This directory contains benchmark and tuning scripts for measuring Digital
Annealer performance.

## Overview

| Script | Description |
|--------|-------------|
| `bench_dense.py` | Dense (fully-connected) DA timing/energy benchmark |
| `bench_sparse.py` | Sparse DA timing/energy benchmark (`ring`/`lattice`/`random`) |
| `bench_compare_openjij.py` | DA/PTDA vs OpenJij comparison (`sizes` or `reads` mode) |
| `bench_night_monitor.py` | Single-run nightly monitor that appends JSONL metrics |
| `bench_dtype_comparison.py` | float32 vs float64 benchmark for DA/PTDA (dense/sparse) |
| `tune_da.py` | DA sweep tuning study (CSV + PNG per problem size) |
| `tune_ptda.py` | PTDA tuning studies for sweeps/replicas/exchange interval |
| `tune_ptda_adaptation.py` | PTDA beta-adaptation tuning for `adapt_rounds` / `adapt_sweeps` |
| `PROFILING_GUIDE.md` | GPU profiling guide (Nsight Systems / Nsight Compute) |

## Shared benchmark helpers

Common benchmark plumbing lives in `benchmarks/_benchmark_utils.py`.

- shared CLI args (`add_standard_sampling_args`, `add_beta_and_seed_args`)
- shared CSV writing (`save_rows_to_csv`)
- shared project-root/path setup (`ensure_project_root_on_path`)
- shared geometric sweep grid helper (`build_geometric_sweeps`)

## Environment setup (uv)

From the repository root:

```bash
uv venv
source .venv/bin/activate

# Install development and benchmark dependencies explicitly. (OpenJij / matplotlib).
uv sync --group bench

```

All command examples below use `uv run --group bench` so they run with the
benchmark dependency group.

## Quick start

### Dense benchmark

```bash
uv run --group bench python benchmarks/bench_dense.py
uv run --group bench python benchmarks/bench_dense.py --sizes 100 500 1000 2000 --num-reads 10
uv run --group bench python benchmarks/bench_dense.py --output dense.csv

```

### Sparse benchmark

```bash
uv run --group bench python benchmarks/bench_sparse.py --topology ring
uv run --group bench python benchmarks/bench_sparse.py --topology lattice --sizes 100 400 900 1600
uv run --group bench python benchmarks/bench_sparse.py --topology random --connectivity 6 --sizes 500 1000 2000

```

For `--topology lattice`, pass perfect-square `--sizes` values (for example,
`100`, `400`, `900`) so the intended lattice dimensions are preserved.

### OpenJij comparison

```bash
# default: compare across sizes
uv run --group bench python benchmarks/bench_compare_openjij.py --problem-type dense

# sparse comparison across sizes (PTDA is skipped in sparse mode)
uv run --group bench python benchmarks/bench_compare_openjij.py --problem-type sparse --sizes 100 200 500

# reads-scaling mode (writes CSV + plot; defaults under benchmarks/outputs/)
uv run --group bench python benchmarks/bench_compare_openjij.py --mode reads --problem-size 200 \
  --reads 1 2 4 8 16 32 \
  --output benchmarks/outputs/ptda_reads_scaling.csv \
  --plot benchmarks/outputs/ptda_reads_scaling.png

```

`matplotlib` is included in the `bench` dependency group for PNG output.

### Night monitor

`bench_night_monitor.py` requires explicit workload arguments:

```bash
# DA monitor run
uv run --group bench python benchmarks/bench_night_monitor.py \
  --sampler da --problem-size 1000 --num-sweeps 2000 --num-reads 8

# PTDA monitor run
uv run --group bench python benchmarks/bench_night_monitor.py \
  --sampler ptda --problem-size 1000 --num-sweeps 2000 --num-reads 8 \
  --num-replicas 12 --exchange-interval 10

```

### Tuning studies

These scripts are configuration-driven (constants at top of file):

```bash
uv run --group bench python benchmarks/tune_da.py
uv run --group bench python benchmarks/tune_ptda.py
uv run --group bench python benchmarks/tune_ptda_adaptation.py

```

### dtype comparison

```bash
uv run --group bench python benchmarks/bench_dtype_comparison.py
uv run --group bench python benchmarks/bench_dtype_comparison.py --skip-sparse
uv run --group bench python benchmarks/bench_dtype_comparison.py --output dtype_comparison.csv

```

## CLI notes

Shared sampling options are used by `bench_dense.py`, `bench_sparse.py`, and
`bench_compare_openjij.py`:

- `--num-sweeps`
- `--num-reads`
- `--beta-min`
- `--beta-max`
- `--seed`

`bench_night_monitor.py` uses required workload args (`--sampler`,
`--problem-size`, `--num-sweeps`, `--num-reads`) plus optional PTDA-specific
flags (`--num-replicas`, `--exchange-interval`, beta-adaptation controls).

## Output locations

- Dense/sparse benchmarks: optional CSV via `--output` (parent directory must already exist)
- OpenJij compare (`sizes` mode): optional CSV via `--output`
- OpenJij compare (`reads` mode): CSV + PNG (default `benchmarks/outputs/`)
- Night monitor: JSONL append output (default `benchmarks/outputs/night_monitor.jsonl`)
- Tuning scripts (`tune_da.py`, `tune_ptda.py`): CSV + PNG under `benchmarks/outputs/tuning/`
- PTDA adaptation tuning (`tune_ptda_adaptation.py`): CSV/JSON + PNG under `benchmarks/outputs/tuning/`
- dtype comparison: optional CSV via `--output` (parent directory must already exist)

## SLURM wrappers (repo root)

The repo-root submit scripts (`bench.sh`, `tuning.sh`, `analyzation.sh`) each
generate an inline sbatch script and submit it directly via `sbatch`, using
standard SLURM defaults that can be overridden via environment variables.

## GPU profiling

For detailed profiling workflows, see [PROFILING_GUIDE.md](PROFILING_GUIDE.md).
