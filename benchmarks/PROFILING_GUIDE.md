# GPU Profiling Guide for Digital Annealer Benchmarks

This guide reflects the current benchmark/profiling scripts after the benchmark refactor.

## Current script roles (root level)

- `bench.sh`, `tuning.sh`, and `analyzation.sh` each generate an inline sbatch script and submit it directly via `sbatch`.
- `analyzation.sh` is the profiling-oriented SLURM submit script.
- `analyzation.sh` builds arguments for `benchmarks/bench_night_monitor.py` and runs that command under `nsys profile` inside a generated sbatch script.
- Benchmark Python scripts under `benchmarks/` now share argument/util helpers in `benchmarks/_benchmark_utils.py`.

## Prerequisites (uv)

From the repository root:

```bash
uv venv
source .venv/bin/activate

# Install development and benchmark dependencies explicitly. (OpenJij / matplotlib).
uv sync --group bench

```

## 1) Recommended profiling workflow (SLURM): `analyzation.sh`

### Show available options

```bash
bash analyzation.sh --help

```

### Submit a DA profiling job

```bash
bash analyzation.sh \
  --sampler da \
  --problem-size 2000 \
  --num-sweeps 20000 \
  --num-reads 100

```

### Submit a PTDA profiling job

```bash
bash analyzation.sh \
  --sampler ptda \
  --problem-size 2000 \
  --num-sweeps 20000 \
  --num-reads 100 \
  --num-replicas 12 \
  --exchange-interval 10 \
  --adapt-rounds 3 \
  --adapt-sweeps 200 \
  --log-adaptation

```

### Optional SLURM resource overrides

```bash
JOB_NAME=profile-ptda \
SLURM_PARTITION=gpu \
SLURM_TIME=02:00:00 \
SLURM_CPUS=8 \
SLURM_GPUS=1 \
bash analyzation.sh \
  --sampler ptda \
  --problem-size 2000 \
  --num-sweeps 20000 \
  --num-reads 100

```

### What `analyzation.sh` executes inside sbatch

The generated sbatch script runs this shape of command:

```bash
source .venv/bin/activate
nsys profile -t nvtx,cuda,osrt,cublas \
  --stats=true \
  -f true \
  -o "${NSYS_OUTPUT}" \
  python3 benchmarks/bench_night_monitor.py ...

```

### Where outputs go

- Nsight Systems report prefix: `nsys_reports/`
- Night monitoring JSONL (default): `benchmarks/outputs/night_monitor.jsonl`
- SLURM logs: `log_slurm/`

Monitor jobs with:

```bash
squeue -u "$USER"

```

## 2) Direct Nsight Systems commands (no SLURM wrapper)

Create output directories first:

```bash
mkdir -p nsys_reports

```

### Night monitor (DA)

```bash
nsys profile -t nvtx,cuda,osrt,cublas --stats=true -f true \
  -o nsys_reports/night_da \
  uv run --group bench python benchmarks/bench_night_monitor.py \
    --sampler da \
    --problem-size 1000 \
    --num-sweeps 2000 \
    --num-reads 8

```

### Dense benchmark script

```bash
nsys profile -t nvtx,cuda,osrt --stats=true -f true \
  -o nsys_reports/dense_n1000 \
  uv run --group bench python benchmarks/bench_dense.py \
    --sizes 1000 \
    --num-sweeps 1000 \
    --num-reads 8

```

### Sparse benchmark script

```bash
nsys profile -t nvtx,cuda,osrt --stats=true -f true \
  -o nsys_reports/sparse_lattice \
  uv run --group bench python benchmarks/bench_sparse.py \
    --sizes 1600 \
    --topology lattice \
    --num-sweeps 1000 \
    --num-reads 8

```

## 3) Nsight Compute examples

Create output directories first:

```bash
mkdir -p ncu_reports

```

### Full metric set on night monitor (DA)

```bash
ncu --set full \
  -o ncu_reports/night_da_full \
  uv run --group bench python benchmarks/bench_night_monitor.py \
    --sampler da \
    --problem-size 800 \
    --num-sweeps 200 \
    --num-reads 1

```

### Memory-focused run on sparse benchmark

```bash
ncu --set memory \
  -o ncu_reports/sparse_random_memory \
  uv run --group bench python benchmarks/bench_sparse.py \
    --sizes 1600 \
    --topology random \
    --connectivity 6 \
    --num-sweeps 200 \
    --num-reads 1

```

## 4) Current option reference

### `analyzation.sh`

| Option | Meaning | Default |
|---|---|---|
| `--sampler da\|ptda` | Sampler type | `da` |
| `--problem-size N` | Number of spins | `2000` |
| `--num-sweeps N` | Sweeps | `20000` |
| `--num-reads N` | Reads | `100` |
| `--beta-min X` | Minimum beta | `0.1` |
| `--beta-max X` | Maximum beta | `10.0` |
| `--offset-delta X` | Dynamic offset increment | `0.1` |
| `--seed N` | RNG seed | `42` |
| `--num-replicas N` | PTDA replicas | `10` |
| `--exchange-interval N` | PTDA exchange interval | `10` |
| `--disable-beta-adaptation` | Disable PTDA beta adaptation | flag off (adaptation enabled) |
| `--adapt-rounds N` | PTDA adaptation rounds | `3` |
| `--adapt-sweeps N` | PTDA adaptation sweeps | auto |
| `--log-adaptation` | Include adaptation log | off |
| `--output PATH` | JSONL output path | `benchmarks/outputs/night_monitor.jsonl` |

SLURM env overrides supported by the script:
`JOB_NAME`, `SLURM_PARTITION`, `SLURM_TIME`, `SLURM_NODES`, `SLURM_TASKS`, `SLURM_CPUS`, `SLURM_GPUS`.

### Benchmark script CLIs used above

- `benchmarks/bench_night_monitor.py`: `--sampler`, `--problem-size`, `--num-sweeps`, `--num-reads`, beta/seed, PTDA options, `--output`, `--no-warmup`
- `benchmarks/bench_dense.py`: `--sizes`, `--num-sweeps`, `--num-reads`, `--beta-min`, `--beta-max`, `--seed`, `--output`, `--no-warmup`
- `benchmarks/bench_sparse.py`: dense options plus `--topology` and `--connectivity`

## 5) Viewing profiler outputs

```bash
# Nsight Systems summary in terminal
nsys stats nsys_reports/night_da.nsys-rep

# Nsight Systems GUI
nsys-ui nsys_reports/night_da.nsys-rep

# Nsight Compute summary in terminal
ncu --import ncu_reports/night_da_full.ncu-rep --page raw

# Nsight Compute GUI
ncu-ui ncu_reports/night_da_full.ncu-rep

```

## Notes

- `bench.sh` and `tuning.sh` use the same SLURM helper infrastructure, but they are benchmark/tuning submitters (not the dedicated profiling wrapper).
- For scheduled profiling and nightly bottleneck tracking, use `analyzation.sh`.
