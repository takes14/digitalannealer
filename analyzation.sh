#!/bin/bash

set -euo pipefail

# Minimal night monitoring job submitter for benchmarks/bench_night_monitor.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_NAME_OVERRIDE="${JOB_NAME:-}"
readonly TS="$(date +%Y%m%dT%H%M%S)"
readonly LOG_DIR="./log_slurm"

readonly SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
readonly SLURM_TIME="${SLURM_TIME:-04:00:00}"
readonly SLURM_NODES="${SLURM_NODES:-1}"
readonly SLURM_TASKS="${SLURM_TASKS:-1}"
readonly SLURM_CPUS="${SLURM_CPUS:-4}"
readonly SLURM_GPUS="${SLURM_GPUS:-1}"

SAMPLER="da"
PROBLEM_SIZE=2000
NUM_SWEEPS=20000
NUM_READS=100
BETA_MIN=0.1
BETA_MAX=10.0
OFFSET_DELTA=0.1
SEED=42

NUM_REPLICAS=10
EXCHANGE_INTERVAL=10
DISABLE_BETA_ADAPTATION=0
ADAPT_ROUNDS=3
ADAPT_SWEEPS=""
LOG_ADAPTATION=0

OUTPUT="benchmarks/outputs/night_monitor.jsonl"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --sampler da|ptda            Sampler type (default: da)
  --problem-size N             Problem size (default: 2000)
  --num-sweeps N               Number of sweeps (default: 20000)
  --num-reads N                Number of reads (default: 100)
  --beta-min X                 Minimum beta (default: 0.1)
  --beta-max X                 Maximum beta (default: 10.0)
  --offset-delta X             Dynamic offset delta (default: 0.1)
  --seed N                     Random seed (default: 42)

  --num-replicas N             PTDA replicas (default: 10)
  --exchange-interval N        PTDA exchange interval (default: 10)
  --disable-beta-adaptation    PTDA: disable beta adaptation
  --adapt-rounds N             PTDA adaptation rounds (default: 3)
  --adapt-sweeps N             PTDA adaptation sweeps (default: auto)
  --log-adaptation             PTDA: include adaptation log

  --output PATH                JSONL output path (default: benchmarks/outputs/night_monitor.jsonl)
  -h, --help                   Show this help

SLURM environment overrides:
  JOB_NAME, SLURM_PARTITION, SLURM_TIME, SLURM_NODES, SLURM_TASKS, SLURM_CPUS, SLURM_GPUS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler)
            SAMPLER="$2"
            shift 2
            ;;
        --problem-size)
            PROBLEM_SIZE="$2"
            shift 2
            ;;
        --num-sweeps)
            NUM_SWEEPS="$2"
            shift 2
            ;;
        --num-reads)
            NUM_READS="$2"
            shift 2
            ;;
        --beta-min)
            BETA_MIN="$2"
            shift 2
            ;;
        --beta-max)
            BETA_MAX="$2"
            shift 2
            ;;
        --offset-delta)
            OFFSET_DELTA="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-replicas)
            NUM_REPLICAS="$2"
            shift 2
            ;;
        --exchange-interval)
            EXCHANGE_INTERVAL="$2"
            shift 2
            ;;
        --disable-beta-adaptation)
            DISABLE_BETA_ADAPTATION=1
            shift
            ;;
        --adapt-rounds)
            ADAPT_ROUNDS="$2"
            shift 2
            ;;
        --adapt-sweeps)
            ADAPT_SWEEPS="$2"
            shift 2
            ;;
        --log-adaptation)
            LOG_ADAPTATION=1
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ "${SAMPLER}" != "da" && "${SAMPLER}" != "ptda" ]]; then
    echo "--sampler must be da or ptda" >&2
    exit 2
fi

if [[ -n "${JOB_NAME_OVERRIDE}" ]]; then
    readonly JOB_NAME="${JOB_NAME_OVERRIDE}"
elif [[ "${SAMPLER}" == "da" ]]; then
    readonly JOB_NAME="bench-night-da"
else
    readonly JOB_NAME="bench-night-ptda"
fi

readonly STDOUT_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stdout.log"
readonly STDERR_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stderr.log"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT}")"

CMD=(python3 benchmarks/bench_night_monitor.py
  --sampler "${SAMPLER}"
  --problem-size "${PROBLEM_SIZE}"
  --num-sweeps "${NUM_SWEEPS}"
  --num-reads "${NUM_READS}"
  --beta-min "${BETA_MIN}"
  --beta-max "${BETA_MAX}"
  --offset-delta "${OFFSET_DELTA}"
  --seed "${SEED}"
  --output "${OUTPUT}"
)

if [[ "${SAMPLER}" == "ptda" ]]; then
    CMD+=(
        --num-replicas "${NUM_REPLICAS}"
        --exchange-interval "${EXCHANGE_INTERVAL}"
        --adapt-rounds "${ADAPT_ROUNDS}"
    )

    if [[ ${DISABLE_BETA_ADAPTATION} -eq 1 ]]; then
        CMD+=(--disable-beta-adaptation)
    fi

    if [[ -n "${ADAPT_SWEEPS}" ]]; then
        CMD+=(--adapt-sweeps "${ADAPT_SWEEPS}")
    fi

    if [[ ${LOG_ADAPTATION} -eq 1 ]]; then
        CMD+=(--log-adaptation)
    fi
fi

BENCH_CMD="$(printf '%q ' "${CMD[@]}")"
NSYS_OUTPUT_PREFIX="nsys_reports/${SAMPLER}_${JOB_NAME}"

echo "==================================================================="
echo "Night Monitoring Job Submission"
echo "==================================================================="
echo "Job Name:       ${JOB_NAME}"
echo "Sampler:        ${SAMPLER}"
echo "Problem size:   ${PROBLEM_SIZE}"
echo "Sweeps/Reads:   ${NUM_SWEEPS}/${NUM_READS}"
echo "Output JSONL:   ${OUTPUT}"
echo "Partition:      ${SLURM_PARTITION}"
echo "GPUs requested: ${SLURM_GPUS}"
echo "Stdout log:     ${STDOUT_LOG}"
echo "Stderr log:     ${STDERR_LOG}"
echo "==================================================================="

mkdir -p "${HOME}/tmp"
readonly TMP_SCRIPT=$(mktemp "${HOME}/tmp/bench_night.XXXXXX.sh")
trap 'rm -f "${TMP_SCRIPT}"' EXIT

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH -t ${SLURM_TIME}
#SBATCH -N ${SLURM_NODES}
#SBATCH -n ${SLURM_TASKS}
#SBATCH -c ${SLURM_CPUS}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${STDOUT_LOG}
#SBATCH -e ${STDERR_LOG}

set -euo pipefail

cd ${ROOT_DIR}
source .venv/bin/activate
mkdir -p nsys_reports

NSYS_OUTPUT="${NSYS_OUTPUT_PREFIX}_\$(date +%Y%m%dT%H%M%S)"

echo "Running: nsys profile -t nvtx,cuda,osrt,cublas --stats=true -f true -o \${NSYS_OUTPUT} ${BENCH_CMD}"
eval "nsys profile -t nvtx,cuda,osrt,cublas \
    --stats=true \
    -f true \
    -o \"\${NSYS_OUTPUT}\" \
    ${BENCH_CMD}"
EOF

if sbatch "${TMP_SCRIPT}"; then
    echo "Submitted. Monitor with: squeue -u \$USER"
else
    echo "Job submission failed" >&2
    exit 1
fi
