#!/bin/bash

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration Variables
#-------------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly PROGRAM_DESCRIPTION="Digital Annealer Tuning Job (DA / PTDA)"
readonly PROGRAM_VERSION="0.1.0"

readonly JOB_NAME="${JOB_NAME:-tuning}"
readonly TS="$(date +%Y%m%dT%H%M%S)"
readonly LOG_DIR="./log_slurm"
readonly STDOUT_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stdout.log"
readonly STDERR_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stderr.log"

readonly SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
readonly SLURM_TIME="${SLURM_TIME:-08:00:00}"
readonly SLURM_NODES="${SLURM_NODES:-1}"
readonly SLURM_TASKS="${SLURM_TASKS:-1}"
readonly SLURM_CPUS="${SLURM_CPUS:-8}"
readonly SLURM_GPUS="${SLURM_GPUS:-1}"

SAMPLER="all"  # da | ptda | ptda_adapt | all

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --sampler da|ptda|ptda_adapt|all     Which tuning benchmark to run (default: all)
  -h, --help                 Show this help

SLURM environment overrides:
  JOB_NAME, SLURM_PARTITION, SLURM_TIME, SLURM_NODES,
  SLURM_TASKS, SLURM_CPUS, SLURM_GPUS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --sampler" >&2
                usage
                exit 2
            fi
            SAMPLER="$2"
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

if [[ "${SAMPLER}" != "da" && "${SAMPLER}" != "ptda" && "${SAMPLER}" != "ptda_adapt" && "${SAMPLER}" != "all" ]]; then
    echo "--sampler must be one of: da|ptda|ptda_adapt|all" >&2
    exit 2
fi

#-------------------------------------------------------------------------------
# Preparation
#-------------------------------------------------------------------------------

echo "==================================================================="
echo "SLURM Job Submission (Tuning)"
echo "==================================================================="
echo "Job Name:       ${JOB_NAME}"
echo "Sampler mode:   ${SAMPLER}"
echo "Partition:      ${SLURM_PARTITION}"
echo "Nodes:          ${SLURM_NODES}"
echo "Tasks:          ${SLURM_TASKS}"
echo "CPUs per task:  ${SLURM_CPUS}"
echo "GPUs requested: ${SLURM_GPUS}"
echo "-------------------------------------------------------------------"
echo "Log Directory:  ${LOG_DIR}"
echo "Standard Out:   ${STDOUT_LOG}"
echo "Standard Error: ${STDERR_LOG}"
echo "==================================================================="

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/benchmarks/outputs/tuning" "${HOME}/tmp"

RUN_BLOCK=""
case "${SAMPLER}" in
    da)
        RUN_BLOCK="python3 benchmarks/tune_da.py"
        ;;
    ptda)
        RUN_BLOCK="python3 benchmarks/tune_ptda.py"
        ;;
    ptda_adapt)
        RUN_BLOCK="python3 benchmarks/tune_ptda_adaptation.py"
        ;;
    all)
        RUN_BLOCK=$'python3 benchmarks/tune_da.py\npython3 benchmarks/tune_ptda.py\npython3 benchmarks/tune_ptda_adaptation.py'
        ;;
esac

#-------------------------------------------------------------------------------
# Generate Temporary SBATCH Script
#-------------------------------------------------------------------------------

readonly TMP_SCRIPT=$(mktemp "${HOME}/tmp/tuning_run.XXXXXX.sh")
trap 'rm -f "${TMP_SCRIPT}"' EXIT

echo "Generating temporary SBATCH script: ${TMP_SCRIPT}"

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

echo "==================================================================="
echo "Starting Job: ${JOB_NAME}"
echo "==================================================================="
echo "Hostname:       \$(hostname)"
echo "Start Time:     \$(date)"
echo "Working Dir:    \$(pwd)"
echo "CPUs per task:  ${SLURM_CPUS}"
echo "GPUs requested: ${SLURM_GPUS}"
echo "Program:        ${PROGRAM_DESCRIPTION}"
echo "Version:        ${PROGRAM_VERSION}"
echo "Sampler mode:   ${SAMPLER}"
echo "==================================================================="

cd "${ROOT_DIR}"
source .venv/bin/activate

${RUN_BLOCK}

echo "==================================================================="
echo "Job Completed: ${JOB_NAME}"
echo "End Time:       \$(date)"
echo "==================================================================="
EOF

#-------------------------------------------------------------------------------
# Submit Job
#-------------------------------------------------------------------------------

echo "Submitting job to SLURM..."
if sbatch "${TMP_SCRIPT}"; then
    echo "Submitted. Monitor with: squeue -u \$USER"
    echo "View stdout: tail -f ${STDOUT_LOG}"
    echo "View stderr: tail -f ${STDERR_LOG}"
else
    echo "Job submission failed" >&2
    exit 1
fi

echo "==================================================================="
echo "Submission complete."
echo "==================================================================="
