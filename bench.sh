#!/bin/bash

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#-------------------------------------------------------------------------------
# Configuration Variables
#-------------------------------------------------------------------------------

# Program metadata
readonly PROGRAM_DESCRIPTION="Digital Annealer Benchmark Suite - Comparing performance with OpenJij"
readonly PROGRAM_VERSION="0.1.0"

# Job name (can be overridden via environment variable)
readonly JOB_NAME="${JOB_NAME:-bench}"

# Log directory configuration
readonly TS="$(date +%Y%m%dT%H%M%S)"
readonly LOG_DIR="./log_slurm"
readonly STDOUT_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stdout.log"
readonly STDERR_LOG="${LOG_DIR}/${JOB_NAME}_${TS}_stderr.log"
readonly TMP_DIR="${HOME}/tmp"

# SLURM configuration (can be overridden via environment variables)
readonly SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
readonly SLURM_TIME="${SLURM_TIME:-1000:00:00}"
readonly SLURM_NODES="${SLURM_NODES:-1}"
readonly SLURM_TASKS="${SLURM_TASKS:-1}"
readonly SLURM_CPUS="${SLURM_CPUS:-10}"
readonly SLURM_GPUS="${SLURM_GPUS:-1}"


#-------------------------------------------------------------------------------
# Preparation
#-------------------------------------------------------------------------------

echo "==================================================================="
echo "SLURM Job Submission"
echo "==================================================================="
echo "Job Name:       ${JOB_NAME}"
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

# Create log directory if it doesn't exist
if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Creating log directory: ${LOG_DIR}"
    mkdir -p "${LOG_DIR}"
fi

#-------------------------------------------------------------------------------
# Generate Temporary SBATCH Script
#-------------------------------------------------------------------------------

echo "Generating temporary SBATCH script..."
mkdir -p "${TMP_DIR}"
readonly TMP_SCRIPT=$(mktemp "${TMP_DIR}/bench_run.XXXXXX.sh")
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

# Exit on error
set -e

cd "${ROOT_DIR}"

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
echo "==================================================================="

# Run benchmark
source .venv/bin/activate

python3 benchmarks/bench_compare_openjij.py --num-reads 100
python3 benchmarks/bench_compare_openjij.py --num-reads 100 --beta-max 5.0
python3 benchmarks/bench_compare_openjij.py --num-reads 100 --num-sweeps 10000
python3 benchmarks/bench_compare_openjij.py --mode reads


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
    echo "✓ Job submitted successfully"
    echo "  Monitor with: squeue -u \$USER"
    echo "  View stdout:  tail -f ${STDOUT_LOG}"
    echo "  View stderr:  tail -f ${STDERR_LOG}"
else
    echo "✗ Job submission failed" >&2
    exit 1
fi

echo "==================================================================="
echo "Submission complete."
echo "==================================================================="
