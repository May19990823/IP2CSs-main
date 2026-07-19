#!/bin/bash
#SBATCH --partition=256G56c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --qos=qos7d
#SBATCH --time=7-00:00:00

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch run_b16_56c.sbatch PROJECT_ROOT RUN_CONFIG" >&2
    exit 2
fi

PROJECT_ROOT=$(realpath "$1")
RUN_CONFIG=$(realpath "$2")
PY=${PYTHON:-python}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
ulimit -l unlimited

JOB_TMP="/tmp/b16_${USER}_${SLURM_JOB_ID}"
install -d -m 700 "$JOB_TMP" "$JOB_TMP/gurobi_nodes"
trap 'rm -rf -- "$JOB_TMP"' EXIT
export TMPDIR="$JOB_TMP"
export GUROBI_NODEFILE_DIR="$JOB_TMP/gurobi_nodes"

mkdir -p "$PROJECT_ROOT/results/logs"
exec >"$PROJECT_ROOT/results/logs/slurm_${SLURM_JOB_ID}.out" \
     2>"$PROJECT_ROOT/results/logs/slurm_${SLURM_JOB_ID}.err"

echo "started_at=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "project_root=$PROJECT_ROOT"
echo "run_config=$RUN_CONFIG"
echo "python=$PY"
sha256sum "$PROJECT_ROOT/multiprocessing_main.py" "$RUN_CONFIG"
scontrol show job "$SLURM_JOB_ID" -o

"$PY" "$PROJECT_ROOT/multiprocessing_main.py" --config "$RUN_CONFIG"

echo "finished_at=$(date --iso-8601=seconds)"
