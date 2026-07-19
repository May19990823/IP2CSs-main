from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "hpc" / "run_b16_56c.sbatch"


def test_slurm_runtime_uses_short_job_local_tmpdir():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'export TMPDIR="$RUN_ROOT/tmp"' not in script
    assert 'JOB_TMP="/tmp/b16_${USER}_${SLURM_JOB_ID}"' in script
    assert 'export TMPDIR="$JOB_TMP"' in script
    assert 'export GUROBI_NODEFILE_DIR="$JOB_TMP/gurobi_nodes"' in script
