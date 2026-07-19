# Boron Allotrope Integer-Programming Search

This repository is the minimal reproducibility package for the boron
allotrope search described in the associated article. It contains the search
code, the compact quadratic Gurobi model, the B16 input-generation workflow,
and the public `r_min = 1.50 A` effective-pair table.

The code is based on the production `boron_allotropes` workflow. The compact
backend and its regression tests are the narrowly scoped updates used for the
formal B16 calculation.

## Scope

The reference calculation searches elemental B16 structures with:

- 225 lattice/grid/space-group entries;
- `GridGeneration.r_min = 1.4 A`;
- `MLIP.r_min = 1.5 A`;
- the `compact_quadratic` backend;
- four workers on a 56-core node and 14 Gurobi threads per worker;
- 9000 s per entry, `PoolSolutions = 200`, `PoolSearchMode = 2`;
- `MIPGap = 1e-3`, `MIPFocus = 1`, and `Seed = 0`.

The repository intentionally excludes production outputs, Slurm logs,
caches, private environment files, and unrelated boron calculations.

## Requirements

- Linux and Python 3.11
- A working Gurobi installation and license
- 256 GB RAM and 56 CPU cores for the reference production run

Gurobi is proprietary software and is not distributed under this repository's
MIT License. Academic users can obtain a license from Gurobi.

Create an environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` records the direct package versions used for the public
regression run. Compatible Gurobi/Python combinations may also work, but the
pinned environment is the supported reproduction target.

Confirm that Gurobi is usable before starting the search:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## Verify The Installation

Run the regression suite:

```bash
python -m pytest -q
```

The compact-backend tests check objective equivalence, minimum-distance
constraints, solver metadata, pair-file resolution, and runtime paths.

## Reproduce The B16 Search

Run every command below from the repository root. First generate the B16
lattice/grid entries without invoking Gurobi:

```bash
python tools/prepare_entries.py \
  --config configs/b16_rmin1p50.yml \
  --summary results/b16_input_summary.json
```

The generated file `data/entry/B/B16.json` must contain 225 entries:

```bash
python - <<'PY'
import json
from pathlib import Path

entries = json.loads(Path("data/entry/B/B16.json").read_text())
assert len(entries) == 225, len(entries)
print("B16 entries:", len(entries))
PY
```

Optionally certify the compact coefficients against the production dense
formulation for each unique B16 lattice/grid combination:

```bash
python tools/validate_compact_equivalence.py \
  --config configs/b16_rmin1p50.yml \
  --output results/compact_equivalence.json
```

The command must finish with `status: PASS` and zero maximum absolute
coefficient difference for every checked combination.

Start the complete calculation:

```bash
python multiprocessing_main.py --config configs/b16_rmin1p50.yml
```

On a scheduler with a `256G56c` partition, the reference Slurm launcher is:

```bash
sbatch hpc/run_b16_56c.sbatch "$PWD" "$PWD/configs/b16_rmin1p50.yml"
```

Set `PYTHON=/absolute/path/to/python` before `sbatch` when the desired Python
is not the default executable on compute nodes. The partition, QoS, and wall
time directives may need adjustment for another cluster.

Solver records and structures are written below:

```text
results/ip_result/B/B16/B16_rmin1p50/
```

`TIME_LIMIT` in a solver record is a normal Gurobi termination status. It is
different from a scheduler-level Slurm timeout.

## Effective-Pair Data

The public table is:

```text
data/effective_pair/B-B_maincluster_1p65_rmin1p50.csv
```

Its SHA-256 checksum is:

```text
b1b848ff4023ee469a74c46c59aa719ea70d81109c0ebc9acd0b511994e59135
```

The first tabulated distance is 1.50 A and the cutoff is 5.00 A. The `E_ij`
column is the energy used by the integer-programming model.

## Repository Layout

```text
Utils/                 configuration and composition helpers
b168_model_opt/        compact orbit/pair model helpers
configs/               portable B16 configuration
data/effective_pair/   public B-B effective-pair table
hpc/                   reference Slurm launcher
ip4ch/                 integer-programming and pair-energy implementation
ipcss/                 lattice/grid enumeration and compatibility checks
tests/                 regression and equivalence tests
tools/                 input preparation and validation utilities
```

## License

The code and included effective-pair table are released under the MIT License.
See `LICENSE`.
