# DFT-Stable Novel Alloy Structures

This directory contains 58 DFT-relaxed alloy structures selected in a fresh,
auditable reanalysis of the binary systems used in the manuscript. All released
files begin with `struc`; no Materials Project (`mp-*`) structure is included.

## Selection Rule

A search structure is retained when all of the following conditions hold:

1. Its MLIP-relaxed structure does not match any available `mp-*` reference at
   the same composition. Matching uses pymatgen `StructureMatcher` with
   `ltol=0.2`, `stol=0.3`, `angle_tol=5` degrees, primitive-cell reduction,
   scaling, and supercell matching enabled.
2. Its MLIP formation energy is no more than `0.05 eV/atom` above the binary
   lower hull constructed from `struc*` search candidates.
3. Its DFT formation energy is on or no more than `0.001 eV/atom` above the
   binary lower hull constructed jointly from DFT-relaxed `struc*` candidates
   and `mp-*` competing phases.
4. Structures that become equivalent after DFT relaxation are clustered with
   the same `StructureMatcher`; the lowest-DFT-energy representative is kept.

The result contains 39 on-hull structures and 19 structures within
`1 meV/atom` of the DFT hull. No DFT duplicate was removed in this particular
selection. The 58 structures occur in 15 of the 21 evaluated binary systems:

| System | Structures | System | Structures | System | Structures |
| --- | ---: | --- | ---: | --- | ---: |
| AlTi | 3 | AlHf | 3 | AlNb | 0 |
| AlSc | 0 | AlPd | 1 | AlAu | 0 |
| MgPd | 5 | MgSc | 0 | AgTi | 0 |
| RhTi | 4 | AuTi | 0 | TiZn | 4 |
| IrTi | 3 | RuTi | 5 | PdTi | 5 |
| PtTi | 3 | MoTi | 5 | OsTi | 4 |
| CuPd | 10 | CuY | 1 | CuSc | 2 |

The manuscript reports 33 novel DFT phases using the historical analysis
state. This directory is deliberately not forced to reproduce that count: it
applies the documented thresholds and strict matcher to the currently
available result tree, and therefore contains 58 structures. The current
`dft_results` archive has no DFT directories for the three ternary benchmark
systems, so this release makes no unsupported ternary assignment.

## Files

- `structures/SYSTEM/PHASE/xyz2poscar/*.vasp`: selected DFT-relaxed structures.
- `novel_alloy_structures.csv`: one row per released structure, including
  composition, space group, DFT and MLIP formation energies, both hull
  distances, MP-reference count, source-relative path, and SHA-256 checksum.
- `selection_summary.json`: selection parameters, per-system hull vertices and
  counts, released records, MP-matched exclusions, and data-quality exceptions.

## Reproduce The Selection

Given the original EP-AGEMS result tree, run:

```bash
python tools/select_dft_novel_alloy_structures.py \
  /path/to/EP-AGEMS/Results/dft_results \
  /path/to/EP-AGEMS/Results/nn_results \
  --dft-threshold 0.001 \
  --mlip-threshold 0.05 \
  --copy-to reproduced_dft_novel_alloy_structures \
  > full_selection_audit.json
```

The copied structures can be checked against the manifest with:

```bash
python - <<'PY'
import csv
import hashlib
from pathlib import Path

root = Path("data/dft_novel_alloy_structures")
with (root / "novel_alloy_structures.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    path = root / "structures" / row["source_relative_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
print(f"verified {len(rows)} structures")
PY
```
