# DFT-Stable Alloy Structures

This directory contains DFT-relaxed alloy structures selected from the
EP-AGEMS `Results/dft_results` dataset. Only search-generated files whose names
start with `struc` are included.

## Selection Rule

- Formation energies are read from `*_dft_formation_energy.csv`. Those tables
  were generated from DFT total energies in `energy.csv` using DFT elemental
  reference chemical potentials.
- Only `struc*` rows are used to construct each binary lower convex hull.
  Materials Project (`mp-*`) rows do not participate in the hull and are not
  distributed here.
- Pure-element endpoints are assigned zero formation energy.
- A structure is retained when its vertical distance to the lower hull is
  less than or equal to `0.001 eV/atom`.
- `SSn` and `SSn_Ref` are excluded because this release is limited to alloys.

The selection contains 154 structures from 29 alloy systems. Of 3,972 alloy
`struc*` files scanned, 3,928 had a matching DFT formation energy. The 44 files
without a matching energy are listed in `excluded_missing_energy.csv` and were
not classified.

## Alignment With The Supplementary Information

The DFT panels in Section S4 of the supplementary information construct a
separate search-structure hull for each binary alloy system; Materials Project
points and their dashed hull are shown only for comparison. This release uses
the same per-system, DFT search-hull definition, then applies the stated
`0.001 eV/atom` stability tolerance to retain both exact hull structures and
near-hull structures.

Twenty systems and 121 released structures correspond to the DFT hulls in
Figs. S23 and S25-S44. The remaining 33 structures are current DFT results for
nine additional systems that are not plotted in the supplementary
information: Ag-Ti, Al-Rh, Cd-Mg, Cd-Ti, Cu-Ti, Cu-Y, Li-Mg, Mg-Sc, and Mg-Y.
The additional systems are evaluated with exactly the same per-system DFT hull
rule.

`system_hull_summary.csv` identifies the relevant SI figure for every plotted
system, labels additional systems, and lists the exact hull phases and counts
used for this release. Al-Ti appears in both Fig. S23 and Fig. S32.

## Layout

Structures retain their source hierarchy:

```text
<system>/<phase>/xyz2poscar/struc*_relaxed.vasp
```

`selection_manifest.csv` records the DFT formation energy, distance to the
hull, source-relative path, and SHA-256 checksum for every included structure.
Aggregate counts, hull vertices, and per-system statistics are in
`selection_summary.json`.

## Reproduce The Selection

Given an EP-AGEMS `Results/dft_results` directory, run:

```bash
python tools/select_dft_hull_structures.py /path/to/Results/dft_results \
  --threshold 0.001 \
  --exclude-system SSn \
  --exclude-system SSn_Ref \
  --copy-to data/dft_hull_near_structures
```

The command writes the full machine-readable selection document to standard
output and copies the selected structures when `--copy-to` is provided.
