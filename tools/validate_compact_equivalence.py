#!/usr/bin/env python3
"""Certify compact coefficients against production dense alpha on B16 entries."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.config_manager import Config
from ip4ch.Generate_alpha_coefficients import (
    _resolve_pair_file_for_atoms,
    get_energy,
)
from ip4ch.IntegerProgram import Allocate
from ip4ch.compact_backend import (
    aggregate_production_dense_alpha,
    build_orbit_data,
    build_pair_coefficients,
)
from ip4ch.generate_structure_entry import process_entry, read_json_entries
from multiprocessing_main import resolve_mlip_pair_paths


def _max_difference(left, right):
    values = [float(np.max(np.abs(left.linear - right.linear), initial=0.0))]
    keys = set(left.quadratic) | set(right.quadratic)
    values.extend(
        abs(float(left.quadratic.get(key, 0.0)) - float(right.quadratic.get(key, 0.0)))
        for key in keys
    )
    return max(values, default=0.0)


def _allocation(config, entry):
    structure = process_entry(entry)
    pair_file, pair_dir = resolve_mlip_pair_paths(config)
    return Allocate(
        ProgramRoot=config.get("base", "root"),
        DataDir=config.get("base", "data"),
        OutputDir=config.get("base", "output"),
        EntryDir=config.get("input", "entry"),
        GridsDir=config.get("input", "grids"),
        EffectivePairDir=config.get("input", "effective_pair"),
        IPResultsDir=config.get("output", "ip_results"),
        PlotDir=config.get("output", "plot_results"),
        lattice=entry["Lattice"],
        lattice_matrix=structure.lattice.matrix,
        lattice_type=entry["Lattice Type"],
        grid_size=entry["Grid Size"],
        full_formula_dict=entry["Full Formula Dictionary"],
        r_min=config.get("MLIP", "r_min"),
        r_max=config.get("MLIP", "r_max"),
        compound_type=str(config.get("Compound")),
        pair_type=str(config.get("MLIP", "pair_type")),
        pair_file=pair_file,
        pair_dir=pair_dir,
    )


def validate_entry(config, entry):
    allocation = _allocation(config, entry)
    group = entry["Space Group"]
    orbit_filename = f"SG{group}_G{allocation.grid_size}.json"
    orbits = allocation._load_or_generate_valid_orbits(group, orbit_filename)
    orbit_keys = list(orbits)
    site_count = int(np.prod(allocation.grid_size))
    orbit_by_site = []
    for site in range(site_count):
        owner = next(
            index for index, representative in enumerate(orbit_keys)
            if site == int(representative) or site in orbits[representative]
        )
        orbit_by_site.append(owner)
    blocked, edges = allocation._find_min_distance_orbit_conflicts(
        orbit_by_site, allocation.r_min
    )
    orbit_data = build_orbit_data(allocation.grid_size, orbits, blocked, edges)
    atom_pair = allocation.pair[0]
    resolved_pair = _resolve_pair_file_for_atoms(
        *atom_pair, pair_file=allocation.pair_file, pair_dir=allocation.pair_dir
    )
    dense = get_energy(
        t1=atom_pair[0], t2=atom_pair[1], phase_name=allocation.phase_name,
        lattice=allocation.lattice, lattice_matrix=allocation.lattice_matrix,
        r_min=allocation.r_min, r_max=allocation.r_max,
        grid_size=allocation.grid_size, frac_pos=allocation.frac_pos,
        pair_type=allocation.pair_type, DataPath=allocation.DataDir,
        compound_type=allocation.compound_type, pair_file=allocation.pair_file,
        pair_dir=allocation.pair_dir,
    )
    production = aggregate_production_dense_alpha(orbit_data, dense)
    compact, source = build_pair_coefficients(
        orbit_data=orbit_data,
        lattice_type=allocation.lattice_type,
        grid_size=allocation.grid_size,
        lattice_matrix=allocation.lattice_matrix,
        fractional_positions=allocation.frac_pos,
        r_min=allocation.r_min,
        r_max=allocation.r_max,
        pair_file=resolved_pair,
    )
    difference = _max_difference(production, compact)
    if difference > 1e-5:
        raise AssertionError(
            f"entry {entry['Index']} compact/dense difference {difference:.12g}"
        )
    return {
        "entry_id": entry["Index"],
        "lattice_type": allocation.lattice_type,
        "space_group": group,
        "grid_size": allocation.grid_size,
        "site_count": site_count,
        "original_orbits": len(orbit_data.original_orbit_ids),
        "active_orbits": len(orbit_data.active_original),
        "coefficient_source": source,
        "linear_terms": int(np.count_nonzero(compact.linear)),
        "quadratic_terms": len(compact.quadratic),
        "max_abs_difference": difference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config(config_path=args.config)
    entry_file = os.path.join(config.get("input", "entry"), "B", "B16.json")
    selected = {}
    for entry in read_json_entries(entry_file):
        key = (
            entry["Lattice Type"],
            json.dumps(entry["Lattice"], sort_keys=True),
            tuple(int(value) for value in entry["Grid Size"]),
        )
        selected.setdefault(key, entry)
    if not selected:
        raise ValueError("B16 entry file is empty")

    records = [validate_entry(config, selected[key]) for key in sorted(selected)]
    payload = {
        "status": "PASS",
        "scope": "one entry per unique lattice-matrix/grid-size/lattice-type combination",
        "combination_count": len(records),
        "entries": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
