#!/usr/bin/env python3
"""Select DFT-relaxed search structures on or near a binary lower hull."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable
from pathlib import Path


FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    tokens = FORMULA_TOKEN.findall(formula)
    if not tokens or "".join(f"{element}{count}" for element, count in tokens) != formula:
        raise ValueError(f"Unsupported chemical formula: {formula}")
    return {element: int(count or 1) for element, count in tokens}


def lower_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the lower convex envelope sorted by composition."""
    minimum_by_x: dict[float, float] = {0.0: 0.0, 1.0: 0.0}
    for x_value, energy in points:
        minimum_by_x[x_value] = min(minimum_by_x.get(x_value, math.inf), energy)

    hull: list[tuple[float, float]] = []
    for point in sorted(minimum_by_x.items()):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (
                (second[0] - first[0]) * (point[1] - first[1])
                - (second[1] - first[1]) * (point[0] - first[0])
            )
            if cross > 1e-14:
                break
            hull.pop()
        hull.append(point)
    return hull


def interpolate_hull(hull: list[tuple[float, float]], x_value: float) -> float:
    for left, right in zip(hull, hull[1:]):
        if left[0] - 1e-12 <= x_value <= right[0] + 1e-12:
            fraction = (x_value - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    raise ValueError(f"Composition {x_value} is outside the hull domain")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_formation_energies(csv_path: Path) -> dict[str, float]:
    energies: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("File Name", "")
            if not filename.startswith("struc"):
                continue
            energies[filename] = float(row["formation_energy_peratom"])
    return energies


def select_structures(
    root: Path,
    threshold: float,
    excluded_systems: set[str],
) -> dict[str, object]:
    if threshold < 0:
        raise ValueError("The hull-distance threshold must be non-negative")

    records: list[dict[str, object]] = []
    missing_energy: list[dict[str, str]] = []
    system_summaries: dict[str, dict[str, object]] = {}
    scanned = 0
    mapped = 0

    for system_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if system_dir.name in excluded_systems:
            continue
        system_formula = parse_formula(system_dir.name)
        if len(system_formula) != 2:
            continue
        target_element = tuple(system_formula)[1]

        phase_rows: list[tuple[Path, str, float, float]] = []
        minimum_points: list[tuple[float, float]] = []
        system_scanned = 0
        system_mapped = 0

        for phase_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
            phase_formula = parse_formula(phase_dir.name)
            if not set(phase_formula).issubset(system_formula):
                raise ValueError(f"Unexpected phase {phase_dir.name} in {system_dir.name}")
            atom_count = sum(phase_formula.values())
            composition = phase_formula.get(target_element, 0) / atom_count
            energy_csv = phase_dir / f"{phase_dir.name}_dft_formation_energy.csv"
            if not energy_csv.is_file():
                continue
            energies = read_formation_energies(energy_csv)
            minimum_points.extend((composition, energy) for energy in energies.values())

            structure_dir = phase_dir / "xyz2poscar"
            if not structure_dir.is_dir():
                continue
            for structure_path in sorted(structure_dir.glob("struc*")):
                if not structure_path.is_file():
                    continue
                scanned += 1
                system_scanned += 1
                if structure_path.name not in energies:
                    missing_energy.append(
                        {
                            "system": system_dir.name,
                            "phase": phase_dir.name,
                            "filename": structure_path.name,
                            "reason": "missing_dft_formation_energy",
                        }
                    )
                    continue
                mapped += 1
                system_mapped += 1
                phase_rows.append(
                    (structure_path, phase_dir.name, composition, energies[structure_path.name])
                )

        if not minimum_points:
            continue
        hull = lower_hull(minimum_points)
        selected_count = 0
        on_hull_count = 0
        near_hull_count = 0
        maximum_selected_distance = 0.0
        for structure_path, phase, composition, formation_energy in phase_rows:
            distance = formation_energy - interpolate_hull(hull, composition)
            if distance > threshold + 1e-12:
                continue
            if abs(distance) < 1e-12:
                distance = 0.0
                on_hull_count += 1
            else:
                near_hull_count += 1
            maximum_selected_distance = max(maximum_selected_distance, distance)
            relative_path = structure_path.relative_to(root)
            records.append(
                {
                    "system": system_dir.name,
                    "phase": phase,
                    "filename": structure_path.name,
                    "formation_energy_eV_per_atom": formation_energy,
                    "distance_to_hull_eV_per_atom": distance,
                    "source_relative_path": relative_path.as_posix(),
                    "sha256": sha256(structure_path),
                }
            )
            selected_count += 1

        system_summaries[system_dir.name] = {
            "scanned_struc_files": system_scanned,
            "energy_mapped_files": system_mapped,
            "selected_files": selected_count,
            "on_hull_files": on_hull_count,
            "near_hull_files": near_hull_count,
            "maximum_selected_distance_eV_per_atom": maximum_selected_distance,
            "hull_vertices": [
                {
                    "target_element_fraction": composition,
                    "formation_energy_eV_per_atom": formation_energy,
                }
                for composition, formation_energy in hull[1:-1]
            ],
        }

    records.sort(key=lambda item: (item["system"], item["phase"], item["filename"]))
    missing_energy.sort(key=lambda item: (item["system"], item["phase"], item["filename"]))
    return {
        "selection": {
            "energy_source": "*_dft_formation_energy.csv",
            "hull_reference": "struc-only binary DFT lower convex hull",
            "distance_operator": "<=",
            "threshold_eV_per_atom": threshold,
            "excluded_systems": sorted(excluded_systems),
            "excluded_filename_prefixes": ["mp-"],
        },
        "counts": {
            "scanned_struc_files": scanned,
            "energy_mapped_files": mapped,
            "missing_energy_files": len(missing_energy),
            "selected_files": len(records),
        },
        "systems": system_summaries,
        "records": records,
        "excluded": missing_energy,
    }


def copy_selected(root: Path, output: Path, document: dict[str, object]) -> None:
    for record in document["records"]:
        relative_path = Path(record["source_relative_path"])
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative_path, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="EP-AGEMS Results/dft_results directory")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--exclude-system", action="append", default=[])
    parser.add_argument("--copy-to", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = select_structures(args.root, args.threshold, set(args.exclude_system))
    if args.copy_to is not None:
        copy_selected(args.root, args.copy_to, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
