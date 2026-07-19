#!/usr/bin/env python3
"""Select DFT-stable alloy structures absent from Materials Project references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


PAPER_BINARY_SYSTEMS = (
    "AlTi",
    "AlHf",
    "AlNb",
    "AlSc",
    "AlPd",
    "AlAu",
    "MgPd",
    "MgSc",
    "AgTi",
    "RhTi",
    "AuTi",
    "TiZn",
    "IrTi",
    "RuTi",
    "PdTi",
    "PtTi",
    "MoTi",
    "OsTi",
    "CuPd",
    "CuY",
    "CuSc",
)
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
ENERGY_COLUMN = "formation_energy_peratom"
MANIFEST_FIELDS = (
    "system",
    "phase",
    "filename",
    "reduced_formula",
    "number_of_atoms",
    "space_group_symbol",
    "space_group_number",
    "target_element",
    "target_element_fraction",
    "dft_formation_energy_eV_per_atom",
    "dft_distance_to_combined_hull_eV_per_atom",
    "mlip_formation_energy_eV_per_atom",
    "mlip_distance_to_search_hull_eV_per_atom",
    "materials_project_match",
    "materials_project_references_checked",
    "source_relative_path",
    "sha256",
)


def parse_formula(formula: str) -> dict[str, int]:
    tokens = FORMULA_TOKEN.findall(formula)
    if not tokens or "".join(f"{element}{count}" for element, count in tokens) != formula:
        raise ValueError(f"Unsupported chemical formula: {formula}")
    return {element: int(count or 1) for element, count in tokens}


def lower_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the lower binary convex envelope, including elemental endpoints."""
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


def hull_distance(hull: list[tuple[float, float]], x_value: float, energy: float) -> float:
    distance = energy - interpolate_hull(hull, x_value)
    return 0.0 if abs(distance) < 1e-10 else distance


def read_formation_energies(csv_path: Path) -> dict[str, float]:
    energies: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("File Name", "").strip()
            if filename.startswith(("struc", "mp-")):
                energies[filename] = float(row[ENERGY_COLUMN])
    return energies


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_matcher() -> StructureMatcher:
    return StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
    )


def load_structure(path: Path, cache: dict[Path, Structure]) -> Structure:
    if path not in cache:
        cache[path] = Structure.from_file(path)
    return cache[path]


def structure_metadata(path: Path) -> dict[str, object]:
    structure = Structure.from_file(path)
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
    return {
        "reduced_formula": structure.composition.reduced_formula,
        "number_of_atoms": len(structure),
        "space_group_symbol": analyzer.get_space_group_symbol(),
        "space_group_number": analyzer.get_space_group_number(),
    }


def _candidate_matches_mp(
    candidate_path: Path,
    mp_paths: list[Path],
    matcher: StructureMatcher,
    structure_cache: dict[Path, Structure],
) -> str | None:
    candidate = load_structure(candidate_path, structure_cache)
    for mp_path in mp_paths:
        reference = load_structure(mp_path, structure_cache)
        if matcher.fit(candidate, reference):
            return mp_path.name
    return None


def _deduplicate_records(
    records: list[dict[str, object]],
    matcher: StructureMatcher,
    structure_cache: dict[Path, Structure],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    representatives: list[dict[str, object]] = []
    duplicates: list[dict[str, str]] = []
    ordered = sorted(
        records,
        key=lambda record: (
            record["system"],
            record["phase"],
            record["dft_formation_energy_eV_per_atom"],
            record["filename"],
        ),
    )
    for record in ordered:
        source = Path(record["_source_path"])
        structure = load_structure(source, structure_cache)
        matched_record = None
        for representative in representatives:
            if representative["system"] != record["system"]:
                continue
            reference_path = Path(representative["_source_path"])
            reference = load_structure(reference_path, structure_cache)
            if matcher.fit(structure, reference):
                matched_record = representative
                break
        if matched_record is None:
            representatives.append(record)
        else:
            duplicates.append(
                {
                    "system": str(record["system"]),
                    "phase": str(record["phase"]),
                    "filename": str(record["filename"]),
                    "representative_phase": str(matched_record["phase"]),
                    "representative_filename": str(matched_record["filename"]),
                    "reason": "dft_structure_match",
                }
            )
    return representatives, duplicates


def _phase_composition(phase: str, system_elements: dict[str, int], target: str) -> float:
    formula = parse_formula(phase)
    if set(formula) != set(system_elements):
        raise ValueError(f"Unexpected phase {phase}; expected elements {tuple(system_elements)}")
    return formula[target] / sum(formula.values())


def select_novel_structures(
    dft_root: Path,
    mlip_root: Path,
    systems: Iterable[str] = PAPER_BINARY_SYSTEMS,
    dft_threshold: float = 0.001,
    mlip_threshold: float = 0.05,
    deduplicate: bool = True,
) -> dict[str, object]:
    if dft_threshold < 0 or mlip_threshold < 0:
        raise ValueError("Hull-distance thresholds must be non-negative")

    systems = tuple(systems)
    matcher = make_matcher()
    structure_cache: dict[Path, Structure] = {}
    selected_before_deduplication: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    system_summaries: dict[str, dict[str, object]] = {}

    for system in systems:
        system_formula = parse_formula(system)
        if len(system_formula) != 2:
            raise ValueError(f"Only binary systems are supported: {system}")
        dft_system = dft_root / system
        mlip_system = mlip_root / system
        if not dft_system.is_dir() or not mlip_system.is_dir():
            excluded.append(
                {
                    "system": system,
                    "reason": "missing_system_directory",
                    "dft_exists": dft_system.is_dir(),
                    "mlip_exists": mlip_system.is_dir(),
                }
            )
            continue

        target_element = tuple(system_formula)[1]
        dft_points: list[tuple[float, float]] = []
        mlip_points: list[tuple[float, float]] = []
        candidates: list[dict[str, object]] = []
        reference_count = 0

        for phase_dir in sorted(path for path in dft_system.iterdir() if path.is_dir()):
            composition = _phase_composition(phase_dir.name, system_formula, target_element)
            dft_csv = phase_dir / f"{phase_dir.name}_dft_formation_energy.csv"
            mlip_csv = mlip_system / phase_dir.name / f"{phase_dir.name}_nn_formation_energy.csv"
            if not dft_csv.is_file() or not mlip_csv.is_file():
                excluded.append(
                    {
                        "system": system,
                        "phase": phase_dir.name,
                        "reason": "missing_formation_energy_csv",
                        "dft_csv_exists": dft_csv.is_file(),
                        "mlip_csv_exists": mlip_csv.is_file(),
                    }
                )
                continue

            dft_energies = read_formation_energies(dft_csv)
            mlip_energies = read_formation_energies(mlip_csv)
            dft_points.extend((composition, energy) for energy in dft_energies.values())
            mlip_points.extend(
                (composition, energy)
                for filename, energy in mlip_energies.items()
                if filename.startswith("struc")
            )

            mlip_phase = mlip_system / phase_dir.name
            mp_paths = sorted(path for path in mlip_phase.glob("mp-*") if path.is_file())
            reference_count += len(mp_paths)
            structure_dir = phase_dir / "xyz2poscar"
            for dft_path in sorted(structure_dir.glob("struc*")):
                filename = dft_path.name
                if filename not in dft_energies:
                    excluded.append(
                        {
                            "system": system,
                            "phase": phase_dir.name,
                            "filename": filename,
                            "reason": "missing_dft_formation_energy",
                        }
                    )
                    continue
                mlip_path = mlip_phase / filename
                if filename not in mlip_energies or not mlip_path.is_file():
                    excluded.append(
                        {
                            "system": system,
                            "phase": phase_dir.name,
                            "filename": filename,
                            "reason": "missing_mlip_candidate",
                        }
                    )
                    continue
                candidates.append(
                    {
                        "system": system,
                        "phase": phase_dir.name,
                        "filename": filename,
                        "target_element": target_element,
                        "target_element_fraction": composition,
                        "dft_formation_energy_eV_per_atom": dft_energies[filename],
                        "mlip_formation_energy_eV_per_atom": mlip_energies[filename],
                        "mp_paths": mp_paths,
                        "_source_path": str(dft_path),
                        "_mlip_path": str(mlip_path),
                    }
                )

        if not dft_points or not mlip_points:
            excluded.append({"system": system, "reason": "insufficient_hull_data"})
            continue
        dft_hull = lower_hull(dft_points)
        mlip_hull = lower_hull(mlip_points)
        system_selected = 0
        matched_mp = 0
        failed_dft = 0
        failed_mlip = 0

        for candidate in candidates:
            composition = float(candidate["target_element_fraction"])
            dft_distance = hull_distance(
                dft_hull,
                composition,
                float(candidate["dft_formation_energy_eV_per_atom"]),
            )
            mlip_distance = hull_distance(
                mlip_hull,
                composition,
                float(candidate["mlip_formation_energy_eV_per_atom"]),
            )
            candidate["dft_distance_to_combined_hull_eV_per_atom"] = dft_distance
            candidate["mlip_distance_to_search_hull_eV_per_atom"] = mlip_distance
            if dft_distance > dft_threshold + 1e-12:
                failed_dft += 1
                excluded.append({**candidate, "reason": "above_dft_hull_threshold"})
                continue
            if mlip_distance > mlip_threshold + 1e-12:
                failed_mlip += 1
                excluded.append({**candidate, "reason": "above_mlip_hull_threshold"})
                continue

            matched_reference = _candidate_matches_mp(
                Path(candidate["_mlip_path"]),
                list(candidate["mp_paths"]),
                matcher,
                structure_cache,
            )
            if matched_reference is not None:
                matched_mp += 1
                excluded.append(
                    {
                        **candidate,
                        "reason": "matches_materials_project_reference",
                        "matched_reference": matched_reference,
                    }
                )
                continue

            source_path = Path(candidate["_source_path"])
            relative_path = source_path.relative_to(dft_root)
            selected_before_deduplication.append(
                {
                    **candidate,
                    **structure_metadata(source_path),
                    "materials_project_match": False,
                    "materials_project_references_checked": len(candidate["mp_paths"]),
                    "source_relative_path": relative_path.as_posix(),
                    "sha256": sha256(source_path),
                }
            )
            system_selected += 1

        system_summaries[system] = {
            "candidate_files": len(candidates),
            "materials_project_reference_files": reference_count,
            "selected_before_deduplication": system_selected,
            "excluded_above_dft_threshold": failed_dft,
            "excluded_above_mlip_threshold": failed_mlip,
            "excluded_matching_materials_project": matched_mp,
            "dft_hull_vertices": [
                {
                    "target_element_fraction": composition,
                    "formation_energy_eV_per_atom": energy,
                }
                for composition, energy in dft_hull[1:-1]
            ],
            "mlip_search_hull_vertices": [
                {
                    "target_element_fraction": composition,
                    "formation_energy_eV_per_atom": energy,
                }
                for composition, energy in mlip_hull[1:-1]
            ],
        }

    if deduplicate:
        records, duplicates = _deduplicate_records(
            selected_before_deduplication, matcher, structure_cache
        )
    else:
        records, duplicates = selected_before_deduplication, []

    for record in records:
        record.pop("mp_paths", None)
        record.pop("_source_path", None)
        record.pop("_mlip_path", None)
    for item in excluded:
        item.pop("mp_paths", None)
        item.pop("_source_path", None)
        item.pop("_mlip_path", None)

    records.sort(key=lambda item: (item["system"], item["phase"], item["filename"]))
    duplicates.sort(key=lambda item: (item["system"], item["phase"], item["filename"]))
    for system, summary in system_summaries.items():
        summary["selected_representatives"] = sum(
            record["system"] == system for record in records
        )
        summary["dft_duplicates_removed"] = sum(
            duplicate["system"] == system for duplicate in duplicates
        )

    return {
        "selection": {
            "systems": list(systems),
            "candidate_filename_prefix": "struc",
            "materials_project_filename_prefix": "mp-",
            "dft_hull_reference": "combined struc and mp binary DFT lower hull",
            "dft_threshold_eV_per_atom": dft_threshold,
            "mlip_hull_reference": "struc-only binary MLIP lower hull",
            "mlip_threshold_eV_per_atom": mlip_threshold,
            "structure_matcher": {
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol_degrees": 5.0,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": True,
            },
            "dft_deduplication": deduplicate,
        },
        "counts": {
            "systems_requested": len(systems),
            "systems_with_results": len(system_summaries),
            "selected_before_deduplication": len(selected_before_deduplication),
            "dft_duplicates_removed": len(duplicates),
            "selected_representatives": len(records),
        },
        "systems": system_summaries,
        "records": records,
        "duplicates": duplicates,
        "excluded": excluded,
    }


def copy_selected(dft_root: Path, output: Path, document: dict[str, object]) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to copy into non-empty directory: {output}")
    for record in document["records"]:
        relative_path = Path(record["source_relative_path"])
        destination = output / "structures" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dft_root / relative_path, destination)


def compact_document(document: dict[str, object]) -> dict[str, object]:
    excluded = document["excluded"]
    return {
        "selection": document["selection"],
        "counts": document["counts"],
        "systems": document["systems"],
        "records": document["records"],
        "duplicates": document["duplicates"],
        "excluded_counts_by_reason": dict(sorted(Counter(item["reason"] for item in excluded).items())),
        "reviewable_exclusions": [
            item for item in excluded if item["reason"] != "above_dft_hull_threshold"
        ],
    }


def write_dataset_metadata(output: Path, document: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "novel_alloy_structures.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in document["records"]:
            writer.writerow({field: record[field] for field in MANIFEST_FIELDS})
    (output / "selection_summary.json").write_text(
        json.dumps(compact_document(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dft_root", type=Path)
    parser.add_argument("mlip_root", type=Path)
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--dft-threshold", type=float, default=0.001)
    parser.add_argument("--mlip-threshold", type=float, default=0.05)
    parser.add_argument("--keep-dft-duplicates", action="store_true")
    parser.add_argument("--copy-to", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = tuple(args.systems or PAPER_BINARY_SYSTEMS)
    document = select_novel_structures(
        args.dft_root,
        args.mlip_root,
        systems=systems,
        dft_threshold=args.dft_threshold,
        mlip_threshold=args.mlip_threshold,
        deduplicate=not args.keep_dft_duplicates,
    )
    if args.copy_to is not None:
        copy_selected(args.dft_root, args.copy_to, document)
        write_dataset_metadata(args.copy_to, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
