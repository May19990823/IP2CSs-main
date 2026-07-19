import csv
from pathlib import Path

from pymatgen.core import Lattice, Structure

from tools.select_dft_novel_alloy_structures import (
    select_novel_structures,
    write_dataset_metadata,
)


def write_structure(path: Path, shift: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        Lattice.cubic(4.0),
        ["Al", "Cu"],
        [[0.0, 0.0, 0.0], [0.5 + shift, 0.5, 0.5]],
    )
    structure.to(filename=path, fmt="poscar")


def write_phase(
    dft_root: Path,
    mlip_root: Path,
    phase: str,
    dft_rows: list[tuple[str, float]],
    mlip_rows: list[tuple[str, float]],
    structures: dict[str, float],
) -> None:
    dft_phase = dft_root / "AlCu" / phase
    mlip_phase = mlip_root / "AlCu" / phase
    dft_phase.mkdir(parents=True)
    mlip_phase.mkdir(parents=True)
    for suffix, root, rows in (
        ("dft", dft_phase, dft_rows),
        ("nn", mlip_phase, mlip_rows),
    ):
        with (root / f"{phase}_{suffix}_formation_energy.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["File Name", "total_energy(eV)", "formation_energy_peratom"])
            for filename, energy in rows:
                writer.writerow([filename, 0.0, energy])
    for filename, shift in structures.items():
        if filename.startswith("struc"):
            write_structure(dft_phase / "xyz2poscar" / filename, shift)
        write_structure(mlip_phase / filename, shift)


def test_combined_dft_hull_includes_mp_references(tmp_path):
    dft_root = tmp_path / "dft"
    mlip_root = tmp_path / "mlip"
    rows = [("struct_candidate.vasp", -0.400), ("mp-1_AlCu.vasp", -0.405)]
    write_phase(
        dft_root,
        mlip_root,
        "Al1Cu1",
        rows,
        rows,
        {"struct_candidate.vasp": 0.04, "mp-1_AlCu.vasp": 0.0},
    )

    document = select_novel_structures(dft_root, mlip_root, systems=["AlCu"])

    assert document["records"] == []
    reasons = {item["reason"] for item in document["excluded"]}
    assert "above_dft_hull_threshold" in reasons


def test_mp_structure_match_excludes_known_candidate(tmp_path):
    dft_root = tmp_path / "dft"
    mlip_root = tmp_path / "mlip"
    rows = [("struct_known.vasp", -0.400), ("mp-1_AlCu.vasp", -0.399)]
    write_phase(
        dft_root,
        mlip_root,
        "Al1Cu1",
        rows,
        rows,
        {"struct_known.vasp": 0.0, "mp-1_AlCu.vasp": 0.0},
    )

    document = select_novel_structures(dft_root, mlip_root, systems=["AlCu"])

    assert document["records"] == []
    matched = [
        item
        for item in document["excluded"]
        if item["reason"] == "matches_materials_project_reference"
    ]
    assert matched[0]["matched_reference"] == "mp-1_AlCu.vasp"


def test_dft_duplicates_keep_lowest_energy_representative(tmp_path):
    dft_root = tmp_path / "dft"
    mlip_root = tmp_path / "mlip"
    dft_rows = [("struct_low.vasp", -0.4000), ("struct_near.vasp", -0.3995)]
    mlip_rows = [("struct_low.vasp", -0.4000), ("struct_near.vasp", -0.3995)]
    write_phase(
        dft_root,
        mlip_root,
        "Al1Cu1",
        dft_rows,
        mlip_rows,
        {"struct_low.vasp": 0.0, "struct_near.vasp": 0.0},
    )

    document = select_novel_structures(dft_root, mlip_root, systems=["AlCu"])

    assert [record["filename"] for record in document["records"]] == ["struct_low.vasp"]
    assert document["counts"]["selected_before_deduplication"] == 2
    assert document["counts"]["dft_duplicates_removed"] == 1
    assert document["duplicates"][0]["filename"] == "struct_near.vasp"


def test_novel_stable_candidate_records_structure_metadata(tmp_path):
    dft_root = tmp_path / "dft"
    mlip_root = tmp_path / "mlip"
    rows = [("struct_novel.vasp", -0.4)]
    write_phase(
        dft_root,
        mlip_root,
        "Al1Cu1",
        rows,
        rows,
        {"struct_novel.vasp": 0.0},
    )

    document = select_novel_structures(dft_root, mlip_root, systems=["AlCu"])

    record = document["records"][0]
    assert record["materials_project_match"] is False
    assert record["dft_distance_to_combined_hull_eV_per_atom"] == 0.0
    assert record["mlip_distance_to_search_hull_eV_per_atom"] == 0.0
    assert record["reduced_formula"] == "AlCu"
    assert record["number_of_atoms"] == 2
    assert record["sha256"]


def test_dataset_metadata_contains_manifest_and_compact_audit(tmp_path):
    dft_root = tmp_path / "dft"
    mlip_root = tmp_path / "mlip"
    rows = [("struct_novel.vasp", -0.4)]
    write_phase(
        dft_root,
        mlip_root,
        "Al1Cu1",
        rows,
        rows,
        {"struct_novel.vasp": 0.0},
    )
    document = select_novel_structures(dft_root, mlip_root, systems=["AlCu"])
    output = tmp_path / "output"

    write_dataset_metadata(output, document)

    with (output / "novel_alloy_structures.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["filename"] == "struct_novel.vasp"
    audit = (output / "selection_summary.json").read_text()
    assert '"selected_representatives": 1' in audit
    assert '"excluded_counts_by_reason"' in audit
