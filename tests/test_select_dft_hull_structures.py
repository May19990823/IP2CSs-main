import csv
import math
from pathlib import Path

from tools.select_dft_hull_structures import lower_hull, select_structures


def write_phase(root: Path, phase: str, rows: list[tuple[str, float]]) -> None:
    phase_dir = root / "AlCu" / phase
    structure_dir = phase_dir / "xyz2poscar"
    structure_dir.mkdir(parents=True)
    with (phase_dir / f"{phase}_dft_formation_energy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["File Name", "total_energy(eV)", "formation_energy_peratom"])
        for filename, energy in rows:
            writer.writerow([filename, "0.0", energy])
            if filename.startswith("struc"):
                (structure_dir / filename).write_text(f"test structure: {filename}\n")


def test_lower_hull_discards_points_above_the_lower_envelope():
    hull = lower_hull([(0.25, -0.3), (0.5, -0.1), (0.75, -0.3)])
    assert hull == [(0.0, 0.0), (0.25, -0.3), (0.75, -0.3), (1.0, 0.0)]


def test_selection_uses_dft_struc_rows_and_excludes_mp(tmp_path):
    write_phase(
        tmp_path,
        "Al1Cu1",
        [
            ("struct_hull_relaxed.vasp", -0.4),
            ("struct_near_relaxed.vasp", -0.3995),
            ("struct_high_relaxed.vasp", -0.398),
            ("mp-1_AlCu_relaxed.vasp", -0.6),
        ],
    )
    write_phase(tmp_path, "Al3Cu1", [("struct_quarter_relaxed.vasp", -0.3)])
    missing = tmp_path / "AlCu" / "Al1Cu1" / "xyz2poscar" / "struct_missing_relaxed.vasp"
    missing.write_text("missing energy\n")

    document = select_structures(tmp_path, threshold=0.001, excluded_systems=set())

    names = {record["filename"] for record in document["records"]}
    assert names == {
        "struct_hull_relaxed.vasp",
        "struct_near_relaxed.vasp",
        "struct_quarter_relaxed.vasp",
    }
    assert document["counts"] == {
        "scanned_struc_files": 5,
        "energy_mapped_files": 4,
        "missing_energy_files": 1,
        "selected_files": 3,
    }
    assert all(not name.startswith("mp-") for name in names)
    system = document["systems"]["AlCu"]
    assert system["on_hull_files"] == 2
    assert system["near_hull_files"] == 1
    assert math.isclose(system["maximum_selected_distance_eV_per_atom"], 0.0005)
    assert system["hull_vertices"] == [
        {
            "target_element_fraction": 0.25,
            "formation_energy_eV_per_atom": -0.3,
        },
        {
            "target_element_fraction": 0.5,
            "formation_energy_eV_per_atom": -0.4,
        },
    ]
