import csv

import numpy as np
import pytest

from b168_model_opt.coefficients import evaluate_compact
from b168_model_opt.pair_table import PairTable
from b168_model_opt.periodic_kernel import build_periodic_kernel
from ip4ch.Generate_alpha_coefficients import (
    _load_csv_pair_table,
    _resolve_pair_file_for_atoms,
    generate_hexagonal,
    generate_orthorhombic,
    get_energy_table,
)
from ip4ch.compact_backend import (
    aggregate_production_dense_alpha,
    build_orbit_data,
    build_pair_coefficients,
)


def _write_pair(path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_ij", "median_smoothed", "E_ij"])
        for index in range(0, 61):
            distance = index / 10
            writer.writerow([distance, 1000 + distance, -distance])


def test_pair_parsers_explicitly_prefer_eij(tmp_path):
    path = tmp_path / "B-B.csv"
    _write_pair(path)

    production = _load_csv_pair_table(path)
    compact = PairTable.from_csv(path)

    assert production[2.0] == -2.0
    assert compact.energy(2.0) == -2.0


def test_explicit_decorated_pair_filename_is_resolved(tmp_path):
    path = tmp_path / "B-B_maincluster_1p65_rmin1p40.csv"
    _write_pair(path)

    assert _resolve_pair_file_for_atoms("B", "B", pair_file=path) == path


def test_translation_kernel_matches_production_origin_row(tmp_path):
    pair_path = tmp_path / "B-B.csv"
    _write_pair(pair_path)
    grid = (2, 2, 2)
    cell = np.diag([4.0, 4.0, 4.0])
    frac_pos = generate_orthorhombic(grid)

    alpha = get_energy_table(
        "B", "B", "B", cell, 1.4, 4.0, grid, frac_pos,
        "B_pair", str(tmp_path), pair_file=str(pair_path),
    )
    kernel = build_periodic_kernel(
        grid, cell, 1.4, 4.0, PairTable.from_csv(pair_path)
    )

    np.testing.assert_array_equal(alpha[0], kernel.reshape(-1))


def test_dense_aggregation_preserves_legacy_objective():
    grid = (2, 2, 1)
    orbits = {"0": [1], "2": [], "3": []}
    orbit_data = build_orbit_data(grid, orbits, set(), {(1, 2)})
    alpha = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    coefficients = aggregate_production_dense_alpha(orbit_data, alpha)

    for assignment in (
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
    ):
        occupied = np.flatnonzero(assignment)
        sites = []
        for active in occupied:
            original = orbit_data.active_original[int(active)]
            sites.extend(orbit_data.members[original])
        legacy = sum(alpha[i, j] for i in sites for j in sites if i < j)
        assert evaluate_compact(assignment, coefficients) == pytest.approx(legacy)


def test_hexagonal_sparse_stream_matches_dense_alpha(tmp_path):
    pair_path = tmp_path / "B-B.csv"
    _write_pair(pair_path)
    grid = (2, 2, 1)
    cell = np.array([[4.0, 0.0, 0.0], [-2.0, 3.464101615, 0.0], [0.0, 0.0, 4.0]])
    positions = generate_hexagonal(grid)
    orbit_data = build_orbit_data(
        grid, {str(index): [] for index in range(4)}, set(), set()
    )
    dense = get_energy_table(
        "B", "B", "B", cell, 1.4, 4.0, grid, positions,
        "B_pair", str(tmp_path), pair_file=str(pair_path),
    )
    expected = aggregate_production_dense_alpha(orbit_data, dense)
    actual, source = build_pair_coefficients(
        orbit_data=orbit_data,
        lattice_type="Hexagonal",
        grid_size=grid,
        lattice_matrix=cell,
        fractional_positions=positions,
        r_min=1.4,
        r_max=4.0,
        pair_file=pair_path,
    )

    assert source == "sparse_production_neighbour_stream"
    np.testing.assert_array_equal(actual.linear, expected.linear)
    assert dict(actual.quadratic) == dict(expected.quadratic)
