import os
import json
import hashlib
import sys

import numpy as np
from ase.spacegroup import Spacegroup
from pymatgen.core import Element, Structure
import spglib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ip4ch.orbit_utils import assert_valid_orbit_partition


ORBIT_CACHE_SCHEMA_VERSION = 2


def _cell_signature(lattice_matrix):
    if lattice_matrix is None:
        return None
    rounded = np.round(np.asarray(lattice_matrix, dtype=float), 8)
    payload = json.dumps(rounded.tolist(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _file_signature(path):
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def orbit_cache_metadata_path(orbit_path):
    return f"{orbit_path}.meta"


def build_orbit_cache_metadata(
    target_space_group,
    grid_size,
    lattice_matrix=None,
    reference_structure_file=None,
    method=None,
    symprec=None,
    angle_tolerance=None,
):
    metadata = {
        "schema_version": ORBIT_CACHE_SCHEMA_VERSION,
        "target_space_group": int(target_space_group),
        "grid_size": [int(value) for value in grid_size],
        "source": "reference" if reference_structure_file else (
            "lattice" if lattice_matrix is not None else "spacegroup"
        ),
    }
    if lattice_matrix is not None and reference_structure_file is None:
        metadata["lattice_signature"] = _cell_signature(lattice_matrix)
    if reference_structure_file is not None:
        metadata["reference_basename"] = os.path.basename(reference_structure_file)
        metadata["reference_signature"] = _file_signature(reference_structure_file)
    if method is not None:
        metadata["method"] = str(method)
    if symprec is not None:
        metadata["symprec"] = float(symprec)
    if angle_tolerance is not None:
        metadata["angle_tolerance"] = float(angle_tolerance)
    return metadata


def orbit_cache_metadata_matches(orbit_path, expected_metadata):
    metadata_path = orbit_cache_metadata_path(orbit_path)
    if not os.path.exists(metadata_path):
        return False, f"missing metadata {metadata_path}"
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            actual_metadata = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"could not read metadata {metadata_path}: {exc}"

    for key, expected_value in expected_metadata.items():
        actual_value = actual_metadata.get(key)
        if actual_value != expected_value:
            return (
                False,
                f"metadata field {key!r} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}",
            )
    return True, "metadata matches"


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def _write_orbit_cache(GridsDir, orbit_filename, orbit_dict, metadata):
    os.makedirs(GridsDir, exist_ok=True)
    orbit_path = os.path.join(GridsDir, orbit_filename)
    _atomic_write_json(orbit_path, orbit_dict)
    _atomic_write_json(orbit_cache_metadata_path(orbit_path), metadata)


def generate_fractional_coordiantion(ions_on_side_list):
    step = np.zeros(3)
    num_frac_pos = 1
    for i in range(len(ions_on_side_list)):
        step[i] = 1.0 / ions_on_side_list[i]
        num_frac_pos = ions_on_side_list[i] * num_frac_pos

    frac_pos = np.zeros((num_frac_pos, 3))

    row = 0
    for i, j, k in np.ndindex(ions_on_side_list[0], ions_on_side_list[1], ions_on_side_list[2]):
        frac_pos[row,] = np.array([i * step[0], j * step[1], k * step[2]])
        row = row + 1
    return frac_pos, num_frac_pos


def _wrap_fractional(frac, tolerance=1e-8):
    wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=tolerance)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=tolerance)] = 0.0
    return wrapped


def _coord_key(frac, decimals=8):
    rounded = np.round(_wrap_fractional(frac), decimals=decimals)
    rounded[np.isclose(rounded, 1.0, atol=10 ** (-decimals))] = 0.0
    rounded[np.isclose(rounded, 0.0, atol=10 ** (-decimals))] = 0.0
    return tuple(float(value) for value in rounded)


def _build_fractional_index(frac_pos_list, decimals=8):
    wrapped_positions = np.array([_wrap_fractional(pos) for pos in frac_pos_list], dtype=float)
    index_by_key = {}
    for index, frac in enumerate(wrapped_positions):
        key = _coord_key(frac, decimals=decimals)
        if key in index_by_key:
            raise ValueError(
                f"duplicate fractional grid coordinate {key} at indices "
                f"{index_by_key[key]} and {index}"
            )
        index_by_key[key] = index
    return wrapped_positions, index_by_key


def _site_to_grid_index(site, wrapped_positions, index_by_key, decimals=8, tolerance=1e-6):
    key = _coord_key(site, decimals=decimals)
    mapped = index_by_key.get(key)
    if mapped is not None:
        return mapped

    wrapped = _wrap_fractional(site)
    delta = np.abs(wrapped_positions - wrapped)
    delta = np.minimum(delta, 1.0 - delta)
    distances = np.max(delta, axis=1)
    nearest = int(np.argmin(distances))
    nearest_distance = float(distances[nearest])
    if nearest_distance <= float(tolerance):
        tie_atol = max(np.finfo(float).eps * 16.0, float(tolerance) * 1e-9)
        tied = np.flatnonzero(
            (distances <= float(tolerance))
            & np.isclose(distances, nearest_distance, rtol=0.0, atol=tie_atol)
        )
        if len(tied) > 1:
            raise ValueError(
                "symmetry-equivalent site maps ambiguously to multiple grid points: "
                f"site={wrapped.tolist()} candidate_indices={tied.tolist()} "
                f"candidate_fracs={wrapped_positions[tied].tolist()} "
                f"max_frac_delta={nearest_distance}"
            )
        return nearest
    raise ValueError(
        "symmetry-equivalent site does not lie on the grid: "
        f"site={wrapped.tolist()} nearest_index={nearest} "
        f"nearest_frac={wrapped_positions[nearest].tolist()} "
        f"max_frac_delta={nearest_distance}"
    )


def _hexagonal_offset_scale(wrapped_positions, grid_size, tolerance=1e-8):
    grid = np.asarray(grid_size, dtype=int)
    if (
        grid.shape != (3,)
        or np.any(grid <= 0)
        or int(grid[0]) <= 1
        or int(grid[1]) <= 1
        or len(wrapped_positions) != int(np.prod(grid))
    ):
        return None

    offset_scale = float(grid[0]) / (2.0 * float(grid[1]))
    regular_positions = _to_hexagonal_regular(wrapped_positions, offset_scale)
    expected_keys = set()
    actual_keys = set()

    for i, j, k in np.ndindex(int(grid[0]), int(grid[1]), int(grid[2])):
        expected_keys.add(_coord_key([i / grid[0], j / grid[1], k / grid[2]]))

    for frac in regular_positions:
        ijk = np.rint(frac * grid).astype(int) % grid
        nearest = ijk / grid
        delta = np.abs(frac - nearest)
        delta = np.minimum(delta, 1.0 - delta)
        if float(np.max(delta)) > float(tolerance):
            return None
        actual_keys.add(_coord_key(frac))

    if actual_keys != expected_keys:
        return None
    return offset_scale


def _to_hexagonal_regular(frac, offset_scale):
    regular = _wrap_fractional(frac)
    regular[..., 1] = regular[..., 1] - float(offset_scale) * regular[..., 0]
    return _wrap_fractional(regular)


def _from_hexagonal_regular(frac, offset_scale):
    offset = _wrap_fractional(frac)
    offset[..., 1] = offset[..., 1] + float(offset_scale) * offset[..., 0]
    return _wrap_fractional(offset)


def _symmetry_input_position(frac, offset_scale):
    if offset_scale is None:
        return frac
    return _to_hexagonal_regular(frac, offset_scale)


def _symmetry_output_position(frac, offset_scale):
    if offset_scale is None:
        return frac
    return _from_hexagonal_regular(frac, offset_scale)


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def generate_one_orbit(target_space_group, frac_pos_list, frac_pos_index):
    wrapped_positions, index_by_key = _build_fractional_index(frac_pos_list)
    target_frac_pos = wrapped_positions[int(frac_pos_index)]
    sg = Spacegroup(int(target_space_group))
    sites, _ = sg.equivalent_sites(target_frac_pos)

    orbit_indices = {int(frac_pos_index)}
    for site in sites:
        orbit_indices.add(_site_to_grid_index(site, wrapped_positions, index_by_key))

    members = sorted(index for index in orbit_indices if index != int(frac_pos_index))
    return {str(frac_pos_index): members}, frac_pos_list, sites


def _singleton_orbit_dict(size):
    orbit_dict = {str(index): [] for index in range(size)}
    total_sites = [[index] for index in range(size)]
    return orbit_dict, total_sites


def generate_orbit_dict(target_space_group, frac_pos_list, grid_size, GridsDir,
                        lattice_matrix=None):
    target_space_group = int(target_space_group)
    frac_pos = np.asarray(frac_pos_list, dtype=float)
    wrapped_positions, index_by_key = _build_fractional_index(frac_pos)
    hex_offset_scale = _hexagonal_offset_scale(wrapped_positions, grid_size)
    union_find = _UnionFind(len(wrapped_positions))
    method = "ase_spacegroup"

    if target_space_group == 1:
        cubic_orbit_dict, cubic_total_sites = _singleton_orbit_dict(len(wrapped_positions))
        method = "p1_singletons"
    elif lattice_matrix is not None:
        # Use spglib with actual lattice matrix to get symmetry operations
        # only when the metric-derived space group is the requested target.
        # Otherwise the target SG must control the orbit construction.
        dummy_cell = (np.asarray(lattice_matrix, dtype=float),
                      np.array([[0.0, 0.0, 0.0]]), [5])
        dataset = spglib.get_symmetry_dataset(dummy_cell, symprec=1e-3)
        if dataset is not None and int(dataset.number) == target_space_group:
            sym = spglib.get_symmetry(dummy_cell, symprec=1e-3)
            rotations = np.asarray(sym["rotations"], dtype=int)
            translations = np.asarray(sym["translations"], dtype=float)
            method = "spglib_lattice_symmetry"

            for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
                symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
                for rotation, translation in zip(rotations, translations):
                    site = np.mod(np.dot(rotation, symmetry_frac_pos) + translation, 1.0)
                    site = _symmetry_output_position(site, hex_offset_scale)
                    mapped_index = _site_to_grid_index(
                        site=site,
                        wrapped_positions=wrapped_positions,
                        index_by_key=index_by_key,
                    )
                    union_find.union(frac_pos_index, mapped_index)
        else:
            sg = Spacegroup(target_space_group)
            for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
                symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
                sites, _ = sg.equivalent_sites(symmetry_frac_pos)
                for site in sites:
                    site = _symmetry_output_position(site, hex_offset_scale)
                    mapped_index = _site_to_grid_index(site, wrapped_positions, index_by_key)
                    union_find.union(frac_pos_index, mapped_index)
    else:
        sg = Spacegroup(target_space_group)
        for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
            symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
            sites, _ = sg.equivalent_sites(symmetry_frac_pos)
            for site in sites:
                site = _symmetry_output_position(site, hex_offset_scale)
                mapped_index = _site_to_grid_index(site, wrapped_positions, index_by_key)
                union_find.union(frac_pos_index, mapped_index)

    if target_space_group != 1:
        cubic_orbit_dict, cubic_total_sites = _component_orbit_dict(union_find, len(wrapped_positions))

    assert_valid_orbit_partition(
        orbits=cubic_orbit_dict,
        grid_size=grid_size,
        context=f"SG{target_space_group}_G{grid_size}",
    )

    orbit_filename = f'SG{target_space_group}_G{grid_size}.json'
    _write_orbit_cache(
        GridsDir=GridsDir,
        orbit_filename=orbit_filename,
        orbit_dict=cubic_orbit_dict,
        metadata=build_orbit_cache_metadata(
            target_space_group=target_space_group,
            grid_size=grid_size,
            lattice_matrix=lattice_matrix,
            method=method,
        ),
    )

    return cubic_orbit_dict, cubic_total_sites

def _component_orbit_dict(union_find, size):
    components = {}
    for index in range(size):
        root = union_find.find(index)
        components.setdefault(root, []).append(index)

    orbit_dict = {}
    total_sites = []
    for component in sorted(components.values(), key=lambda values: min(values)):
        orbit_indices = sorted(component)
        representative = orbit_indices[0]
        orbit_dict[str(representative)] = orbit_indices[1:]
        total_sites.append(orbit_indices)
    return orbit_dict, total_sites


def _regular_grid_index(site, grid_size, tolerance=1e-5):
    # Only valid for ordinary regular grids; offset-grid orbit generation must use fractional lookup.
    grid = np.asarray(grid_size, dtype=int)
    wrapped = _wrap_fractional(site)
    ijk = np.rint(wrapped * grid).astype(int) % grid
    nearest = ijk / grid
    delta = np.abs(wrapped - nearest)
    delta = np.minimum(delta, 1.0 - delta)
    max_delta = float(np.max(delta))
    if max_delta > float(tolerance):
        raise ValueError(
            "symmetry-equivalent site does not lie on the regular grid: "
            f"site={wrapped.tolist()} nearest_frac={nearest.tolist()} "
            f"grid={grid.tolist()} max_frac_delta={max_delta}"
        )
    return int(ijk[0]) * int(grid[1]) * int(grid[2]) + int(ijk[1]) * int(grid[2]) + int(ijk[2])


def generate_reference_orbit_dict(target_space_group, frac_pos_list, grid_size, GridsDir,
                                  reference_structure_file, orbit_filename=None,
                                  symprec=1e-3, angle_tolerance=5.0, tolerance=1e-5):
    frac_pos = np.asarray(frac_pos_list, dtype=float)
    wrapped_positions, index_by_key = _build_fractional_index(frac_pos)
    hex_offset_scale = _hexagonal_offset_scale(wrapped_positions, grid_size)

    reference_structure = Structure.from_file(reference_structure_file)
    reference_cell = (
        np.asarray(reference_structure.lattice.matrix, dtype=float),
        np.mod(np.asarray([site.frac_coords for site in reference_structure], dtype=float), 1.0),
        [int(site.specie.Z) for site in reference_structure],
    )
    dataset = spglib.get_symmetry_dataset(
        reference_cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if dataset is None:
        raise ValueError(f"could not determine symmetry for {reference_structure_file}")
    if int(dataset.number) != int(target_space_group):
        raise ValueError(
            f"reference symmetry mismatch for {reference_structure_file}: "
            f"expected SG{target_space_group}, got SG{int(dataset.number)}"
        )

    symmetry = spglib.get_symmetry(
        reference_cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    rotations = np.asarray(symmetry["rotations"], dtype=int)
    translations = np.asarray(symmetry["translations"], dtype=float)
    union_find = _UnionFind(len(wrapped_positions))

    for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
        symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
        for rotation, translation in zip(rotations, translations):
            site = np.mod(np.dot(rotation, symmetry_frac_pos) + translation, 1.0)
            site = _symmetry_output_position(site, hex_offset_scale)
            mapped_index = _site_to_grid_index(
                site=site,
                wrapped_positions=wrapped_positions,
                index_by_key=index_by_key,
                tolerance=float(tolerance),
            )
            union_find.union(frac_pos_index, mapped_index)

    orbit_dict, total_sites = _component_orbit_dict(union_find, len(wrapped_positions))
    context = f"SG{target_space_group}_G{grid_size}_ref_{os.path.basename(reference_structure_file)}"
    assert_valid_orbit_partition(orbits=orbit_dict, grid_size=grid_size, context=context)

    if orbit_filename is None:
        reference_stem = os.path.splitext(os.path.basename(reference_structure_file))[0]
        orbit_filename = f"SG{target_space_group}_G{grid_size}_ref_{reference_stem}.json"
    _write_orbit_cache(
        GridsDir=GridsDir,
        orbit_filename=orbit_filename,
        orbit_dict=orbit_dict,
        metadata=build_orbit_cache_metadata(
            target_space_group=target_space_group,
            grid_size=grid_size,
            reference_structure_file=reference_structure_file,
            method="spglib_reference",
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ),
    )

    return orbit_dict, total_sites


def visualize_orbit(orbit_dict, frac_pos, atom_num, lattice_matrix, conventional_cell, grid_size):
    symbols = ''
    i = 0
    for key in orbit_dict.keys():
        temp_symbols = str(Element.from_Z(i + 1)) * (len(orbit_dict[key]) + 1)
        symbols += temp_symbols
        i += 1
    print(symbols)

    temp_list = []
    for key, sublist in orbit_dict.items():
        temp_list.append(int(key))
        for i in sublist:
            temp_list.append(int(i))

    matrix = np.zeros((atom_num, 3))
    for i in range(len(temp_list)):
        matrix[i] = frac_pos[i]
    print(matrix)


def _orbit_size_histogram(orbit_dict):
    histogram = {}
    for members in orbit_dict.values():
        size = len(members) + 1
        histogram[size] = histogram.get(size, 0) + 1
    return dict(sorted(histogram.items()))


def run_debug_demo(grids_dir=None):
    """
    Hard-coded orbit-generation demo for local debugging.

    Run from the repository root with:
        python ip4ch/generate_crystal_orbit.py
    """
    grid_size = [3, 3, 3]
    lattice_matrix = np.diag([5.0, 6.0, 7.0])
    demo_cases = [
        ("p1_singletons", 1),
        ("sg47_orthorhombic_demo", 47),
    ]
    if grids_dir is None:
        grids_dir = os.path.join(os.getcwd(), "debug_orbit_demo_grids")

    frac_pos, num_grid_points = generate_fractional_coordiantion(ions_on_side_list=grid_size)
    cases = []
    for label, target_space_group in demo_cases:
        orbit_dict, total_sites = generate_orbit_dict(
            target_space_group=target_space_group,
            frac_pos_list=frac_pos,
            grid_size=grid_size,
            GridsDir=grids_dir,
            lattice_matrix=lattice_matrix,
        )
        orbit_filename = f"SG{target_space_group}_G{grid_size}.json"
        orbit_path = os.path.join(grids_dir, orbit_filename)
        cases.append({
            "label": label,
            "target_space_group": int(target_space_group),
            "orbit_count": len(orbit_dict),
            "orbit_size_histogram": _orbit_size_histogram(orbit_dict),
            "total_sites_counted": int(sum(len(site_group) for site_group in total_sites)),
            "orbit_file": orbit_path,
            "metadata_file": orbit_cache_metadata_path(orbit_path),
        })

    return {
        "demo": "generate_crystal_orbit_debug",
        "grid_size": grid_size,
        "lattice_matrix": lattice_matrix.tolist(),
        "num_grid_points": int(num_grid_points),
        "grids_dir": os.path.abspath(grids_dir),
        "cases": cases,
    }


def main():
    summary = run_debug_demo()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
