from ase.atoms import Atoms
import csv
import hashlib
import json
import numpy as np
import os
import tempfile
from pathlib import Path
from matscipy.neighbours import neighbour_list


def generate_orthorhombic(ions_on_sides):
    """
    Ions_on_side=4 will generate points with x coordinates 0, 0.25, 0.5 and 0.75.
    Note that ip4ch is kind of another cell already. So, side/ions is the step size
    There will be prod(ions_on_side) points in total in the cell.
    Check separately whether you get charge NEUTRAL system!
    """

    step = 1.0 / np.array(ions_on_sides)

    pos = np.zeros((ions_on_sides[0] * ions_on_sides[1] * ions_on_sides[2], 3))
    # print("The total number of points in the cell is ", len(self.ions))

    row = 0
    for (i, j, k) in np.ndindex(ions_on_sides[0], ions_on_sides[1], ions_on_sides[2]):
        pos[row,] = np.array([i * step[0], j * step[1], k * step[2]])
        row = row + 1
    return pos


def generate_hexagonal(ions_on_sides):
    """
    Generate fractional coordinates for a hexagonal lattice.

    Args:
        ions_on_sides (list or tuple): Number of divisions along a, b, and c axes.

    Returns:
        np.ndarray: Fractional coordinates of ions within the hexagonal lattice.
    """
    step_a = 1.0 / ions_on_sides[0]  # Step size along a-axis
    step_b = 1.0 / ions_on_sides[1]  # Step size along b-axis
    step_c = 1.0 / ions_on_sides[2]  # Step size along c-axis

    # Total number of points in the cell
    pos = np.zeros((ions_on_sides[0] * ions_on_sides[1] * ions_on_sides[2], 3))

    # Iterate over all indices in the hexagonal grid
    row = 0
    for i, j, k in np.ndindex(ions_on_sides[0], ions_on_sides[1], ions_on_sides[2]):
        # Convert grid indices to fractional coordinates
        x = i * step_a
        y = (j * step_b + i * step_b / 2) % 1.0  # Offset y due to hexagonal geometry
        z = k * step_c
        pos[row] = np.array([x, y, z])
        row += 1

    return pos


def cubic(ions_on_side):
    """
    generate points for the cubic grid
    """
    if isinstance(ions_on_side, list):
        return generate_orthorhombic([ions_on_side[0], ions_on_side[1], ions_on_side[2]])
    if isinstance(ions_on_side, int):
        return generate_orthorhombic([ions_on_side, ions_on_side, ions_on_side])


def _infer_distance_precision(pair_table):
    keys = sorted({round(float(key), 8) for key in pair_table.keys()})
    if len(keys) < 2:
        return 1
    diffs = [right - left for left, right in zip(keys, keys[1:]) if right - left > 1e-8]
    if not diffs:
        return 1
    step = min(diffs)
    for decimals in range(5):
        if abs(step - round(step, decimals)) < 1e-8:
            return decimals
    return 4


def _normalize_pair_table(pair_table):
    precision = _infer_distance_precision(pair_table)
    return {round(float(distance), precision): float(energy) for distance, energy in pair_table.items()}


def _load_csv_pair_table(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"CSV pair table has no header: {path}")
        distance_column = "r_ij" if "r_ij" in fieldnames else fieldnames[0]
        if "E_ij" in fieldnames:
            energy_column = "E_ij"
        elif "median_smoothed" in fieldnames:
            energy_column = "median_smoothed"
        elif "median_effective_pair" in fieldnames:
            energy_column = "median_effective_pair"
        elif "q10_smoothed" in fieldnames:
            energy_column = "q10_smoothed"
        elif "q10_interpolated" in fieldnames:
            energy_column = "q10_interpolated"
        elif "energy" in fieldnames:
            energy_column = "energy"
        else:
            energy_column = fieldnames[1]

        pair_table = {}
        for row in reader:
            if not row.get(distance_column) or not row.get(energy_column):
                continue
            pair_table[float(row[distance_column])] = float(row[energy_column])
    if not pair_table:
        raise ValueError(f"CSV pair table has no usable rows: {path}")
    return _normalize_pair_table(pair_table)


def _canonical_pair_file_stems(t1, t2):
    ordered = sorted([str(t1), str(t2)])
    canonical = f"{ordered[0]}-{ordered[1]}"
    direct = f"{t1}-{t2}"
    reverse = f"{t2}-{t1}"
    stems = [canonical, direct, reverse]
    return list(dict.fromkeys(stems))


def _resolve_pair_file_for_atoms(t1, t2, pair_file=None, pair_dir=None):
    if pair_file is None and pair_dir is None:
        raise ValueError("pair_file or pair_dir is required for alpha matrix construction")

    if pair_dir is not None:
        configured_dir = Path(os.path.expandvars(str(pair_dir))).expanduser()
        suffix = ".csv"
    else:
        configured = Path(os.path.expandvars(str(pair_file))).expanduser()
        configured_dir = configured if configured.is_dir() else configured.parent
        suffix = configured.suffix.lower() if configured.suffix else ".csv"

    for stem in _canonical_pair_file_stems(t1, t2):
        candidate = configured_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    if pair_dir is None and configured.is_file():
        configured_stem = configured.stem
        valid_stems = _canonical_pair_file_stems(t1, t2)
        if (
            "-" not in configured_stem
            or any(
                configured_stem == stem or configured_stem.startswith(f"{stem}_")
                for stem in valid_stems
            )
        ):
            return configured

    expected = ", ".join(
        str(configured_dir / f"{stem}{suffix}") for stem in _canonical_pair_file_stems(t1, t2)
    )
    raise FileNotFoundError(
        f"No pair file found for atom pair {t1}-{t2}. Expected one of: {expected}"
    )


def _load_pair_file(pair_file):
    if pair_file is None:
        raise ValueError("pair_file is required for alpha matrix construction")

    source = Path(os.path.expandvars(str(pair_file))).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Required pair_file does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return _load_csv_pair_table(source)
    if suffix == ".npy":
        with source.open("rb") as handle:
            return _normalize_pair_table(np.load(handle, allow_pickle=True).item())
    raise ValueError(f"Unsupported pair_file extension for {source}; expected .csv or .npy")


def _cell_signature(lattice_matrix):
    rounded = np.round(np.asarray(lattice_matrix, dtype=float), 8)
    payload = json.dumps(rounded.tolist(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _pair_file_signature(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_cache_path(t1, t2, phase_name, pair_type, DataPath, lattice_matrix, r_min, r_max, grid_size,
                     pair_signature=None):
    pair_type = str(pair_type)
    grid_token = "x".join(str(int(value)) for value in grid_size)
    r_token = f"rmin{float(r_min):.2f}_rmax{float(r_max):.2f}".replace(".", "p")
    cell_token = _cell_signature(lattice_matrix)
    pair_token = f"_pair{str(pair_signature)[:12]}" if pair_signature else ""
    filename = f"{t1}_{t2}_G{grid_token}_{r_token}_hardmin_{cell_token}{pair_token}.npy"
    return Path(DataPath) / phase_name / "alpha_cache" / pair_type / filename


def _alpha_cache_metadata(t1, t2, phase_name, pair_type, lattice_matrix, r_min, r_max, grid_size,
                          pair_file=None, pair_signature=None):
    return {
        "t1": t1,
        "t2": t2,
        "phase_name": phase_name,
        "pair_type": str(pair_type),
        "grid_size": [int(value) for value in grid_size],
        "r_min": float(r_min),
        "r_max": float(r_max),
        "min_distance_mode": "hard_constraint_no_objective_penalty",
        "lattice_matrix": np.asarray(lattice_matrix, dtype=float).tolist(),
        "cell_signature": _cell_signature(lattice_matrix),
        "pair_file": None if pair_file is None else str(pair_file),
        "pair_sha256": pair_signature,
    }


def _load_alpha_cache(cache_path, expected_shape):
    if not Path(cache_path).is_file():
        return None
    matrix = np.load(cache_path)
    if tuple(matrix.shape) != tuple(expected_shape):
        return None
    return matrix


def _save_alpha_cache(cache_path, alpha_matrix, metadata):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, prefix=f".{cache_path.name}.", suffix=".tmp",
            delete=False) as handle:
        temp_matrix = Path(handle.name)
        np.save(handle, alpha_matrix)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_matrix, cache_path)

    metadata_path = cache_path.with_suffix(".json")
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.", suffix=".tmp", delete=False) as handle:
        temp_metadata = Path(handle.name)
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_metadata, metadata_path)


def get_energy_table(t1, t2, phase_name, lattice_matrix, r_min, r_max, grid_size, frac_pos,
                     pair_type, DataPath, pair_file=None, pair_dir=None):
    # 计算总点数
    N = np.prod(grid_size)

    # 创建临时结构
    tmp_struc = Atoms(symbols='H' * N, scaled_positions=frac_pos, pbc=True, cell=lattice_matrix)

    resolved_pair_file = _resolve_pair_file_for_atoms(t1, t2, pair_file=pair_file, pair_dir=pair_dir)
    alpha_dict = _load_pair_file(resolved_pair_file)

    distance_precision = _infer_distance_precision(alpha_dict)
    min_distance = float(r_min)

    # 初始化 Alpha 矩阵
    alpha_matrix = np.zeros((N, N), dtype=np.float32)  # 使用 float32 减少内存占用

    available_distances = set(alpha_dict.keys())

    # 计算邻近列表
    i, j, d, offset = neighbour_list(quantities='ijdD', atoms=tmp_struc, cutoff=r_max)

    # 填充 Alpha 矩阵
    for i0, j0, d0 in zip(i, j, d):
        if i0 == j0:
            continue

        d_rounded = round(float(d0), distance_precision)
        if float(d0) < min_distance - 1e-8:
            continue
        if d_rounded in available_distances:
            alpha_matrix[i0, j0] += alpha_dict[d_rounded]

    """# 保存 Alpha 矩阵
    os.makedirs(os.path.dirname(alpha_matrix_filepath), exist_ok=True)
    with open(alpha_matrix_filepath, 'wb') as outfile:
        np.save(outfile, alpha_matrix)"""

    # 使用布尔索引找到非零元素，并将其减去 ip4ch
    # alpha_matrix[alpha_matrix != 0] -= ip4ch
    return alpha_matrix


def get_energy(t1, t2, phase_name, lattice, lattice_matrix, r_min, r_max, grid_size, frac_pos, pair_type, DataPath,
               compound_type=None, pair_file=None, pair_dir=None):
    # 设置文件路径

    # alpha_matrix_folder = os.path.join(PhaseDir, pair_type, 'Cell_Grid')
    # alpha_matrix_filename = f'{t1}_{t2}_{tuple(map(round, lattice))}_G{grid_size}.npy'
    # alpha_matrix_filepath = os.path.join(alpha_matrix_folder, alpha_matrix_filename)
    """

    # 如果文件存在，直接加载
    if os.path.exists(alpha_matrix_filepath):
        with open(alpha_matrix_filepath, 'rb') as f:
            return np.load(f)
    """

    resolved_pair_file = _resolve_pair_file_for_atoms(
        t1, t2, pair_file=pair_file, pair_dir=pair_dir
    )
    pair_signature = _pair_file_signature(resolved_pair_file)
    expected_shape = (int(np.prod(grid_size)), int(np.prod(grid_size)))
    cache_path = alpha_cache_path(
        t1=t1, t2=t2, phase_name=phase_name, pair_type=pair_type, DataPath=DataPath,
        lattice_matrix=lattice_matrix, r_min=r_min, r_max=r_max, grid_size=grid_size,
        pair_signature=pair_signature,
    )
    cached = _load_alpha_cache(cache_path, expected_shape)
    if cached is not None:
        print(f"Loaded alpha matrix cache: {cache_path}", flush=True)
        return cached

    alpha_matrix = get_energy_table(
        t1=t1, t2=t2, phase_name=phase_name, lattice_matrix=lattice_matrix,
        r_min=r_min, r_max=r_max, grid_size=grid_size, frac_pos=frac_pos,
        pair_type=pair_type, DataPath=DataPath, pair_file=pair_file, pair_dir=pair_dir
    )
    _save_alpha_cache(
        cache_path,
        alpha_matrix,
        _alpha_cache_metadata(
            t1=t1, t2=t2, phase_name=phase_name, pair_type=pair_type,
            lattice_matrix=lattice_matrix, r_min=r_min, r_max=r_max, grid_size=grid_size,
            pair_file=resolved_pair_file, pair_signature=pair_signature,
        ),
    )
    print(f"Saved alpha matrix cache: {cache_path}", flush=True)
    return alpha_matrix


if __name__ == '__main__':
    print('')
