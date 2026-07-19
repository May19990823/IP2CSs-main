from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from b168_model_opt.pair_table import PairTable


def _validated_grid_size(grid_size: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        grid = tuple(grid_size)
    except TypeError as exc:
        raise ValueError("grid_size must contain exactly three positive integers") from exc
    if (
        len(grid) != 3
        or any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) for value in grid)
        or any(value <= 0 for value in grid)
    ):
        raise ValueError("grid_size must contain exactly three positive integers")
    return tuple(int(value) for value in grid)


def _validated_cell(cell: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("cell matrix must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("cell matrix must contain finite values")
    sigma_min = float(np.linalg.svd(matrix, compute_uv=False).min())
    if sigma_min <= np.finfo(np.float64).eps:
        raise ValueError("cell matrix must be nonsingular")
    return matrix


def _validated_cutoffs(r_min: float, r_max: float) -> tuple[float, float]:
    minimum = float(r_min)
    maximum = float(r_max)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("r_min must be finite and nonnegative")
    if not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("r_max must be finite and greater than r_min")
    return minimum, maximum


def _cross_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.array(
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    )


def _matrix_vector_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2]
            for row in matrix
        ]
    )


def _vector_length(vector: np.ndarray) -> float:
    return math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )


def _matscipy_bin_geometry(
    cell: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell1, cell2, cell3 = cell
    normal1 = _cross_product(cell2, cell3)
    normal2 = _cross_product(cell3, cell1)
    normal3 = _cross_product(cell1, cell2)
    volume = abs(
        cell3[0] * normal3[0] + cell3[1] * normal3[1] + cell3[2] * normal3[2]
    )
    face_distances = np.array(
        [
            volume / _vector_length(normal1),
            volume / _vector_length(normal2),
            volume / _vector_length(normal3),
        ]
    )
    bins = np.array(
        [max(int(math.floor(distance / cutoff)), 1) for distance in face_distances],
        dtype=np.int64,
    )
    search = np.array(
        [int(math.ceil(cutoff * count / distance)) for count, distance in zip(bins, face_distances)],
        dtype=np.int64,
    )
    return bins, search, cell / bins[:, None]


def _origin_neighbour_sequence(
    grid_size: tuple[int, int, int], cell: np.ndarray, cutoff: float
):
    """Yield matscipy's i=0 neighbour sequence without building a pair list."""
    grid_shape = _validated_grid_size(grid_size)
    matrix = _validated_cell(cell)
    _, maximum = _validated_cutoffs(0.0, cutoff)
    grid = np.asarray(grid_shape, dtype=np.float64)
    scaled_positions = np.asarray(tuple(np.ndindex(*grid_shape)), dtype=np.float64)
    scaled_positions /= grid
    positions = np.dot(scaled_positions, matrix)
    bins, search, bin_vectors = _matscipy_bin_geometry(matrix, maximum)
    inverse_cell = np.linalg.inv(matrix.T)
    bin_members = [[] for _ in range(int(np.prod(bins)))]
    cell_indices: list[np.ndarray] = []

    for index, position in enumerate(positions):
        fractional_position = _matrix_vector_product(inverse_cell, position)
        cell_index = np.floor(fractional_position * bins).astype(np.int64)
        cell_indices.append(cell_index)
        wrapped = cell_index % bins
        flattened = int(wrapped[0] + bins[0] * (wrapped[1] + bins[1] * wrapped[2]))
        bin_members[flattened].append(index)

    cutoff_squared = maximum * maximum
    for z_offset in range(-int(search[2]), int(search[2]) + 1):
        offset3 = z_offset * bin_vectors[2]
        for y_offset in range(-int(search[1]), int(search[1]) + 1):
            offset2 = offset3 + y_offset * bin_vectors[1]
            for x_offset in range(-int(search[0]), int(search[0]) + 1):
                wrapped = np.array([x_offset, y_offset, z_offset]) % bins
                flattened = int(
                    wrapped[0] + bins[0] * (wrapped[1] + bins[1] * wrapped[2])
                )
                offset = offset2 + x_offset * bin_vectors[0]
                for right in bin_members[flattened]:
                    if right == 0 and x_offset == y_offset == z_offset == 0:
                        continue
                    cell_index = cell_indices[right]
                    position = positions[right]
                    local = np.array(
                        [
                            position[0]
                            - cell_index[0] * bin_vectors[0, 0]
                            - cell_index[1] * bin_vectors[1, 0]
                            - cell_index[2] * bin_vectors[2, 0],
                            position[1]
                            - cell_index[0] * bin_vectors[0, 1]
                            - cell_index[1] * bin_vectors[1, 1]
                            - cell_index[2] * bin_vectors[2, 1],
                            position[2]
                            - cell_index[0] * bin_vectors[0, 2]
                            - cell_index[1] * bin_vectors[1, 2]
                            - cell_index[2] * bin_vectors[2, 2],
                        ]
                    )
                    vector = local + offset
                    distance_squared = (
                        vector[0] * vector[0]
                        + vector[1] * vector[1]
                        + vector[2] * vector[2]
                    )
                    if distance_squared < cutoff_squared:
                        yield right, vector, math.sqrt(distance_squared)


def build_periodic_kernel(
    grid_size: tuple[int, int, int],
    cell: np.ndarray,
    r_min: float,
    r_max: float,
    table: PairTable,
) -> np.ndarray:
    """Build the ordered-site displacement kernel used by the legacy alpha matrix."""
    grid_shape = _validated_grid_size(grid_size)
    matrix = _validated_cell(cell)
    minimum, maximum = _validated_cutoffs(r_min, r_max)
    if not isinstance(table, PairTable):
        raise TypeError("table must be a PairTable")

    totals = np.zeros(int(np.prod(grid_shape)), dtype=np.float32)

    for right, _, distance in _origin_neighbour_sequence(
        grid_shape, matrix, maximum
    ):
        if right == 0 or distance < minimum - 1e-8:
            continue
        energy = table.energy(float(distance))
        if energy is not None:
            totals[right] = np.float32(totals[right] + energy)

    return totals.astype(np.float64).reshape(grid_shape)
