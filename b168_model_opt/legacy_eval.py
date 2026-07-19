from __future__ import annotations

import numpy as np

from b168_model_opt.types import OrbitData


def _validated_kernel(orbit_data: OrbitData, kernel: np.ndarray) -> np.ndarray:
    if not isinstance(kernel, np.ndarray):
        raise TypeError("kernel must be a numpy ndarray")
    if kernel.shape != orbit_data.grid_size:
        raise ValueError("kernel shape must equal orbit_data.grid_size")
    if kernel.dtype.kind != "f":
        raise ValueError("kernel must have a floating-point dtype")
    if not np.isfinite(kernel).all():
        raise ValueError("kernel must contain only finite values")
    return kernel


def _validated_assignment(assignment: np.ndarray, active_count: int) -> np.ndarray:
    values = np.asarray(assignment)
    if values.ndim != 1 or len(values) != active_count:
        raise ValueError("assignment must be a vector with one value per active orbit")
    if values.dtype.kind not in "biu":
        raise ValueError("assignment must have a boolean or integer dtype")
    if not np.logical_or(values == 0, values == 1).all():
        raise ValueError("assignment must contain only binary values")
    return values


def _coordinates(grid_size: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(list(np.ndindex(*grid_size)), dtype=np.int64)


def evaluate_site_objective(
    assignment: np.ndarray,
    orbit_data: OrbitData,
    kernel: np.ndarray,
) -> float:
    """Evaluate the production i < j site-pair objective without a solver."""
    values = _validated_assignment(assignment, len(orbit_data.active_original))
    displacement_kernel = _validated_kernel(orbit_data, kernel)
    occupied = np.asarray(
        [active >= 0 and values[active] == 1 for active in orbit_data.site_to_active],
        dtype=bool,
    )
    selected_sites = np.flatnonzero(occupied)
    coordinates = _coordinates(orbit_data.grid_size)
    grid = np.asarray(orbit_data.grid_size, dtype=np.int64)
    total = 0.0
    for offset, left in enumerate(selected_sites):
        for right in selected_sites[offset + 1 :]:
            displacement = tuple(((coordinates[right] - coordinates[left]) % grid).tolist())
            total += float(displacement_kernel[displacement])
    return total
