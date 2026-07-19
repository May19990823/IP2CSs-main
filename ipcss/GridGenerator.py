import numpy as np


def generate_grid(lattice_matrix, r_min, safety_factor=2.0, max_points=5000,
                  min_grid_per_axis=1, max_grid_axis_ratio=None,
                  return_diagnostics=False):
    """
    Generate physically-adaptive grid dimensions for a lattice.

    For each lattice basis vector a_i, ensure adjacent grid points
    are spaced at most r_min / safety_factor apart in real space.

    Returns:
        (grid_size: [n1, n2, n3], resolution: float, status: str)
        If return_diagnostics is True, appends a diagnostics dictionary.
    """
    if float(r_min) <= 0:
        raise ValueError("r_min must be positive")
    if float(safety_factor) <= 0:
        raise ValueError("safety_factor must be positive")
    if max_points is not None and int(max_points) <= 0:
        raise ValueError("max_points must be positive")
    if not isinstance(min_grid_per_axis, (int, np.integer)) or min_grid_per_axis <= 0:
        raise ValueError("min_grid_per_axis must be a positive integer")

    lattice = np.array(lattice_matrix, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_matrix must be a finite 3x3 matrix")

    lengths = np.array([np.linalg.norm(lattice[i]) for i in range(3)])
    target_spacing = float(r_min) / float(safety_factor)
    n_raw = np.ceil(lengths / target_spacing).astype(int)
    n_raw = np.maximum(n_raw, min_grid_per_axis)

    total_raw = int(np.prod(n_raw))
    grid_size = [int(n) for n in n_raw]
    status = "good"

    if max_points is not None and total_raw > int(max_points):
        scale = (float(max_points) / float(total_raw)) ** (1.0 / 3.0)
        grid_size = [max(min_grid_per_axis, int(np.ceil(n * scale))) for n in n_raw]
        status = "truncated"

        while int(np.prod(grid_size)) > int(max_points):
            reducible = [
                idx for idx, value in enumerate(grid_size)
                if value > min_grid_per_axis
            ]
            if not reducible:
                status = "too_many_points"
                break
            idx = max(reducible, key=lambda i: grid_size[i])
            grid_size[idx] -= 1

    grid_axis_ratio = max(grid_size) / max(1, min(grid_size))
    if max_grid_axis_ratio is not None and grid_axis_ratio > max_grid_axis_ratio:
        status = "anisotropic"

    actual_spacings = [lengths[i] / grid_size[i] for i in range(3)]
    resolution = max(actual_spacings)
    grid_points = int(np.prod(grid_size))
    diagnostics = {
        "raw_grid_size": [int(n) for n in n_raw],
        "target_spacing": float(target_spacing),
        "actual_spacings": [float(spacing) for spacing in actual_spacings],
        "min_spacing": float(min(actual_spacings)),
        "max_spacing": float(max(actual_spacings)),
        "grid_points": grid_points,
        "grid_axis_ratio": float(grid_axis_ratio),
        "truncation_ratio": float(grid_points / total_raw) if total_raw else 1.0,
        "status": status,
    }

    result = (grid_size, round(resolution, 4), status)
    if return_diagnostics:
        return result + (diagnostics,)
    return result
