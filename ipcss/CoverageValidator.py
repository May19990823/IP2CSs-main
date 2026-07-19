import numpy as np


def validate_coverage(lattice_matrix, grid_size, r_min, r_max,
                      min_grid_per_axis=1, max_grid_axis_ratio=None,
                      max_rdf_resolution_ratio=None,
                      max_spacing_anisotropy=None):
    """
    Evaluate discretization quality for a (lattice, grid) pair.

    Returns:
        {rdf_resolution_ratio, coverage_estimate, grid_points, status}
    """
    if float(r_min) <= 0:
        raise ValueError("r_min must be positive")

    lattice = np.array(lattice_matrix, dtype=float)
    grid = np.array(grid_size, dtype=int)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_matrix must be a finite 3x3 matrix")
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError("grid_size must contain three positive integers")

    lengths = np.array([np.linalg.norm(lattice[i]) for i in range(3)])
    spacings = lengths / grid
    avg_spacing = float(np.mean(spacings))
    max_spacing = float(np.max(spacings))
    min_spacing = float(np.min(spacings))

    rdf_ratio = avg_spacing / r_min
    max_rdf_ratio = max_spacing / r_min
    grid_axis_ratio = float(np.max(grid) / max(1, np.min(grid)))
    spacing_anisotropy = max_spacing / max(min_spacing, 1e-12)

    passes_min_grid_axis = int(np.min(grid)) >= int(min_grid_per_axis)
    passes_grid_axis_ratio = (
        max_grid_axis_ratio is None
        or grid_axis_ratio <= float(max_grid_axis_ratio)
    )
    passes_max_spacing = (
        max_rdf_resolution_ratio is None
        or max_rdf_ratio <= float(max_rdf_resolution_ratio)
    )
    passes_spacing_anisotropy = (
        max_spacing_anisotropy is None
        or spacing_anisotropy <= float(max_spacing_anisotropy)
    )

    if not all([
        passes_min_grid_axis,
        passes_grid_axis_ratio,
        passes_max_spacing,
        passes_spacing_anisotropy,
    ]):
        status = "poor"
    elif max_rdf_ratio <= 0.5:
        status = "good"
    elif max_rdf_ratio < 1.0:
        status = "warning"
    else:
        status = "poor"

    coverage = np.exp(-2.0 * rdf_ratio) + 0.05 * np.exp(-rdf_ratio)
    coverage = min(1.0, max(0.0, coverage))

    return {
        "rdf_resolution_ratio": round(rdf_ratio, 4),
        "max_rdf_resolution_ratio": round(max_rdf_ratio, 4),
        "coverage_estimate": round(coverage, 4),
        "grid_points": int(np.prod(grid_size)),
        "min_grid_axis": int(np.min(grid)),
        "avg_spacing": avg_spacing,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "grid_axis_ratio": round(grid_axis_ratio, 4),
        "spacing_anisotropy": round(spacing_anisotropy, 4),
        "passes_min_grid_axis": bool(passes_min_grid_axis),
        "passes_grid_axis_ratio": bool(passes_grid_axis_ratio),
        "passes_max_spacing": bool(passes_max_spacing),
        "passes_spacing_anisotropy": bool(passes_spacing_anisotropy),
        "status": status,
    }
