from fractions import Fraction
import math

import numpy as np
from ase.spacegroup import Spacegroup
import spglib

from ip4ch.Generate_alpha_coefficients import generate_hexagonal, generate_orthorhombic
from ip4ch.generate_crystal_orbit import (
    _build_fractional_index,
    _coord_key,
    _hexagonal_offset_scale,
    _symmetry_input_position,
    _symmetry_output_position,
    _wrap_fractional,
)


DEFAULT_WARNING_GRID_POINTS = 5000
DEFAULT_HARD_GRID_POINTS = 12000
FAST_VALIDATION_METHOD = "fast_spacegroup_operation_closure"
VALIDATION_MODES = ("auto", "fast", "exhaustive")


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _grid_points(grid_size):
    total = 1
    for value in grid_size:
        total *= int(value)
    return int(total)


def _normalize_validation_mode(validation_mode):
    mode = str(validation_mode or "auto").strip().lower()
    if mode not in VALIDATION_MODES:
        raise ValueError("validation_mode must be one of %s" % ", ".join(VALIDATION_MODES))
    return mode


def _uses_regular_fractional_grid(lattice_type):
    return str(lattice_type) in (
        "Triclinic",
        "Monoclinic",
        "Orthorhombic",
        "Tetragonal",
        "Rhombohedral",
        "Hexagonal",
        "Cubic",
    )


def _gcd(a, b):
    return math.gcd(int(a), int(b))


def _lcm(a, b):
    a = int(a)
    b = int(b)
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _gcd(a, b)


def _find(parent, axis):
    axis = int(axis)
    while parent[axis] != axis:
        parent[axis] = parent[parent[axis]]
        axis = parent[axis]
    return axis


def _union(parent, left, right):
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _axis_ratio(grid_size):
    values = [int(value) for value in grid_size]
    return float(max(values) / max(1, min(values)))


def _frac_positions(grid_size, lattice_type):
    grid_size = [int(value) for value in grid_size]
    if lattice_type == "Hexagonal":
        return generate_hexagonal(grid_size)
    return generate_orthorhombic(grid_size)


def _nearest_grid_failure(site, wrapped_positions, index_by_key, decimals=8, tolerance=1e-6):
    key = _coord_key(site, decimals=decimals)
    mapped = index_by_key.get(key)
    if mapped is not None:
        return None

    wrapped = _wrap_fractional(site)
    delta = np.abs(wrapped_positions - wrapped)
    delta = np.minimum(delta, 1.0 - delta)
    distances = np.max(delta, axis=1)
    nearest = int(np.argmin(distances))
    nearest_distance = float(distances[nearest])
    if nearest_distance <= float(tolerance):
        return None

    return {
        "failed_site": wrapped.tolist(),
        "nearest_index": nearest,
        "nearest_frac": wrapped_positions[nearest].tolist(),
        "max_frac_delta": nearest_distance,
    }


def _spglib_operations_if_target_matches(space_group, lattice_matrix):
    if lattice_matrix is None:
        return None
    dummy_cell = (
        np.asarray(lattice_matrix, dtype=float),
        np.array([[0.0, 0.0, 0.0]]),
        [5],
    )
    dataset = spglib.get_symmetry_dataset(dummy_cell, symprec=1e-3)
    if dataset is None or int(dataset.number) != int(space_group):
        return None
    sym = spglib.get_symmetry(dummy_cell, symprec=1e-3)
    return (
        np.asarray(sym["rotations"], dtype=int),
        np.asarray(sym["translations"], dtype=float),
        "spglib_lattice_symmetry",
    )


def _spacegroup_full_operations(space_group):
    sg = Spacegroup(int(space_group))
    rotations, translations = sg.get_op()
    return (
        np.asarray(rotations, dtype=float),
        np.asarray(translations, dtype=float),
    )


def _translation_denominator(value, max_denominator=24):
    wrapped = float(value) % 1.0
    if abs(wrapped - 1.0) < 1e-8 or abs(wrapped) < 1e-8:
        return 1
    frac = Fraction(wrapped).limit_denominator(int(max_denominator))
    if abs(float(frac) - wrapped) > 1e-6:
        return 1
    return max(1, int(frac.denominator))


def _fast_grid_suggestion_from_operations(space_group, grid_size, max_translation_denominator=24):
    grid = [int(value) for value in grid_size]
    rotations, translations = _spacegroup_full_operations(space_group)
    parent = [0, 1, 2]
    translation_denominators = [1, 1, 1]
    unsupported = []

    for op_index, (rotation, translation) in enumerate(zip(rotations, translations)):
        rotation = np.asarray(rotation, dtype=float)
        for out_axis in range(3):
            for in_axis in range(3):
                coeff = float(rotation[out_axis, in_axis])
                coeff_int = int(round(coeff))
                if abs(coeff - coeff_int) > 1e-8:
                    unsupported.append({
                        "operation_index": int(op_index),
                        "axis": [int(out_axis), int(in_axis)],
                        "coefficient": coeff,
                    })
                    continue
                if coeff_int != 0:
                    _union(parent, out_axis, in_axis)

            denominator = _translation_denominator(
                translation[out_axis],
                max_denominator=max_translation_denominator,
            )
            translation_denominators[out_axis] = _lcm(
                translation_denominators[out_axis],
                denominator,
            )

    if unsupported:
        return {
            "supported": False,
            "unsupported_rotations": unsupported,
        }

    components = {}
    for axis in range(3):
        root = _find(parent, axis)
        components.setdefault(root, []).append(axis)

    suggested = list(grid)
    component_payload = []
    for axes in components.values():
        required_multiple = 1
        for axis in axes:
            required_multiple = _lcm(required_multiple, translation_denominators[axis])
        target = max(grid[axis] for axis in axes)
        if required_multiple > 1:
            target = _nearest_multiple_at_least(target, required_multiple)
        for axis in axes:
            suggested[axis] = target
        component_payload.append({
            "axes": [int(axis) for axis in axes],
            "required_multiple": int(required_multiple),
            "target_grid": int(target),
        })

    return {
        "supported": True,
        "suggested_grid": [int(value) for value in suggested],
        "axis_components": sorted(component_payload, key=lambda item: item["axes"]),
        "translation_denominators": [int(value) for value in translation_denominators],
        "operation_count": int(len(rotations)),
    }


def _failure_with_limits(result, warning_grid_points, hard_grid_points, max_grid_axis_ratio):
    grid = result["grid"]
    points = _grid_points(grid)
    warnings = list(result.get("warnings", []))
    if warning_grid_points is not None and points >= int(warning_grid_points):
        warnings.append("large_grid_warning")
    if max_grid_axis_ratio is not None and _axis_ratio(grid) > float(max_grid_axis_ratio):
        warnings.append("grid_axis_ratio_warning")
    result["grid_points"] = points
    result["grid_axis_ratio"] = _axis_ratio(grid)
    result["warnings"] = sorted(set(warnings))

    if hard_grid_points is not None and points > int(hard_grid_points):
        result.update({
            "compatible": False,
            "status": "rejected",
            "reason": "grid_points_exceed_hard_limit",
        })
    if (
        max_grid_axis_ratio is not None
        and _axis_ratio(grid) > float(max_grid_axis_ratio)
    ):
        result.update({
            "compatible": False,
            "status": "rejected",
            "reason": "grid_axis_ratio_exceeds_limit",
        })
    return result


def evaluate_fast_grid_space_group_compatibility(
    space_group,
    grid_size,
    lattice_type,
    warning_grid_points=DEFAULT_WARNING_GRID_POINTS,
    hard_grid_points=None,
    max_grid_axis_ratio=None,
):
    """Check regular fractional grids by operation-closure divisibility."""
    grid = [int(value) for value in grid_size]
    base = {
        "compatible": True,
        "status": "compatible",
        "reason": None,
        "space_group": int(space_group),
        "grid": grid,
        "original_grid": grid,
        "lattice_type": str(lattice_type),
        "adjusted": False,
        "method": FAST_VALIDATION_METHOD,
    }

    if not _uses_regular_fractional_grid(lattice_type):
        result = dict(base)
        result.update({
            "compatible": False,
            "status": "unsupported",
            "reason": "fast_validation_unsupported_for_lattice_type",
        })
        return _json_safe(_failure_with_limits(
            result,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    if int(space_group) == 1:
        return _json_safe(_failure_with_limits(
            base,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    suggestion = _fast_grid_suggestion_from_operations(space_group, grid)
    if not suggestion.get("supported"):
        result = dict(base)
        result.update({
            "compatible": False,
            "status": "unsupported",
            "reason": "fast_validation_unsupported_rotation",
            "unsupported_rotations": suggestion.get("unsupported_rotations", []),
        })
        return _json_safe(_failure_with_limits(
            result,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    base.update({
        "operation_count": suggestion["operation_count"],
        "axis_components": suggestion["axis_components"],
        "translation_denominators": suggestion["translation_denominators"],
    })
    suggested = suggestion["suggested_grid"]
    if suggested != grid:
        result = dict(base)
        result.update({
            "compatible": False,
            "status": "incompatible",
            "reason": "grid_space_group_incompatible",
            "suggested_grid": suggested,
            "suggested_grid_points": _grid_points(suggested),
            "suggested_grid_axis_ratio": _axis_ratio(suggested),
        })
        return _json_safe(_failure_with_limits(
            result,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    return _json_safe(_failure_with_limits(
        base,
        warning_grid_points=warning_grid_points,
        hard_grid_points=hard_grid_points,
        max_grid_axis_ratio=max_grid_axis_ratio,
    ))


def evaluate_grid_space_group_compatibility(
    space_group,
    grid_size,
    lattice_type,
    lattice_matrix=None,
    warning_grid_points=DEFAULT_WARNING_GRID_POINTS,
    hard_grid_points=None,
    max_grid_axis_ratio=None,
    tolerance=1e-6,
    validation_mode="exhaustive",
):
    """Check whether a grid is closed under the requested space group."""
    mode = _normalize_validation_mode(validation_mode)
    if mode == "fast" or (
        mode == "auto" and _uses_regular_fractional_grid(lattice_type)
    ):
        return evaluate_fast_grid_space_group_compatibility(
            space_group=space_group,
            grid_size=grid_size,
            lattice_type=lattice_type,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        )

    grid = [int(value) for value in grid_size]
    frac_pos = _frac_positions(grid, lattice_type)
    wrapped_positions, index_by_key = _build_fractional_index(frac_pos)
    hex_offset_scale = _hexagonal_offset_scale(wrapped_positions, grid)
    operations = _spglib_operations_if_target_matches(space_group, lattice_matrix)

    base = {
        "compatible": True,
        "status": "compatible",
        "reason": None,
        "space_group": int(space_group),
        "grid": grid,
        "original_grid": grid,
        "lattice_type": str(lattice_type),
        "adjusted": False,
        "method": "ase_spacegroup",
    }

    def fail(payload):
        result = dict(base)
        result.update({
            "compatible": False,
            "status": "incompatible",
            "reason": "grid_space_group_incompatible",
        })
        result.update(payload)
        return _json_safe(_failure_with_limits(
            result,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    if int(space_group) == 1:
        return _json_safe(_failure_with_limits(
            base,
            warning_grid_points=warning_grid_points,
            hard_grid_points=hard_grid_points,
            max_grid_axis_ratio=max_grid_axis_ratio,
        ))

    if operations is not None:
        rotations, translations, method = operations
        base["method"] = method
        for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
            symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
            for rotation, translation in zip(rotations, translations):
                site = np.mod(np.dot(rotation, symmetry_frac_pos) + translation, 1.0)
                site = _symmetry_output_position(site, hex_offset_scale)
                failure = _nearest_grid_failure(
                    site,
                    wrapped_positions,
                    index_by_key,
                    tolerance=tolerance,
                )
                if failure is not None:
                    failure["source_grid_index"] = int(frac_pos_index)
                    return fail(failure)
    else:
        sg = Spacegroup(int(space_group))
        for frac_pos_index, target_frac_pos in enumerate(wrapped_positions):
            symmetry_frac_pos = _symmetry_input_position(target_frac_pos, hex_offset_scale)
            sites, _ = sg.equivalent_sites(symmetry_frac_pos)
            for site in sites:
                site = _symmetry_output_position(site, hex_offset_scale)
                failure = _nearest_grid_failure(
                    site,
                    wrapped_positions,
                    index_by_key,
                    tolerance=tolerance,
                )
                if failure is not None:
                    failure["source_grid_index"] = int(frac_pos_index)
                    return fail(failure)

    return _json_safe(_failure_with_limits(
        base,
        warning_grid_points=warning_grid_points,
        hard_grid_points=hard_grid_points,
        max_grid_axis_ratio=max_grid_axis_ratio,
    ))


def _nearest_multiple_at_least(value, divisor):
    value = int(value)
    divisor = int(divisor)
    return int(math.ceil(value / divisor) * divisor)


def _coordinate_denominator(value, max_denominator=12):
    frac = Fraction(float(value)).limit_denominator(int(max_denominator))
    if abs(float(frac) - float(value)) > 1e-6:
        return 1
    return max(1, int(frac.denominator))


def suggest_grid_for_failed_site(grid_size, failed_site, max_denominator=12):
    grid = [int(value) for value in grid_size]
    suggested = list(grid)
    for axis, coord in enumerate(failed_site):
        denominator = _coordinate_denominator(coord, max_denominator=max_denominator)
        if denominator > 1 and suggested[axis] % denominator != 0:
            suggested[axis] = _nearest_multiple_at_least(suggested[axis], denominator)
    return suggested


def ensure_grid_space_group_compatibility(
    space_group,
    grid_size,
    lattice_type,
    lattice_matrix=None,
    warning_grid_points=DEFAULT_WARNING_GRID_POINTS,
    hard_grid_points=DEFAULT_HARD_GRID_POINTS,
    max_grid_axis_ratio=None,
    max_iterations=8,
    validation_mode="auto",
):
    """Return a compatible grid when a small denominator adjustment is enough."""
    mode = _normalize_validation_mode(validation_mode)
    original_grid = [int(value) for value in grid_size]
    grid = list(original_grid)
    attempts = []

    for iteration in range(int(max_iterations) + 1):
        result = evaluate_grid_space_group_compatibility(
            space_group=space_group,
            grid_size=grid,
            lattice_type=lattice_type,
            lattice_matrix=lattice_matrix,
            warning_grid_points=warning_grid_points,
            hard_grid_points=None,
            max_grid_axis_ratio=max_grid_axis_ratio,
            validation_mode=mode,
        )
        attempts.append(result)
        if result["compatible"]:
            result["original_grid"] = original_grid
            result["adjusted"] = grid != original_grid
            result["adjustment_iterations"] = int(iteration)
            result["status"] = "adjusted" if grid != original_grid else "compatible"
            return _json_safe(_failure_with_limits(
                result,
                warning_grid_points=warning_grid_points,
                hard_grid_points=hard_grid_points,
                max_grid_axis_ratio=max_grid_axis_ratio,
            ))

        suggested = result.get("suggested_grid")
        failed_site = result.get("failed_site")
        if suggested is None and failed_site is None:
            rejected = dict(result)
            rejected.update({
                "status": "rejected",
                "reason": result.get("reason") or "grid_space_group_incompatible",
                "attempts": attempts,
            })
            return _json_safe(rejected)

        if suggested is None:
            suggested = suggest_grid_for_failed_site(grid, failed_site)
        suggested = [int(value) for value in suggested]
        if suggested == grid:
            rejected = dict(result)
            rejected.update({
                "status": "rejected",
                "reason": "grid_space_group_incompatible_no_denominator_adjustment",
                "suggested_grid": suggested,
                "attempts": attempts,
            })
            return _json_safe(rejected)

        suggested_points = _grid_points(suggested)
        if hard_grid_points is not None and suggested_points > int(hard_grid_points):
            rejected = dict(result)
            rejected.update({
                "status": "rejected",
                "reason": "grid_space_group_adjustment_exceeds_hard_limit",
                "compatible": False,
                "original_grid": original_grid,
                "suggested_grid": suggested,
                "grid": grid,
                "grid_points": _grid_points(grid),
                "suggested_grid_points": suggested_points,
                "attempts": attempts,
            })
            return _json_safe(rejected)
        if max_grid_axis_ratio is not None and _axis_ratio(suggested) > float(max_grid_axis_ratio):
            rejected = dict(result)
            rejected.update({
                "status": "rejected",
                "reason": "grid_space_group_adjustment_exceeds_axis_ratio_limit",
                "compatible": False,
                "original_grid": original_grid,
                "suggested_grid": suggested,
                "grid": grid,
                "grid_axis_ratio": _axis_ratio(grid),
                "suggested_grid_axis_ratio": _axis_ratio(suggested),
                "attempts": attempts,
            })
            return _json_safe(rejected)

        grid = suggested

    last = dict(attempts[-1])
    last.update({
        "status": "rejected",
        "reason": "grid_space_group_adjustment_did_not_converge",
        "compatible": False,
        "original_grid": original_grid,
        "suggested_grid": grid,
        "attempts": attempts,
    })
    return _json_safe(last)
