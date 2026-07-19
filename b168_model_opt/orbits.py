from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path

import ase
import numpy as np
from matscipy.neighbours import neighbour_list

from b168_model_opt.types import OrbitData


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _validated_grid_size(grid_size: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        grid = tuple(grid_size)
    except TypeError as exc:
        raise ValueError("grid_size must contain exactly three positive integers") from exc
    if (
        len(grid) != 3
        or any(not _is_integer(value) or value <= 0 for value in grid)
    ):
        raise ValueError("grid_size must contain exactly three positive integers")
    return tuple(int(value) for value in grid)


def _validated_cell(cell: np.ndarray) -> np.ndarray:
    try:
        matrix = np.asarray(cell, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell matrix must have shape (3, 3)") from exc
    if matrix.shape != (3, 3):
        raise ValueError("cell matrix must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("cell matrix must contain finite values")
    if float(np.linalg.svd(matrix, compute_uv=False).min()) <= np.finfo(np.float64).eps:
        raise ValueError("cell matrix must be nonsingular")
    return matrix


def _validated_positive_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _validated_nonnegative_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _json_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, str) and str(result) != value:
        raise ValueError(f"{field} must be an integer")
    return result


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"orbit JSON contains duplicate key {key!r}")
        payload[key] = value
    return payload


def load_orbit_members(path: str | Path) -> tuple[tuple[int, ...], ...]:
    """Load the production orbit-cache JSON into deterministic member tuples."""
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("orbit JSON must contain an object mapping representatives to members")

    ordered: list[tuple[int, tuple[int, ...]]] = []
    for raw_key, raw_members in payload.items():
        key = _json_integer(raw_key, "orbit representative")
        if not isinstance(raw_members, list):
            raise ValueError(f"members for orbit {key} must be a list")
        members = tuple(
            _json_integer(member, f"member of orbit {key}") for member in raw_members
        )
        ordered.append((key, members))
    ordered.sort(key=lambda item: item[0])
    return tuple(
        (key, *sorted(members)) for key, members in ordered
    )


def _validated_orbit_members(
    orbit_members: Iterable[Iterable[int]], site_count: int
) -> tuple[tuple[int, ...], ...]:
    try:
        normalized = tuple(tuple(members) for members in orbit_members)
    except TypeError as exc:
        raise ValueError("orbit_members must contain iterable orbits") from exc

    owner = np.full(site_count, -1, dtype=np.int64)
    for orbit_id, members in enumerate(normalized):
        if not members:
            raise ValueError(f"orbit {orbit_id} is empty")
        for site in members:
            if not _is_integer(site):
                raise ValueError(f"site {site!r} must be a non-boolean integer")
            if site < 0 or site >= site_count:
                raise ValueError(f"site {site} outside grid")
            if owner[site] != -1:
                raise ValueError(f"site {site} occurs in more than one orbit")
            owner[site] = orbit_id
    if np.any(owner < 0):
        missing = np.flatnonzero(owner < 0).tolist()
        raise ValueError(f"orbit partition misses sites: {missing[:20]}")
    return normalized


def _validated_blocked_original(
    blocked_original: Iterable[int], orbit_count: int
) -> frozenset[int]:
    try:
        blocked = frozenset(blocked_original)
    except TypeError as exc:
        raise ValueError("blocked_original must contain orbit IDs") from exc
    for orbit_id in blocked:
        if not _is_integer(orbit_id) or not 0 <= orbit_id < orbit_count:
            raise ValueError(
                f"blocked_original contains invalid orbit ID {orbit_id!r}"
            )
    return blocked


def _validated_conflict_edges(
    original_conflict_edges: Iterable[Iterable[int]], orbit_count: int
) -> frozenset[tuple[int, int]]:
    try:
        raw_edges = tuple(original_conflict_edges)
    except TypeError as exc:
        raise ValueError("original_conflict_edges must contain pairs") from exc

    edges: set[tuple[int, int]] = set()
    for edge in raw_edges:
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("original_conflict_edges must contain pairs") from exc
        if (
            not _is_integer(left)
            or not _is_integer(right)
            or not 0 <= left < orbit_count
            or not 0 <= right < orbit_count
        ):
            raise ValueError(f"conflict edge has invalid endpoint: {edge!r}")
        if left == right:
            raise ValueError("conflict edges must connect distinct orbit IDs")
        edges.add(tuple(sorted((int(left), int(right)))))
    return frozenset(edges)


def compact_orbits(
    grid_size: tuple[int, int, int],
    orbit_members: tuple[tuple[int, ...], ...],
    blocked_original: set[int],
    original_conflict_edges: set[tuple[int, int]],
) -> OrbitData:
    """Validate an orbit partition and remove blocked orbits from model space."""
    grid = _validated_grid_size(grid_size)
    site_count = math.prod(grid)
    members = _validated_orbit_members(orbit_members, site_count)
    orbit_count = len(members)
    blocked = _validated_blocked_original(blocked_original, orbit_count)
    original_edges = _validated_conflict_edges(original_conflict_edges, orbit_count)

    site_to_original = np.full(site_count, -1, dtype=np.int64)
    for orbit_id, orbit_sites in enumerate(members):
        site_to_original[list(orbit_sites)] = orbit_id

    active_original = tuple(
        orbit_id for orbit_id in range(orbit_count) if orbit_id not in blocked
    )
    original_to_active = {
        original: active for active, original in enumerate(active_original)
    }
    active_by_original = np.full(orbit_count, -1, dtype=np.int64)
    active_by_original[list(active_original)] = np.arange(
        len(active_original), dtype=np.int64
    )
    site_to_active = active_by_original[site_to_original]

    conflict_edges = frozenset(
        tuple(sorted((original_to_active[left], original_to_active[right])))
        for left, right in original_edges
        if left in original_to_active and right in original_to_active
    )
    return OrbitData(
        grid_size=grid,
        original_orbit_ids=tuple(range(orbit_count)),
        members=members,
        sizes=np.asarray([len(orbit_sites) for orbit_sites in members], dtype=np.int64),
        site_to_original=site_to_original,
        blocked_original=blocked,
        active_original=active_original,
        original_to_active=original_to_active,
        site_to_active=site_to_active,
        conflict_edges=conflict_edges,
    )


def _validated_fractional_positions(
    fractional_positions: object, site_count: int
) -> np.ndarray:
    try:
        positions = np.asarray(fractional_positions, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"fractional_positions must have shape ({site_count}, 3)"
        ) from exc
    if positions.shape != (site_count, 3):
        raise ValueError(f"fractional_positions must have shape ({site_count}, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("fractional_positions must contain finite values")
    return positions


def find_min_distance_conflicts(
    grid_size: tuple[int, int, int],
    cell: np.ndarray,
    site_to_original: np.ndarray,
    r_min: float,
    tolerance: float = 1e-8,
    *,
    fractional_positions: np.ndarray,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Return blocked orbits and cross-orbit edges under the production hard minimum.

    ``fractional_positions`` is the required production coordinate stream.
    """
    grid = _validated_grid_size(grid_size)
    matrix = _validated_cell(cell)
    minimum = _validated_positive_float(r_min, "r_min")
    threshold_tolerance = _validated_nonnegative_float(tolerance, "tolerance")
    if threshold_tolerance >= minimum:
        raise ValueError("tolerance must be less than r_min")
    owners = np.asarray(site_to_original)
    site_count = math.prod(grid)
    if owners.ndim != 1 or len(owners) != site_count:
        raise ValueError(
            f"site_to_original must be one-dimensional with length {site_count}"
        )
    if owners.dtype.kind not in "iu" or owners.dtype.kind == "b":
        raise ValueError("site_to_original must have a non-boolean integer dtype")
    if np.any(owners < 0):
        raise ValueError("site_to_original must contain nonnegative orbit IDs")

    coordinates = _validated_fractional_positions(fractional_positions, site_count)
    atoms = ase.Atoms(
        symbols="H" * site_count,
        scaled_positions=coordinates,
        pbc=True,
        cell=matrix,
    )
    left_sites, right_sites, distances = neighbour_list(
        "ijd", atoms=atoms, cutoff=minimum
    )

    blocked: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for left_site, right_site, distance in zip(left_sites, right_sites, distances):
        if float(distance) >= minimum - threshold_tolerance:
            continue
        left_orbit = int(owners[int(left_site)])
        right_orbit = int(owners[int(right_site)])
        if left_orbit == right_orbit:
            blocked.add(left_orbit)
        else:
            edges.add(tuple(sorted((left_orbit, right_orbit))))
    return blocked, edges
