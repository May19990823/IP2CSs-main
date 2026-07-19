from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite, prod
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np


def _read_only_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True)
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)


def _deep_freeze(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("unsupported metadata ndarray object dtype")
        return _read_only_copy(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(nested) for nested in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(nested) for nested in value)
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _nested_tuples(values: Iterable[Iterable[object]], field: str) -> tuple[tuple, ...]:
    try:
        return tuple(tuple(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{field} must contain iterable values") from exc


def _edge_tuple(edge: Iterable[int], field: str) -> tuple[int, int]:
    try:
        normalized = tuple(edge)
    except TypeError as exc:
        raise ValueError(f"{field} must contain pairs") from exc
    if len(normalized) != 2:
        raise ValueError(f"{field} must contain pairs")
    return normalized


def _edge_tuple_sequence(
    edges: Iterable[Iterable[int]], field: str
) -> tuple[tuple[int, int], ...]:
    return tuple(_edge_tuple(edge, field) for edge in edges)


def _edge_frozenset(
    edges: Iterable[Iterable[int]], field: str
) -> frozenset[tuple[int, int]]:
    return frozenset(_edge_tuple(edge, field) for edge in edges)


@dataclass(frozen=True, eq=False)
class OrbitData:
    grid_size: tuple[int, int, int]
    original_orbit_ids: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]
    sizes: np.ndarray
    site_to_original: np.ndarray
    blocked_original: frozenset[int]
    active_original: tuple[int, ...]
    original_to_active: Mapping[int, int]
    site_to_active: np.ndarray
    conflict_edges: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        try:
            grid_size = tuple(self.grid_size)
            original_orbit_ids = tuple(self.original_orbit_ids)
            active_original = tuple(self.active_original)
            blocked_original = frozenset(self.blocked_original)
        except TypeError as exc:
            raise ValueError("orbit fields must use iterable values") from exc

        members = _nested_tuples(self.members, "members")
        conflict_edges = _edge_frozenset(self.conflict_edges, "conflict_edges")
        sizes = _read_only_copy(self.sizes)
        site_to_original = _read_only_copy(self.site_to_original)
        site_to_active = _read_only_copy(self.site_to_active)
        original_to_active = MappingProxyType(dict(self.original_to_active))

        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "original_orbit_ids", original_orbit_ids)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "site_to_original", site_to_original)
        object.__setattr__(self, "blocked_original", blocked_original)
        object.__setattr__(self, "active_original", active_original)
        object.__setattr__(self, "original_to_active", original_to_active)
        object.__setattr__(self, "site_to_active", site_to_active)
        object.__setattr__(self, "conflict_edges", conflict_edges)

        if (
            len(grid_size) != 3
            or any(not _is_integer(dimension) for dimension in grid_size)
            or any(dimension <= 0 for dimension in grid_size)
        ):
            raise ValueError("grid_size must contain exactly three positive integers")

        if (
            any(
                not _is_integer(original) or original < 0
                for original in original_orbit_ids
            )
            or len(frozenset(original_orbit_ids)) != len(original_orbit_ids)
        ):
            raise ValueError(
                "original_orbit_ids must contain unique nonnegative integers"
            )

        site_count = prod(grid_size)
        for field, array in (
            ("site_to_original", site_to_original),
            ("site_to_active", site_to_active),
        ):
            if array.ndim != 1 or len(array) != site_count:
                raise ValueError(
                    f"{field} must be one-dimensional with length {site_count}"
                )
            if array.dtype.kind not in "iu":
                raise ValueError(f"{field} must have a non-boolean integer dtype")

        if sizes.ndim != 1 or len(sizes) != len(original_orbit_ids):
            raise ValueError(
                "sizes must be one-dimensional with one entry per original orbit"
            )
        if sizes.dtype.kind not in "iu":
            raise ValueError("sizes must have a non-boolean integer dtype")

        if len(members) != len(original_orbit_ids):
            raise ValueError("members must contain one orbit for each original_orbit_id")
        if original_orbit_ids != tuple(range(len(members))):
            raise ValueError(
                "original_orbit_ids must equal tuple(range(len(members)))"
            )
        if any(
            not orbit_members
            or any(not _is_integer(site) for site in orbit_members)
            for orbit_members in members
        ):
            raise ValueError("members must contain nonempty tuples of integer sites")

        site_owners = [None] * site_count
        for original, orbit_members in zip(original_orbit_ids, members):
            for site in orbit_members:
                if site < 0 or site >= site_count or site_owners[site] is not None:
                    raise ValueError(
                        f"members must partition sites 0 through {site_count - 1} "
                        "exactly once"
                    )
                site_owners[site] = original
        if any(owner is None for owner in site_owners):
            raise ValueError(
                f"members must partition sites 0 through {site_count - 1} exactly once"
            )

        expected_sizes = np.asarray([len(orbit_members) for orbit_members in members])
        if not np.array_equal(sizes, expected_sizes):
            raise ValueError("sizes must be positive and equal the member tuple lengths")
        if not np.array_equal(site_to_original, np.asarray(site_owners)):
            raise ValueError(
                "site_to_original must identify the original orbit owning each site"
            )

        original_id_set = frozenset(original_orbit_ids)
        active_original_set = frozenset(active_original)
        if any(not _is_integer(original) for original in blocked_original):
            raise ValueError("blocked_original must contain non-boolean integers")
        if any(not _is_integer(original) for original in active_original):
            raise ValueError("active_original must contain non-boolean integers")
        if not blocked_original <= original_id_set:
            raise ValueError(
                "blocked_original must contain only IDs from original_orbit_ids"
            )
        if not active_original_set <= original_id_set:
            raise ValueError(
                "active_original must contain only IDs from original_orbit_ids"
            )
        if blocked_original & active_original_set:
            raise ValueError("blocked_original and active_original must be disjoint")

        expected_mapping = {
            original: active for active, original in enumerate(active_original)
        }
        if any(
            not _is_integer(original) or not _is_integer(active)
            for original, active in original_to_active.items()
        ):
            raise ValueError(
                "original_to_active keys and values must be non-boolean integers"
            )
        if (
            len(expected_mapping) != len(active_original)
            or dict(original_to_active) != expected_mapping
        ):
            raise ValueError(
                "original_to_active must map each active original orbit to its position"
            )

        active_count = len(active_original)
        for left, right in conflict_edges:
            if (
                not _is_integer(left)
                or not _is_integer(right)
                or not 0 <= left < right < active_count
            ):
                raise ValueError(
                    "conflict_edges must contain canonical integer active indices "
                    "with 0 <= left < right < active_count"
                )

        for original, active in zip(site_to_original, site_to_active):
            if original in blocked_original:
                expected_active = -1
            elif original in expected_mapping:
                expected_active = expected_mapping[original]
            else:
                raise ValueError(
                    "site_to_original must reference a blocked or active original orbit"
                )
            if active != expected_active:
                raise ValueError(
                    "site_to_active is inconsistent with site_to_original, "
                    "blocked_original, active_original, and original_to_active"
                )


@dataclass(frozen=True, eq=False)
class PairCoefficients:
    linear: np.ndarray
    quadratic: Mapping[tuple[int, int], float]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.linear, np.ndarray):
            raise ValueError("linear must be a float64 numpy vector")
        linear = _read_only_copy(self.linear)
        if linear.ndim != 1 or linear.dtype != np.dtype(np.float64):
            raise ValueError("linear must be a one-dimensional float64 vector")
        if not np.isfinite(linear).all():
            raise ValueError("linear must contain only finite values")

        if not isinstance(self.quadratic, Mapping):
            raise ValueError("quadratic must be a mapping")
        normalized_quadratic: dict[tuple[int, int], float] = {}
        for edge, coefficient in self.quadratic.items():
            left, right = _edge_tuple(edge, "quadratic")
            if (
                not _is_integer(left)
                or not _is_integer(right)
                or not 0 <= left < right < len(linear)
            ):
                raise ValueError(
                    "quadratic keys must be canonical non-boolean integer pairs in bounds"
                )
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, Real)
                or isinstance(coefficient, Integral)
                or not isfinite(coefficient)
            ):
                raise ValueError("quadratic coefficients must be finite float values")
            normalized_quadratic[(int(left), int(right))] = float(coefficient)
        quadratic = MappingProxyType(
            normalized_quadratic
        )
        if not isinstance(self.metadata, Mapping) or any(
            not isinstance(key, str) for key in self.metadata
        ):
            raise ValueError("metadata must be a mapping with string keys")
        metadata = MappingProxyType(
            {key: _deep_freeze(value) for key, value in self.metadata.items()}
        )

        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "metadata", metadata)

@dataclass(frozen=True)
class CliqueCover:
    cliques: tuple[tuple[int, ...], ...]
    residual_edges: tuple[tuple[int, int], ...]
    covered_edges: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        cliques = _nested_tuples(self.cliques, "cliques")
        residual_edges = _edge_tuple_sequence(self.residual_edges, "residual_edges")
        covered_edges = _edge_frozenset(self.covered_edges, "covered_edges")

        object.__setattr__(self, "cliques", cliques)
        object.__setattr__(self, "residual_edges", residual_edges)
        object.__setattr__(self, "covered_edges", covered_edges)

        for field, edges in (
            ("residual_edges", residual_edges),
            ("covered_edges", covered_edges),
        ):
            if any(left >= right for left, right in edges):
                raise ValueError(f"{field} must contain canonical edges with left < right")
