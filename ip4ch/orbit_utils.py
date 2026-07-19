from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class OrbitPartitionReport:
    grid_size: tuple[int, int, int]
    total_grid_points: int
    orbit_count: int
    duplicate_points: dict[int, list[str]]
    missing_points: list[int]
    out_of_range_points: list[int]
    non_integer_points: list[str]
    key_in_members: list[int]

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_points
            or self.missing_points
            or self.out_of_range_points
            or self.non_integer_points
            or self.key_in_members
        )

    def error_summary(self, max_examples: int = 12) -> str:
        parts: list[str] = []
        if self.duplicate_points:
            examples = [
                f"{point}: {owners}"
                for point, owners in list(self.duplicate_points.items())[:max_examples]
            ]
            parts.append(
                "duplicate grid points "
                f"({len(self.duplicate_points)}): " + "; ".join(examples)
            )
        if self.missing_points:
            parts.append(
                "missing grid points "
                f"({len(self.missing_points)}): {self.missing_points[:max_examples]}"
            )
        if self.out_of_range_points:
            parts.append(
                "out-of-range grid points "
                f"({len(self.out_of_range_points)}): {self.out_of_range_points[:max_examples]}"
            )
        if self.non_integer_points:
            parts.append(
                "non-integer grid point ids "
                f"({len(self.non_integer_points)}): {self.non_integer_points[:max_examples]}"
            )
        if self.key_in_members:
            parts.append(
                "orbit keys also listed as members "
                f"({len(self.key_in_members)}): {self.key_in_members[:max_examples]}"
            )
        return "; ".join(parts) if parts else "valid orbit partition"


def _parse_grid_size(grid_size: Any) -> tuple[int, int, int]:
    if len(grid_size) != 3:
        raise ValueError(f"grid_size must contain three values, got {grid_size!r}")
    parsed = tuple(int(value) for value in grid_size)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"grid_size values must be positive, got {grid_size!r}")
    return parsed


def _parse_grid_point(value: Any, label: str, errors: list[str]) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label}={value!r}")
        return None
    if str(parsed) != str(value) and not isinstance(value, int):
        errors.append(f"{label}={value!r}")
        return None
    return parsed


def validate_orbit_partition(orbits: dict[str, list[int]], grid_size: Any) -> OrbitPartitionReport:
    grid = _parse_grid_size(grid_size)
    total_grid_points = grid[0] * grid[1] * grid[2]
    owners_by_point: dict[int, list[str]] = defaultdict(list)
    out_of_range: set[int] = set()
    non_integer: list[str] = []
    key_in_members: set[int] = set()

    for raw_key, raw_members in orbits.items():
        orbit_key = _parse_grid_point(raw_key, "orbit_key", non_integer)
        if orbit_key is None:
            owner_label = str(raw_key)
        else:
            owner_label = str(orbit_key)
            owners_by_point[orbit_key].append(owner_label)
            if orbit_key < 0 or orbit_key >= total_grid_points:
                out_of_range.add(orbit_key)

        if not isinstance(raw_members, list):
            non_integer.append(f"members_of_{raw_key}=not_a_list")
            continue

        for raw_member in raw_members:
            member = _parse_grid_point(raw_member, f"member_of_{raw_key}", non_integer)
            if member is None:
                continue
            owners_by_point[member].append(owner_label)
            if member < 0 or member >= total_grid_points:
                out_of_range.add(member)
            if orbit_key is not None and member == orbit_key:
                key_in_members.add(member)

    duplicate_points = {
        point: owners
        for point, owners in sorted(owners_by_point.items())
        if len(owners) > 1 and 0 <= point < total_grid_points
    }
    present_points = {point for point in owners_by_point if 0 <= point < total_grid_points}
    missing_points = [point for point in range(total_grid_points) if point not in present_points]

    return OrbitPartitionReport(
        grid_size=grid,
        total_grid_points=total_grid_points,
        orbit_count=len(orbits),
        duplicate_points=duplicate_points,
        missing_points=missing_points,
        out_of_range_points=sorted(out_of_range),
        non_integer_points=non_integer,
        key_in_members=sorted(key_in_members),
    )


def assert_valid_orbit_partition(
    orbits: dict[str, list[int]],
    grid_size: Any,
    context: str = "orbit partition",
) -> OrbitPartitionReport:
    report = validate_orbit_partition(orbits=orbits, grid_size=grid_size)
    if not report.ok:
        raise ValueError(f"Invalid {context}: {report.error_summary()}")
    return report
