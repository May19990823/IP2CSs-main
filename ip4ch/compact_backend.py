from __future__ import annotations

import math
from collections.abc import Mapping

import ase
import numpy as np
from matscipy.neighbours import neighbour_list

from b168_model_opt.cliques import greedy_edge_clique_cover
from b168_model_opt.orbits import compact_orbits
from b168_model_opt.pair_table import PairTable
from b168_model_opt.types import OrbitData, PairCoefficients


def ordered_orbit_members(
    orbits: Mapping[str, list[int]], site_count: int
) -> tuple[tuple[int, ...], ...]:
    """Convert the production representative/member mapping to full orbits."""
    members = []
    seen = set()
    for representative, equivalents in orbits.items():
        orbit = (int(representative), *(int(site) for site in equivalents))
        if len(set(orbit)) != len(orbit):
            raise ValueError(f"orbit {representative} contains duplicate sites")
        overlap = seen.intersection(orbit)
        if overlap:
            raise ValueError(f"sites occur in multiple orbits: {sorted(overlap)[:10]}")
        seen.update(orbit)
        members.append(orbit)
    expected = set(range(int(site_count)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"orbit partition mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    return tuple(members)


def build_orbit_data(
    grid_size,
    orbits,
    blocked_orbits,
    blocked_orbit_pairs,
) -> OrbitData:
    site_count = math.prod(int(value) for value in grid_size)
    return compact_orbits(
        grid_size=tuple(int(value) for value in grid_size),
        orbit_members=ordered_orbit_members(orbits, site_count),
        blocked_original=set(int(value) for value in blocked_orbits),
        original_conflict_edges={
            tuple(sorted((int(left), int(right))))
            for left, right in blocked_orbit_pairs
        },
    )


def aggregate_production_dense_alpha(
    orbit_data: OrbitData, alpha_matrix: np.ndarray
) -> PairCoefficients:
    """Aggregate the exact production ``i < j`` dense-alpha objective."""
    active_by_site = np.asarray(orbit_data.site_to_active, dtype=np.int64)
    site_count = len(active_by_site)
    matrix = np.asarray(alpha_matrix)
    if matrix.shape != (site_count, site_count):
        raise ValueError(
            "production dense alpha matrix must be square with one row per grid site"
        )
    if matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        raise ValueError("production dense alpha matrix must contain finite floats")

    active_count = len(orbit_data.active_original)
    linear = np.zeros(active_count, dtype=np.float64)
    quadratic: dict[tuple[int, int], float] = {}
    for first in range(site_count - 1):
        left = int(active_by_site[first])
        if left < 0:
            continue
        targets = active_by_site[first + 1 :]
        weights = np.asarray(matrix[first, first + 1 :], dtype=np.float64)
        active = targets >= 0
        targets = targets[active]
        weights = weights[active]

        same = targets == left
        if np.any(same):
            linear[left] += float(np.add.reduce(weights[same]))
        for right, coefficient in zip(targets[~same], weights[~same]):
            if coefficient == 0.0:
                continue
            edge = tuple(sorted((left, int(right))))
            if edge in orbit_data.conflict_edges:
                continue
            quadratic[edge] = quadratic.get(edge, 0.0) + float(coefficient)

    return PairCoefficients(
        linear=linear,
        quadratic={edge: value for edge, value in sorted(quadratic.items()) if value},
        metadata={
            "active_orbits": active_count,
            "pair_order": "production-i<j",
            "source": "production_dense_alpha",
        },
    )


def aggregate_sparse_neighbour_coefficients(
    *,
    orbit_data: OrbitData,
    lattice_matrix,
    fractional_positions,
    r_min: float,
    r_max: float,
    pair_file,
) -> PairCoefficients:
    """Aggregate the production neighbour stream without allocating dense alpha."""
    site_count = len(orbit_data.site_to_active)
    positions = np.asarray(fractional_positions, dtype=np.float64)
    if positions.shape != (site_count, 3):
        raise ValueError(
            f"fractional_positions must have shape ({site_count}, 3)"
        )
    table = PairTable.from_csv(pair_file)
    atoms = ase.Atoms(
        symbols="H" * site_count,
        scaled_positions=positions,
        pbc=True,
        cell=np.asarray(lattice_matrix, dtype=np.float64),
    )
    first_sites, second_sites, distances = neighbour_list(
        "ijd", atoms=atoms, cutoff=float(r_max)
    )

    # Match production's float32 accumulation for each ordered site pair.
    site_pair_totals: dict[int, np.float32] = {}
    minimum = float(r_min)
    for first, second, distance in zip(first_sites, second_sites, distances):
        first = int(first)
        second = int(second)
        if first >= second or float(distance) < minimum - 1e-8:
            continue
        energy = table.energy(float(distance))
        if energy is None:
            continue
        code = first * site_count + second
        site_pair_totals[code] = np.float32(
            site_pair_totals.get(code, np.float32(0.0)) + energy
        )

    active_by_site = np.asarray(orbit_data.site_to_active, dtype=np.int64)
    active_count = len(orbit_data.active_original)
    linear = np.zeros(active_count, dtype=np.float64)
    quadratic: dict[tuple[int, int], float] = {}
    for code in sorted(site_pair_totals):
        first, second = divmod(code, site_count)
        left = int(active_by_site[first])
        right = int(active_by_site[second])
        if left < 0 or right < 0:
            continue
        coefficient = float(site_pair_totals[code])
        if left == right:
            linear[left] += coefficient
            continue
        edge = tuple(sorted((left, right)))
        if edge in orbit_data.conflict_edges:
            continue
        quadratic[edge] = quadratic.get(edge, 0.0) + coefficient

    return PairCoefficients(
        linear=linear,
        quadratic={edge: value for edge, value in sorted(quadratic.items()) if value},
        metadata={
            "active_orbits": active_count,
            "pair_order": "production-i<j",
            "source": "sparse_production_neighbour_stream",
            "site_pair_terms": len(site_pair_totals),
        },
    )


def build_pair_coefficients(
    *,
    orbit_data: OrbitData,
    lattice_type: str,
    grid_size,
    lattice_matrix,
    fractional_positions,
    r_min: float,
    r_max: float,
    pair_file,
) -> tuple[PairCoefficients, str]:
    coefficients = aggregate_sparse_neighbour_coefficients(
        orbit_data=orbit_data,
        lattice_matrix=lattice_matrix,
        fractional_positions=fractional_positions,
        r_min=r_min,
        r_max=r_max,
        pair_file=pair_file,
    )
    return coefficients, "sparse_production_neighbour_stream"


def conflict_cover(orbit_data: OrbitData):
    return greedy_edge_clique_cover(
        node_count=len(orbit_data.active_original),
        edges=orbit_data.conflict_edges,
    )
