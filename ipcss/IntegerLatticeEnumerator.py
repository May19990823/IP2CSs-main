import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np

from ipcss.LatticeEnumerator import actual_lattice_volume, niggli_key


SUPPORTED_INTEGER_SYSTEMS = [
    "Cubic",
    "Tetragonal",
    "Orthorhombic",
    "Hexagonal",
    "Rhombohedral",
]


def _finite_float(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("%s must be a finite number" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be positive" % name)
    return value


def _systems_from_config(crystal_systems):
    if crystal_systems == "all":
        return list(SUPPORTED_INTEGER_SYSTEMS)
    if isinstance(crystal_systems, str):
        systems = [crystal_systems]
    elif crystal_systems is None or not isinstance(crystal_systems, Iterable):
        raise ValueError("crystal_systems must be 'all', a string, or an iterable")
    else:
        systems = list(crystal_systems)
    if not systems:
        raise ValueError("crystal_systems must not be empty")
    unknown = [name for name in systems if name not in SUPPORTED_INTEGER_SYSTEMS]
    if unknown:
        raise ValueError("unsupported integer crystal system(s): %s" % ", ".join(unknown))
    return systems


def _build_lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        raise ValueError("gamma produces a singular lattice")
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < -1e-8:
        raise ValueError("angles produce an invalid lattice")
    cz = math.sqrt(max(0.0, cz_sq))
    return np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=float)


def _volume_in_range(volume, v_min, v_max):
    return v_min <= volume <= v_max


def _axis_ratio_ok(lengths, max_axis_ratio):
    if max_axis_ratio is None:
        return True
    return max(lengths) / min(lengths) <= float(max_axis_ratio)


def _orthogonal_family_and_output(sorted_lengths):
    a, b, c = sorted_lengths
    if a == b == c:
        return "Cubic", [a, b, c]
    if a == b:
        return "Tetragonal", [a, a, c]
    if b == c:
        return "Tetragonal", [b, b, a]
    return "Orthorhombic", [a, b, c]


def _format_angle_value(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def _candidate(lattice_type, lengths, angles, target_volume, sampling_method,
               dedup_key=None, reduction="kept"):
    matrix = _build_lattice_matrix(
        lengths[0], lengths[1], lengths[2],
        angles[0], angles[1], angles[2],
    )
    actual_volume = actual_lattice_volume(matrix)
    key, metadata = niggli_key(matrix, return_metadata=True)
    rel_error = (actual_volume - target_volume) / target_volume if target_volume else 0.0
    return {
        "lattice_matrix": matrix.tolist(),
        "lattice_type": lattice_type,
        "volume": round(actual_volume, 4),
        "crystal_system": lattice_type,
        "SamplingMethod": sampling_method,
        "TargetVolume": float(target_volume),
        "ActualVolume": float(actual_volume),
        "VolumeRelativeError": float(rel_error),
        "LatticeParameters": {
            "lengths": [int(x) for x in lengths],
            "angles": [_format_angle_value(x) for x in angles],
        },
        "DedupKey": list(dedup_key) if dedup_key is not None else list(lengths),
        "IntegerLattice": True,
        "Reduction": reduction,
        "niggli_key": key,
        "NiggliReducer": metadata["reducer"],
        "NiggliCanonical": metadata["canonical"],
    }


def _integer_range(start, stop, step):
    return range(int(start), int(stop) + 1, int(step))


def _enumerate_orthogonal(v_min, v_max, min_lattice_length, integer_step,
                          systems, max_axis_ratio, target_volume):
    max_length = int(math.floor(v_max / (min_lattice_length * min_lattice_length))) + integer_step
    candidates = []
    for lengths in itertools.combinations_with_replacement(
            _integer_range(min_lattice_length, max_length, integer_step), 3):
        volume = lengths[0] * lengths[1] * lengths[2]
        if not _volume_in_range(volume, v_min, v_max):
            continue
        if not _axis_ratio_ok(lengths, max_axis_ratio):
            continue
        lattice_type, output_lengths = _orthogonal_family_and_output(list(lengths))
        if lattice_type not in systems:
            continue
        candidates.append(_candidate(
            lattice_type=lattice_type,
            lengths=output_lengths,
            angles=[90, 90, 90],
            target_volume=target_volume,
            sampling_method="integer_lattice_enumeration",
            dedup_key=list(lengths),
        ))
    return candidates


def _enumerate_hexagonal(v_min, v_max, min_lattice_length, integer_step,
                         include_flat_hexagonal, max_axis_ratio, target_volume):
    factor = math.sqrt(3.0) / 2.0
    max_a = int(math.floor(math.sqrt(v_max / (factor * min_lattice_length)))) + integer_step
    max_c = int(math.floor(v_max / (factor * min_lattice_length * min_lattice_length))) + integer_step
    candidates = []
    for a in _integer_range(min_lattice_length, max_a, integer_step):
        c_start = min_lattice_length if include_flat_hexagonal else a
        for c in _integer_range(c_start, max_c, integer_step):
            lengths = [a, a, c]
            if not _axis_ratio_ok(lengths, max_axis_ratio):
                continue
            volume = factor * a * a * c
            if not _volume_in_range(volume, v_min, v_max):
                continue
            candidates.append(_candidate(
                lattice_type="Hexagonal",
                lengths=lengths,
                angles=[90, 90, 120],
                target_volume=target_volume,
                sampling_method="integer_lattice_enumeration",
                dedup_key=lengths,
            ))
    return candidates


def _rhombohedral_volume_factor(alpha_deg):
    alpha = math.radians(alpha_deg)
    cos_a = math.cos(alpha)
    return math.sqrt(max(0.0, 1.0 - 3.0 * cos_a * cos_a + 2.0 * cos_a ** 3))


def _enumerate_rhombohedral(v_min, v_max, min_lattice_length, integer_step,
                            rhombohedral_angles, target_volume):
    candidates = []
    for alpha in rhombohedral_angles:
        alpha = _finite_float("rhombohedral angle", alpha)
        factor = _rhombohedral_volume_factor(alpha)
        if factor <= 0:
            continue
        max_a = int(math.floor((v_max / factor) ** (1.0 / 3.0))) + integer_step
        for a in _integer_range(min_lattice_length, max_a, integer_step):
            volume = a ** 3 * factor
            if not _volume_in_range(volume, v_min, v_max):
                continue
            candidates.append(_candidate(
                lattice_type="Rhombohedral",
                lengths=[a, a, a],
                angles=[alpha, alpha, alpha],
                target_volume=target_volume,
                sampling_method="integer_lattice_enumeration",
                dedup_key=[a, a, a, alpha],
            ))
    return candidates


def _continuous_runs(items, value_key, step):
    ordered = sorted(items, key=lambda item: item[value_key])
    if not ordered:
        return []
    runs = []
    current = [ordered[0]]
    for item in ordered[1:]:
        if item[value_key] == current[-1][value_key] + step:
            current.append(item)
        else:
            runs.append(current)
            current = [item]
    runs.append(current)
    return runs


def _representatives_for_run(run, target_volume):
    n_items = len(run)
    if n_items == 1:
        return list(run), []
    if n_items == 2:
        kept = min(run, key=lambda item: abs(item["ActualVolume"] - target_volume))
        return [kept], [item for item in run if item is not kept]
    if n_items <= 5:
        kept = [run[0], run[-1]]
        return kept, [item for item in run if item not in kept]
    kept = [run[0], run[(n_items - 1) // 2], run[-1]]
    return kept, [item for item in run if item not in kept]


def _reduce_sequence_groups(candidates, group_key, value_key, integer_step, target_volume):
    groups = defaultdict(list)
    for cand in candidates:
        row = dict(cand)
        row["_sequence_value"] = value_key(row)
        groups[group_key(row)].append(row)

    kept = []
    rejected = []
    for group in groups.values():
        for run in _continuous_runs(group, "_sequence_value", integer_step):
            run_kept, run_rejected = _representatives_for_run(run, target_volume)
            kept.extend(run_kept)
            rejected.extend(run_rejected)

    for row in kept:
        row.pop("_sequence_value", None)
        row["Reduction"] = "kept"
    for row in rejected:
        row.pop("_sequence_value", None)
        row["Reduction"] = "sequence_reduced"
        row["reason"] = "sequence_reduced"
    return kept, rejected


def _apply_sequence_reduction(candidates, integer_step, target_volume):
    by_type = defaultdict(list)
    for cand in candidates:
        by_type[cand["lattice_type"]].append(cand)

    kept = []
    rejected = []
    kept.extend(by_type.get("Cubic", []))
    kept.extend(by_type.get("Orthorhombic", []))

    tetr_kept, tetr_rejected = _reduce_sequence_groups(
        by_type.get("Tetragonal", []),
        group_key=lambda cand: cand["LatticeParameters"]["lengths"][0],
        value_key=lambda cand: cand["LatticeParameters"]["lengths"][2],
        integer_step=integer_step,
        target_volume=target_volume,
    )
    kept.extend(tetr_kept)
    rejected.extend(tetr_rejected)

    hex_kept, hex_rejected = _reduce_sequence_groups(
        by_type.get("Hexagonal", []),
        group_key=lambda cand: cand["LatticeParameters"]["lengths"][0],
        value_key=lambda cand: cand["LatticeParameters"]["lengths"][2],
        integer_step=integer_step,
        target_volume=target_volume,
    )
    kept.extend(hex_kept)
    rejected.extend(hex_rejected)

    rhom_kept, rhom_rejected = _reduce_sequence_groups(
        by_type.get("Rhombohedral", []),
        group_key=lambda cand: cand["LatticeParameters"]["angles"][0],
        value_key=lambda cand: cand["LatticeParameters"]["lengths"][0],
        integer_step=integer_step,
        target_volume=target_volume,
    )
    kept.extend(rhom_kept)
    rejected.extend(rhom_rejected)
    return kept, rejected


def _sort_key(cand):
    order = {name: idx for idx, name in enumerate(SUPPORTED_INTEGER_SYSTEMS)}
    return (
        order.get(cand["lattice_type"], 99),
        round(cand["ActualVolume"], 8),
        cand["LatticeParameters"]["lengths"],
        cand["LatticeParameters"]["angles"],
    )


def enumerate_integer_lattices(V_min, V_max, crystal_systems="all",
                               min_lattice_length=4, integer_step=1,
                               rhombohedral_angles=(55, 60, 65, 70),
                               include_flat_hexagonal=True,
                               reduce_sequences=True,
                               max_axis_ratio=None,
                               return_rejected=False):
    V_min = _finite_float("V_min", V_min)
    V_max = _finite_float("V_max", V_max)
    if V_min <= 0 or V_max <= 0:
        raise ValueError("V_min and V_max must be positive")
    if V_min > V_max:
        raise ValueError("V_min must be <= V_max")
    min_lattice_length = _positive_int("min_lattice_length", min_lattice_length)
    integer_step = _positive_int("integer_step", integer_step)
    systems = _systems_from_config(crystal_systems)
    target_volume = (V_min + V_max) / 2.0

    raw = []
    orthogonal_systems = [name for name in systems if name in ("Cubic", "Tetragonal", "Orthorhombic")]
    if orthogonal_systems:
        raw.extend(_enumerate_orthogonal(
            V_min, V_max, min_lattice_length, integer_step,
            set(orthogonal_systems), max_axis_ratio, target_volume,
        ))
    if "Hexagonal" in systems:
        raw.extend(_enumerate_hexagonal(
            V_min, V_max, min_lattice_length, integer_step,
            bool(include_flat_hexagonal), max_axis_ratio, target_volume,
        ))
    if "Rhombohedral" in systems:
        raw.extend(_enumerate_rhombohedral(
            V_min, V_max, min_lattice_length, integer_step,
            rhombohedral_angles, target_volume,
        ))

    if reduce_sequences:
        reduced, rejected = _apply_sequence_reduction(raw, integer_step, target_volume)
    else:
        reduced, rejected = list(raw), []

    reduced = sorted(reduced, key=_sort_key)
    rejected = sorted(rejected, key=_sort_key)
    diagnostics = {
        "enumeration_mode": "integer",
        "raw_candidate_count": len(raw),
        "reduced_candidate_count": len(reduced),
        "rejected_candidate_count": len(rejected),
        "V_min": V_min,
        "V_max": V_max,
        "target_volume": target_volume,
        "min_lattice_length": min_lattice_length,
        "integer_step": integer_step,
        "include_flat_hexagonal": bool(include_flat_hexagonal),
        "reduce_sequences": bool(reduce_sequences),
        "crystal_systems": systems,
        "rhombohedral_angles": list(rhombohedral_angles),
    }
    if return_rejected:
        return reduced, rejected, diagnostics
    return reduced
