import itertools
import hashlib
import json
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np

# ============================================================
# Crystal system definitions
# ============================================================

CRYSTAL_SYSTEMS = {
    "Triclinic":    {"n_length_params": 3, "n_angle_params": 3, "angle_range": (60, 120)},
    "Monoclinic":   {"n_length_params": 3, "n_angle_params": 1, "fixed_angles": {"alpha": 90, "gamma": 90}, "angle_range": (60, 120)},
    "Orthorhombic": {"n_length_params": 3, "n_angle_params": 0, "fixed_angles": {"alpha": 90, "beta": 90, "gamma": 90}},
    "Tetragonal":   {"n_length_params": 2, "n_angle_params": 0, "fixed_angles": {"alpha": 90, "beta": 90, "gamma": 90}},
    "Hexagonal":    {"n_length_params": 2, "n_angle_params": 0, "fixed_angles": {"alpha": 90, "beta": 90, "gamma": 120}},
    "Cubic":        {"n_length_params": 1, "n_angle_params": 0, "fixed_angles": {"alpha": 90, "beta": 90, "gamma": 90}},
}


def get_crystal_system_params(system_name):
    if system_name not in CRYSTAL_SYSTEMS:
        raise ValueError(f"Unknown crystal system: {system_name}")
    return CRYSTAL_SYSTEMS[system_name]


# ============================================================
# Volume sampling
# ============================================================

def log_volume_grid(V_min, V_max, volume_step_factor):
    if V_min >= V_max:
        return [V_min]
    log_min, log_max = np.log(V_min), np.log(V_max)
    n = max(1, int(np.ceil((log_max - log_min) / volume_step_factor)))
    return np.exp(np.linspace(log_min, log_max, n + 1)).tolist()


# ============================================================
# Lattice matrix builder
# ============================================================

def _build_lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    cz = np.sqrt(max(0.0, cz_sq))
    return np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])


def angle_volume_factor(alpha_deg, beta_deg, gamma_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    factor_sq = (
        1.0
        + 2.0 * cos_a * cos_b * cos_g
        - cos_a ** 2
        - cos_b ** 2
        - cos_g ** 2
    )
    return float(np.sqrt(max(0.0, factor_sq)))


def actual_lattice_volume(lattice_matrix):
    return float(abs(np.linalg.det(np.array(lattice_matrix, dtype=float))))


def _volume_rejection_reason(actual_volume, V_min, V_max, tolerance_fraction):
    low = V_min * (1.0 - tolerance_fraction)
    high = V_max * (1.0 + tolerance_fraction)
    if actual_volume < low or actual_volume > high:
        return "actual_volume_out_of_bounds"
    return None


def _axis_ratio_values(max_axis_ratio, n_samples):
    max_ratio = float(max_axis_ratio or 3.0)
    n = max(2, int(n_samples))
    return np.linspace(1.0, max_ratio, n).tolist()


def _require_finite_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_finite_volume_bounds(V_min, V_max):
    values = (V_min, V_max)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("V_min and V_max must be finite")
    if not np.isfinite(V_min) or not np.isfinite(V_max):
        raise ValueError("V_min and V_max must be finite")
    return float(V_min), float(V_max)


def _require_positive_real(name, value):
    value = _require_finite_real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_min_real(name, value, min_value):
    value = _require_finite_real(name, value)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return value


def _require_non_negative_real(name, value):
    value = _require_finite_real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


# ============================================================
# Length / angle sampling
# ============================================================

def _passes_bulk_length_filter(lengths, min_lattice_length=None, max_axis_ratio=None):
    lengths = np.array(lengths, dtype=float)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        return False
    if min_lattice_length is not None and np.min(lengths) < min_lattice_length:
        return False
    if max_axis_ratio is not None:
        axis_ratio = float(np.max(lengths) / np.min(lengths))
        if axis_ratio > max_axis_ratio:
            return False
    return True


def _sample_lengths_for_volume(
    n_free,
    volume,
    min_atom_spacing,
    n_angle_samples,
    min_lattice_length=None,
    max_axis_ratio=None,
    angle_factor=1.0,
):
    results = []
    min_length = max(min_atom_spacing, min_lattice_length or min_atom_spacing)
    factor = max(float(angle_factor), 1e-12)
    ratios = _axis_ratio_values(max_axis_ratio, n_angle_samples + 1)

    if n_free == 1:
        a = (volume / factor) ** (1.0 / 3.0)
        if _passes_bulk_length_filter((a, a, a), min_length, max_axis_ratio):
            results.append((a, a, a))

    elif n_free == 2:
        for c_over_a in ratios:
            a = (volume / (factor * c_over_a)) ** (1.0 / 3.0)
            c = c_over_a * a
            lengths = (a, a, c)
            if _passes_bulk_length_filter(lengths, min_length, max_axis_ratio):
                results.append(lengths)

    elif n_free == 3:
        for b_over_a in ratios:
            for c_over_a in ratios:
                a = (volume / (factor * b_over_a * c_over_a)) ** (1.0 / 3.0)
                lengths = (a, b_over_a * a, c_over_a * a)
                if _passes_bulk_length_filter(lengths, min_length, max_axis_ratio):
                    results.append(tuple(sorted(lengths)))
    return sorted(set(results))


def sample_lattice_for_system(system_name, volume, min_atom_spacing=1.0,
                               n_angle_samples=3, min_lattice_length=None,
                               max_axis_ratio=None):
    params = get_crystal_system_params(system_name)
    n_free_len = params["n_length_params"]
    n_free_ang = params["n_angle_params"]
    fixed_angles = params.get("fixed_angles", {})
    angle_range = params.get("angle_range", (60, 120))

    if n_free_ang == 0:
        angle_combos = [{}]
    elif n_free_ang == 1:
        ang = np.linspace(angle_range[0], angle_range[1], n_angle_samples + 2)[1:-1]
        angle_combos = [{"beta": a} for a in ang]
    elif n_free_ang == 3:
        ang = np.linspace(angle_range[0], angle_range[1], n_angle_samples + 2)[1:-1]
        angle_combos = [{"alpha": al, "beta": be, "gamma": ga}
                        for al, be, ga in itertools.product(ang, repeat=3)]
    else:
        angle_combos = [{}]

    candidates = []
    for ad in angle_combos:
        alpha = ad.get("alpha", fixed_angles.get("alpha", 90))
        beta  = ad.get("beta",  fixed_angles.get("beta",  90))
        gamma = ad.get("gamma", fixed_angles.get("gamma", 90))
        factor = angle_volume_factor(alpha, beta, gamma)
        if factor <= 0.0:
            continue

        length_combos = _sample_lengths_for_volume(
            n_free_len, volume, min_atom_spacing, n_angle_samples,
            min_lattice_length=min_lattice_length,
            max_axis_ratio=max_axis_ratio,
            angle_factor=factor)

        for (a, b, c) in length_combos:
            lattice = _build_lattice_matrix(a, b, c, alpha, beta, gamma)
            actual_vol = actual_lattice_volume(lattice)
            rel_error = (actual_vol - volume) / volume if volume else 0.0
            candidates.append({
                "lattice_matrix": lattice.tolist(),
                "lattice_type": system_name,
                "volume": round(actual_vol, 4),
                "crystal_system": system_name,
                "SamplingMethod": "length_angle_stratified",
                "TargetVolume": float(volume),
                "ActualVolume": actual_vol,
                "VolumeRelativeError": float(rel_error),
            })
    return candidates


# ============================================================
# Niggli reduction and deduplication
# ============================================================

def _standard_niggli_reduce(lattice_matrix):
    """Try optional standard-library Niggli reducers without requiring them."""
    lattice = np.array(lattice_matrix, dtype=float)

    try:
        import spglib

        if hasattr(spglib, "error") and hasattr(spglib.error, "OLD_ERROR_HANDLING"):
            spglib.error.OLD_ERROR_HANDLING = False
        reduced = spglib.niggli_reduce(lattice, eps=1e-5)
        if reduced is not None:
            reduced = np.array(reduced, dtype=float)
            if reduced.shape == (3, 3) and np.all(np.isfinite(reduced)):
                return reduced, "spglib"
    except Exception:
        pass

    try:
        from pymatgen.core import Lattice

        reduced = Lattice(lattice).get_niggli_reduced_lattice()
        reduced_matrix = np.array(reduced.matrix, dtype=float)
        if reduced_matrix.shape == (3, 3) and np.all(np.isfinite(reduced_matrix)):
            return reduced_matrix, "pymatgen"
    except Exception:
        pass

    return None, None


def _niggli_reduce(lattice_matrix, eps=1e-5):
    """Niggli reduction with overflow protection."""
    try:
        A = np.array(lattice_matrix, dtype=float)
        G = A @ A.T
        a, b, c = float(G[0, 0]), float(G[1, 1]), float(G[2, 2])
        xi   = 2.0 * float(G[1, 2])
        eta  = 2.0 * float(G[0, 2])
        zeta = 2.0 * float(G[0, 1])

        # Guard against extreme values
        for v in [a, b, c, xi, eta, zeta]:
            if not np.isfinite(v) or abs(v) > 1e12:
                return np.array(lattice_matrix)

        with np.errstate(all="raise"):
            for _ in range(1000):
                changed = False

                if a > b + eps:
                    a, b = b, a
                    eta, xi = xi, eta
                if b > c + eps:
                    b, c = c, b
                    xi, zeta = zeta, xi

                if abs(xi) > b + eps:
                    c = c + b - xi * np.sign(xi)
                    xi = xi - 2.0 * b * np.sign(xi)
                    changed = True

                if abs(eta) > a + eps:
                    c = c + a - eta * np.sign(eta)
                    xi = xi - zeta * np.sign(eta)
                    eta = eta - 2.0 * a * np.sign(eta)
                    changed = True

                if abs(zeta) > a + eps:
                    b = b + a - zeta * np.sign(zeta)
                    xi = xi - eta * np.sign(zeta)
                    zeta = zeta - 2.0 * a * np.sign(zeta)
                    changed = True

                if eta > b + eps or zeta > a + eps or xi > a + eps:
                    if eta > b + eps:
                        c = c + b - eta
                        xi = xi - zeta
                        eta = eta - 2.0 * b
                        changed = True
                    if zeta > a + eps:
                        b = b + a - zeta
                        xi = xi - eta
                        zeta = zeta - 2.0 * a
                        changed = True
                    if xi > a + eps:
                        b = b + a - xi
                        eta = eta - zeta
                        xi = xi - 2.0 * a
                        changed = True

                if eta < -eps or zeta < -eps or xi < -eps:
                    eta, zeta, xi = abs(eta), abs(zeta), abs(xi)
                    changed = True

                if not changed:
                    break

        G_red = np.array([
            [a, zeta / 2.0, eta / 2.0],
            [zeta / 2.0, b, xi / 2.0],
            [eta / 2.0, xi / 2.0, c],
        ])
        L = np.linalg.cholesky(G_red)
        return L.T
    except (OverflowError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.array(lattice_matrix)


def _permutation_canonical_tuple(lattice_matrix, length_tol, angle_tol):
    lattice = np.array(lattice_matrix, dtype=float)

    def _norm(v):
        return float(np.linalg.norm(v))

    def _angle(v1, v2):
        cos = np.dot(v1, v2) / (_norm(v1) * _norm(v2))
        cos = np.clip(cos, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def _quantized_tuple(basis):
        a = round(_norm(basis[0]) / length_tol) * length_tol
        b = round(_norm(basis[1]) / length_tol) * length_tol
        c = round(_norm(basis[2]) / length_tol) * length_tol
        alpha = round(_angle(basis[1], basis[2]) / angle_tol) * angle_tol
        beta = round(_angle(basis[0], basis[2]) / angle_tol) * angle_tol
        gamma = round(_angle(basis[0], basis[1]) / angle_tol) * angle_tol
        return (
            round(a, 4),
            round(b, 4),
            round(c, 4),
            round(alpha, 2),
            round(beta, 2),
            round(gamma, 2),
        )

    return min(
        _quantized_tuple(lattice[list(order)])
        for order in itertools.permutations(range(3))
    )


def _canonical_metadata(canon):
    return {
        "a": canon[0],
        "b": canon[1],
        "c": canon[2],
        "alpha": canon[3],
        "beta": canon[4],
        "gamma": canon[5],
    }


def niggli_key(lattice_matrix, length_tol=0.05, angle_tol=0.5, return_metadata=False):
    length_tol = _require_positive_real("length_tol", length_tol)
    angle_tol = _require_positive_real("angle_tol", angle_tol)

    standard, reducer = _standard_niggli_reduce(lattice_matrix)
    if standard is None:
        basis_for_key = _niggli_reduce(np.array(lattice_matrix, dtype=float))
        reducer = "local"
    else:
        basis_for_key = standard

    canon = _permutation_canonical_tuple(basis_for_key, length_tol, angle_tol)
    h = hashlib.sha256(json.dumps(canon).encode())
    key = h.hexdigest()[:12]
    metadata = {"reducer": reducer, "canonical": _canonical_metadata(canon)}

    if return_metadata:
        return key, metadata
    return key


def niggli_dedup(candidates, length_tol=0.05, angle_tol=0.5, return_rejected=False):
    seen = set()
    unique = []
    rejected = []
    for cand in candidates:
        lat = np.array(cand["lattice_matrix"], dtype=float)
        key, metadata = niggli_key(
            lat,
            length_tol=length_tol,
            angle_tol=angle_tol,
            return_metadata=True,
        )
        if key in seen:
            duplicate = dict(cand)
            duplicate["reason"] = "duplicate_niggli_key"
            duplicate["niggli_key"] = key
            duplicate["NiggliReducer"] = metadata["reducer"]
            duplicate["NiggliCanonical"] = metadata["canonical"]
            rejected.append(duplicate)
            continue
        seen.add(key)
        cand["niggli_key"] = key
        cand["NiggliReducer"] = metadata["reducer"]
        cand["NiggliCanonical"] = metadata["canonical"]
        unique.append(cand)
    if return_rejected:
        return unique, rejected
    return unique


# ============================================================
# Full enumeration pipeline
# ============================================================

def enumerate_lattices(V_min, V_max, volume_step_factor, min_atom_spacing,
                        crystal_systems, n_angle_samples=3,
                        length_tol=0.05, angle_tol=0.5,
                        min_lattice_length=None, max_axis_ratio=None,
                        return_rejected=False,
                        volume_tolerance_fraction=0.01):
    V_min, V_max = _require_finite_volume_bounds(V_min, V_max)
    if V_min <= 0 or V_max <= 0:
        raise ValueError("V_min and V_max must be positive")
    if V_min > V_max:
        raise ValueError("V_min must be <= V_max")
    volume_step_factor = _require_positive_real(
        "volume_step_factor",
        volume_step_factor,
    )
    min_atom_spacing = _require_positive_real("min_atom_spacing", min_atom_spacing)
    n_angle_samples = _require_positive_integer("n_angle_samples", n_angle_samples)
    length_tol = _require_positive_real("length_tol", length_tol)
    angle_tol = _require_positive_real("angle_tol", angle_tol)
    if max_axis_ratio is not None:
        max_axis_ratio = _require_min_real("max_axis_ratio", max_axis_ratio, 1.0)
    if min_lattice_length is not None:
        min_lattice_length = _require_positive_real(
            "min_lattice_length",
            min_lattice_length,
        )
    volume_tolerance_fraction = _require_non_negative_real(
        "volume_tolerance_fraction",
        volume_tolerance_fraction,
    )

    if crystal_systems == "all":
        systems = list(CRYSTAL_SYSTEMS.keys())
    elif isinstance(crystal_systems, str):
        systems = [crystal_systems]
    elif crystal_systems is None or not isinstance(crystal_systems, Iterable):
        raise ValueError("crystal_systems must be 'all' or an iterable of names")
    else:
        systems = list(crystal_systems)
    if not systems:
        raise ValueError("crystal_systems must not be empty")
    for sys_name in systems:
        get_crystal_system_params(sys_name)

    volumes = log_volume_grid(V_min, V_max, volume_step_factor)

    raw_candidates = []
    generated_count = 0
    rejected = []
    for vol in volumes:
        for sys_name in systems:
            cands = sample_lattice_for_system(
                system_name=sys_name, volume=vol,
                min_atom_spacing=min_atom_spacing,
                n_angle_samples=n_angle_samples,
                min_lattice_length=min_lattice_length,
                max_axis_ratio=max_axis_ratio)
            generated_count += len(cands)
            for cand in cands:
                reason = _volume_rejection_reason(
                    cand["ActualVolume"],
                    V_min,
                    V_max,
                    volume_tolerance_fraction,
                )
                if reason:
                    row = dict(cand)
                    row["reason"] = reason
                    rejected.append(row)
                else:
                    raw_candidates.append(cand)

    unique, duplicate_rejections = niggli_dedup(
        raw_candidates,
        length_tol=length_tol,
        angle_tol=angle_tol,
        return_rejected=True,
    )
    rejected.extend(duplicate_rejections)

    diagnostics = {
        "sampled_volume_count": len(volumes),
        "raw_candidate_count": generated_count,
        "accepted_pre_dedup_count": len(raw_candidates),
        "unique_candidate_count": len(unique),
        "rejected_candidate_count": len(rejected),
        "niggli_reducer": unique[0].get("NiggliReducer", "none") if unique else "none",
    }

    if return_rejected:
        return unique, rejected, diagnostics
    return unique
