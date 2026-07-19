import os
import csv
import math
from collections import defaultdict
from collections.abc import Mapping


def _ensure_finite(name, value):
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive_float(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    _ensure_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_percentile(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("percentile must be a number")
    _ensure_finite("percentile", value)
    if value < 0 or value > 100:
        raise ValueError("percentile must be between 0 and 100")
    return value


def _percentile(values, percentile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalise_column(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_path(path, config_root=None):
    if not path:
        raise ValueError("dataset path is required")
    path = os.fspath(path)
    if os.path.isabs(path) or not config_root:
        return path
    return os.path.join(config_root, path)


def _read_volume_per_atom_dataset(path):
    if not os.path.isfile(path):
        raise ValueError(f"dataset path does not exist: {path}")

    values = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("dataset must contain headers")
        field_lookup = {_normalise_column(field): field for field in reader.fieldnames}
        vpa_field = field_lookup.get("volumeperatom")
        volume_field = field_lookup.get("volume")
        natoms_field = (
            field_lookup.get("natoms")
            or field_lookup.get("atomnumber")
            or field_lookup.get("atomcount")
        )
        if not vpa_field and not (volume_field and natoms_field):
            raise ValueError("dataset must contain volume_per_atom or volume and n_atoms")

        for row in reader:
            try:
                if vpa_field:
                    value = float(row[vpa_field])
                else:
                    volume = float(row[volume_field])
                    n_atoms = float(row[natoms_field])
                    value = volume / n_atoms
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)

    if not values:
        raise ValueError("dataset contains no positive finite volume values")
    return values


def _resolve_volume_bound_from_dataset(config, config_root=None):
    dataset_path = _resolve_path(
        config.get("path") or config.get("dataset_path"),
        config_root=config_root,
    )
    percentile = _nonnegative_percentile(config.get("percentile", 95.0))
    safety_factor = _positive_float("safety_factor", config.get("safety_factor", 1.1))
    values = _read_volume_per_atom_dataset(dataset_path)
    percentile_value = _percentile(values, percentile)
    value = percentile_value * safety_factor
    return value, {
        "mode": "dataset",
        "path": dataset_path,
        "percentile": percentile,
        "safety_factor": safety_factor,
        "percentile_volume_per_atom": percentile_value,
        "sample_count": len(values),
    }


def _resolve_max_volume_from_dataset(config, config_root=None):
    return _resolve_volume_bound_from_dataset(config, config_root=config_root)


def resolve_max_volume_per_atom(max_volume_per_atom=50.0, config_root=None):
    """
    Resolve max volume per atom using either manual or dataset mode.
    """
    if isinstance(max_volume_per_atom, dict):
        config = dict(max_volume_per_atom)
        mode = str(config.get("mode", "manual")).strip().lower()
        mode = mode.replace("_", "")
        if mode == "manual":
            value = _positive_float(
                "max_volume_per_atom",
                config.get("value", config.get("max_volume_per_atom")),
            )
            metadata = {"mode": "manual", "value": value}
        elif mode == "dataset":
            value, metadata = _resolve_max_volume_from_dataset(config, config_root=config_root)
        else:
            raise ValueError("max_volume_per_atom mode must be manual or dataset")
    else:
        value = _positive_float("max_volume_per_atom", max_volume_per_atom)
        metadata = {"mode": "manual", "value": value}

    _ensure_finite("max_volume_per_atom", value)
    if value <= 0:
        raise ValueError("max_volume_per_atom must be positive")
    return float(value), metadata


def _resolve_min_volume_per_atom(config, config_root=None):
    config = {} if config is None else dict(config)
    combine_with_spacing = str(config.get("combine_with_spacing", "max")).strip().lower()
    if combine_with_spacing != "max":
        raise ValueError("combine_with_spacing must be max")
    mode = str(config.get("mode", "auto")).strip().lower()
    mode = mode.replace("_", "")
    if mode == "auto":
        return None, {"mode": "auto"}
    if mode == "manual":
        value = _positive_float(
            "min_volume_per_atom",
            config.get("value", config.get("min_volume_per_atom")),
        )
        return value, {"mode": "manual", "value": value}
    if mode == "dataset":
        config.setdefault("percentile", 5.0)
        config.setdefault("safety_factor", 0.85)
        return _resolve_volume_bound_from_dataset(config, config_root=config_root)
    raise ValueError("min volume_per_atom mode must be auto, manual, or dataset")


def _resolve_max_volume_per_atom_bound(config, fallback, config_root=None):
    if config is None:
        return resolve_max_volume_per_atom(fallback, config_root=config_root)
    config = dict(config)
    mode = str(config.get("mode", "manual")).strip().lower()
    mode = mode.replace("_", "")
    if mode == "manual":
        value = _positive_float(
            "max_volume_per_atom",
            config.get("value", config.get("max_volume_per_atom")),
        )
        return value, {"mode": "manual", "value": value}
    if mode == "dataset":
        config.setdefault("percentile", 95.0)
        config.setdefault("safety_factor", 1.1)
        return _resolve_volume_bound_from_dataset(config, config_root=config_root)
    raise ValueError("max volume_per_atom mode must be manual or dataset")


def _resolve_boron_bulk_policy(n_atoms, formula_dict, volume_policy):
    if volume_policy is None:
        return None
    if not isinstance(volume_policy, Mapping):
        raise ValueError("VolumePolicy must be a mapping")

    config = dict(volume_policy)
    mode = str(config.get("mode", "default")).strip().lower().replace("_", "")
    if mode in ("default", "auto", "none"):
        return None
    if mode != "boronbulk":
        raise ValueError("VolumePolicy mode must be default or boron_bulk")

    formula = {} if formula_dict is None else {
        str(specie): int(count)
        for specie, count in formula_dict.items()
    }
    if set(formula.keys()) != {"B"} or int(formula.get("B", 0)) != int(n_atoms):
        raise ValueError("boron_bulk volume policy requires pure B formula matching n_atoms")

    boron_config = dict(config.get("boron_bulk", {}))
    min_vpa = _positive_float(
        "boron_bulk.min_volume_per_atom",
        boron_config.get("min_volume_per_atom", 7.0),
    )
    max_vpa = _positive_float(
        "boron_bulk.max_volume_per_atom",
        boron_config.get("max_volume_per_atom", 8.05),
    )
    if min_vpa > max_vpa:
        raise ValueError("boron_bulk min_volume_per_atom must be <= max_volume_per_atom")

    metadata = {
        "mode": "boron_bulk",
        "min_volume_per_atom": min_vpa,
        "max_volume_per_atom": max_vpa,
    }
    return min_vpa * n_atoms, max_vpa * n_atoms, metadata


def compute_min_bond(effective_pair_dir):
    """
    Scan all pair CSVs in effective_pair_dir.
    For each pair, find the r_ij with the minimum average E_ij.
    Return the minimum of these optimal distances across all pairs.
    Returns None if no CSVs found.
    """
    optimal_distances = []

    if not os.path.isdir(effective_pair_dir):
        return None

    for fname in sorted(os.listdir(effective_pair_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(effective_pair_dir, fname)

        r_to_energies = defaultdict(list)
        with open(fpath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            required_columns = {'r_ij', 'E_ij'}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"{fname} missing required columns: r_ij, E_ij"
                )
            for row in reader:
                try:
                    r = float(row['r_ij'])
                    e = float(row['E_ij'])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(r) or not math.isfinite(e):
                    continue
                if r <= 0:
                    continue
                r_to_energies[r].append(e)

        if not r_to_energies:
            continue

        best_r = min(r_to_energies.keys(),
                     key=lambda r_val: sum(r_to_energies[r_val]) / len(r_to_energies[r_val]))
        optimal_distances.append(best_r)

    if not optimal_distances:
        return None
    return min(optimal_distances)


def compute_search_bounds(n_atoms, effective_pair_dir=None, r_min=2.0,
                           V_min=None, V_max=None, max_volume_per_atom=50.0,
                           volume_per_atom_bounds=None,
                           config_root=None, formula_dict=None,
                           volume_policy=None):
    """
    Compute V_min, V_max, and min_atom_spacing.

    Volume priority:
    1. Manual V_min/V_max from config (if both not None)
    2. boron_bulk VolumePolicy for pure B systems
    3. Existing default dataset / spacing fallback logic

    Returns dict: {n_atoms, V_min, V_max, min_atom_spacing, min_bond,
    volume_source, spacing_source, source}
    """
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    _ensure_finite("r_min", r_min)
    if r_min <= 0:
        raise ValueError("r_min must be positive")
    if volume_per_atom_bounds is not None and not isinstance(volume_per_atom_bounds, Mapping):
        raise ValueError("VolumePerAtomBounds must be a mapping")
    manual_volume = V_min is not None and V_max is not None
    policy_bounds = None if manual_volume else _resolve_boron_bulk_policy(
        n_atoms=n_atoms,
        formula_dict=formula_dict,
        volume_policy=volume_policy,
    )
    if manual_volume and volume_per_atom_bounds is not None:
        max_volume_per_atom_value = None
        max_volume_metadata = {"mode": "unused_manual_volume"}
    elif policy_bounds is not None:
        max_volume_per_atom_value = None
        max_volume_metadata = {"mode": "pending_boron_bulk_policy"}
    elif volume_per_atom_bounds is not None:
        max_volume_config = volume_per_atom_bounds.get("max")
        max_volume_per_atom_value, max_volume_metadata = _resolve_max_volume_per_atom_bound(
            max_volume_config,
            max_volume_per_atom,
            config_root=config_root,
        )
    else:
        max_volume_per_atom_value, max_volume_metadata = resolve_max_volume_per_atom(
            max_volume_per_atom=max_volume_per_atom,
            config_root=config_root,
        )
    if V_min is not None:
        _ensure_finite("V_min", V_min)
    if V_max is not None:
        _ensure_finite("V_max", V_max)

    if manual_volume:
        min_volume_per_atom = None
        min_volume_metadata = {"mode": "unused_manual_volume"}
    elif policy_bounds is not None:
        min_volume_per_atom = None
        min_volume_metadata = {"mode": "pending_boron_bulk_policy"}
    else:
        min_volume_per_atom = None
        min_volume_metadata = {"mode": "auto"}
        if volume_per_atom_bounds is not None:
            min_volume_config = volume_per_atom_bounds.get("min")
            min_volume_per_atom, min_volume_metadata = _resolve_min_volume_per_atom(
                min_volume_config,
                config_root=config_root,
            )

    # Determine min_atom_spacing
    min_bond = None
    if effective_pair_dir and os.path.isdir(effective_pair_dir):
        min_bond = compute_min_bond(effective_pair_dir)

    if min_bond is not None:
        min_atom_spacing = min_bond * 0.75
        spacing_source = 'effective_pair'
    else:
        min_atom_spacing = r_min * 0.75
        spacing_source = 'r_min_fallback'
    _ensure_finite("min_atom_spacing", min_atom_spacing)

    # Determine source and volumes
    if V_min is not None and V_max is not None:
        volume_source = 'manual'
        spacing_based_V_min = max(min_atom_spacing ** 3 * n_atoms * 0.5, 1.0)
        dataset_based_V_min = None
        V_min_source = "manual"
    elif policy_bounds is not None:
        V_min, V_max, policy_metadata = policy_bounds
        volume_source = "boron_bulk"
        spacing_based_V_min = max(min_atom_spacing ** 3 * n_atoms * 0.5, 1.0)
        dataset_based_V_min = None
        V_min_source = "boron_bulk"
        min_volume_per_atom = policy_metadata["min_volume_per_atom"]
        min_volume_metadata = policy_metadata
        max_volume_per_atom_value = policy_metadata["max_volume_per_atom"]
        max_volume_metadata = policy_metadata
    else:
        volume_source = 'auto'
        spacing_based_V_min = max(min_atom_spacing ** 3 * n_atoms * 0.5, 1.0)
        dataset_based_V_min = None
        V_min = spacing_based_V_min
        V_min_source = "spacing"
        if min_volume_per_atom is not None:
            dataset_based_V_min = min_volume_per_atom * n_atoms
            if dataset_based_V_min >= spacing_based_V_min:
                V_min = dataset_based_V_min
                V_min_source = min_volume_metadata["mode"]
        V_max = max_volume_per_atom_value * n_atoms

    _ensure_finite("V_min", V_min)
    _ensure_finite("V_max", V_max)
    if V_min <= 0:
        raise ValueError("V_min must be positive")
    if V_max <= 0:
        raise ValueError("V_max must be positive")
    if V_min > V_max:
        raise ValueError("V_min must be <= V_max")

    return {
        'n_atoms': n_atoms,
        'V_min': float(V_min),
        'V_max': float(V_max),
        'spacing_based_V_min': (
            None if spacing_based_V_min is None else float(spacing_based_V_min)
        ),
        'dataset_based_V_min': (
            None if dataset_based_V_min is None else float(dataset_based_V_min)
        ),
        'min_volume_per_atom': (
            None if min_volume_per_atom is None else float(min_volume_per_atom)
        ),
        'min_volume_per_atom_source': min_volume_metadata["mode"],
        'min_volume_per_atom_metadata': min_volume_metadata,
        'max_volume_per_atom': (
            None if max_volume_per_atom_value is None else float(max_volume_per_atom_value)
        ),
        'max_volume_per_atom_source': max_volume_metadata["mode"],
        'max_volume_per_atom_metadata': max_volume_metadata,
        'V_min_source': V_min_source,
        'min_atom_spacing': float(min_atom_spacing),
        'min_bond': None if min_bond is None else float(min_bond),
        'volume_source': volume_source,
        'spacing_source': spacing_source,
        'source': volume_source,
    }
