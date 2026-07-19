import os
import csv
import json
from collections import Counter
import numpy as np

from ipcss.SearchBounds import compute_search_bounds, resolve_max_volume_per_atom
from ipcss.IntegerLatticeEnumerator import enumerate_integer_lattices
from ipcss.GridGenerator import generate_grid
from ipcss.GridSymmetryCompatibility import (
    DEFAULT_HARD_GRID_POINTS,
    DEFAULT_WARNING_GRID_POINTS,
    ensure_grid_space_group_compatibility,
)
from ipcss.CoverageValidator import validate_coverage
from ipcss.WyckoffFilter import filter_space_groups_by_wyckoff


class LatticeGridEngine:
    """Physics-driven lattice enumeration and adaptive grid generation."""

    CRYSTAL_SYSTEM_NAMES = [
        "Cubic",
        "Tetragonal",
        "Orthorhombic",
        "Hexagonal",
        "Rhombohedral",
    ]

    def __init__(self, config):
        self.config = config
        self._get = self._dict_get if isinstance(config, dict) else config.get

        self.ProgramRoot = self._get("base", "root")
        self.EntryDir    = self._get("input", "entry")
        self.EffectivePairDir = self._get("input", "effective_pair")

        self.r_min = float(self._get("MLIP", "r_min"))
        self.r_max = float(self._get("MLIP", "r_max"))

        self.crystal_systems     = self._get("LatticeEnumeration", "crystal_systems")
        self.volume_step_factor  = float(self._get("LatticeEnumeration", "volume_step_factor"))
        self.max_volume_per_atom = self._get("LatticeEnumeration", "max_volume_per_atom")
        self.min_lattice_length  = self._optional_float("LatticeEnumeration", "min_lattice_length")
        self.max_axis_ratio      = self._optional_float("LatticeEnumeration", "max_axis_ratio")
        self.integer_step = self._optional_int("LatticeEnumeration", "integer_step", 1)
        self.rhombohedral_angles = self._optional_value(
            "LatticeEnumeration",
            "rhombohedral_angles",
            [55, 60, 65, 70],
        )
        self.include_flat_hexagonal = self._optional_bool(
            "LatticeEnumeration",
            "include_flat_hexagonal",
            True,
        )
        self.reduce_lattice_sequences = self._optional_bool(
            "LatticeEnumeration",
            "reduce_sequences",
            True,
        )

        self.safety_factor   = float(self._get("GridGeneration", "safety_factor"))
        self.grid_r_min = self._optional_float("GridGeneration", "r_min")
        if self.grid_r_min is None:
            self.grid_r_min = self.r_min
        self.max_grid_points = self._optional_int("GridGeneration", "max_grid_points")
        self.min_grid_per_axis = self._optional_int("GridGeneration", "min_grid_per_axis", 1)
        self.max_grid_axis_ratio = self._optional_float("GridGeneration", "max_grid_axis_ratio")
        self.warning_grid_points = self._optional_int(
            "GridGeneration",
            "warning_grid_points",
            DEFAULT_WARNING_GRID_POINTS,
        )
        self.hard_grid_points = self._optional_int(
            "GridGeneration",
            "hard_grid_points",
            self.max_grid_points if self.max_grid_points is not None else DEFAULT_HARD_GRID_POINTS,
        )
        self.grid_compatibility_validation_mode = str(self._optional_value(
            "GridGeneration",
            "compatibility_validation_mode",
            "auto",
        )).strip().lower()

        self.length_tol = float(self._get("Deduplication", "length_tolerance"))
        self.angle_tol  = float(self._get("Deduplication", "angle_tolerance"))

        V_min_raw = self._get("VolumeConstraints", "V_min")
        V_max_raw = self._get("VolumeConstraints", "V_max")
        self.V_min_manual = float(V_min_raw) if V_min_raw is not None else None
        self.V_max_manual = float(V_max_raw) if V_max_raw is not None else None
        try:
            self.volume_per_atom_bounds = self._get("VolumePerAtomBounds")
        except (KeyError, TypeError, AttributeError):
            self.volume_per_atom_bounds = None
        self.volume_policy = self._optional_section("VolumePolicy", None)

        self.min_rdf_ratio = float(self._get("QualityFilter", "min_rdf_resolution_ratio"))
        self.min_coverage   = float(self._get("QualityFilter", "min_coverage_estimate"))
        self.max_rdf_ratio = self._optional_float("QualityFilter", "max_rdf_resolution_ratio")
        self.max_spacing_anisotropy = self._optional_float("QualityFilter", "max_spacing_anisotropy")
        self.reject_poor_quality = self._optional_bool("QualityFilter", "reject_poor_quality", True)

        self.space_group_policy = self._optional_value(
            "SpaceGroupPolicy",
            "mode",
            "full_range",
        )
        self.user_selected_space_groups = self._optional_value(
            "SpaceGroupPolicy",
            "user_selected",
            {},
        )
        self.wyckoff_filter_enabled = self._optional_bool(
            "SpaceGroupPolicy",
            "wyckoff_filter",
            True,
        )
        self._validate_config()

    def _dict_get(self, section, key=None):
        val = self.config[section]
        return val if key is None else val[key]

    def _optional_value(self, section, key, default=None):
        try:
            value = self._get(section, key)
        except (KeyError, TypeError, AttributeError):
            return default
        return default if value is None else value

    def _optional_section(self, section, default=None):
        try:
            if isinstance(self.config, dict):
                value = self._get(section)
            else:
                value = self.config[section]
        except (KeyError, TypeError, AttributeError):
            return default
        return default if value is None else value

    def _optional_float(self, section, key, default=None):
        value = self._optional_value(section, key, default)
        return None if value is None else float(value)

    def _optional_int(self, section, key, default=None):
        value = self._optional_value(section, key, default)
        return None if value is None else int(value)

    def _optional_bool(self, section, key, default=False):
        value = self._optional_value(section, key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y")
        return bool(value)

    # ----------------------------------------------------------------
    def _validate_config(self):
        if self.volume_step_factor <= 0:
            raise ValueError("volume_step_factor must be positive")
        if self.integer_step <= 0:
            raise ValueError("integer_step must be positive")
        if not self.rhombohedral_angles:
            raise ValueError("rhombohedral_angles must not be empty")
        if self.r_min <= 0:
            raise ValueError("r_min must be positive")
        if self.r_max <= 0:
            raise ValueError("r_max must be positive")
        self._validate_max_volume_per_atom_config()
        if self.safety_factor <= 0:
            raise ValueError("safety_factor must be positive")
        if self.volume_per_atom_bounds is not None and not isinstance(self.volume_per_atom_bounds, dict):
            raise ValueError("VolumePerAtomBounds must be a mapping")
        if self.volume_policy is not None and not hasattr(self.volume_policy, "get"):
            raise ValueError("VolumePolicy must be a mapping")
        if self.min_grid_per_axis <= 0:
            raise ValueError("min_grid_per_axis must be positive")
        if self.max_grid_points is not None and self.max_grid_points < int(self.min_grid_per_axis) ** 3:
            raise ValueError("max_grid_points must be >= min_grid_per_axis ** 3")
        if self.warning_grid_points is not None and self.warning_grid_points <= 0:
            raise ValueError("warning_grid_points must be positive")
        if self.hard_grid_points is not None and self.hard_grid_points <= 0:
            raise ValueError("hard_grid_points must be positive")
        if self.grid_compatibility_validation_mode not in ("auto", "fast", "exhaustive"):
            raise ValueError("compatibility_validation_mode must be auto, fast, or exhaustive")
        if (
            self.warning_grid_points is not None
            and self.hard_grid_points is not None
            and self.warning_grid_points > self.hard_grid_points
        ):
            raise ValueError("warning_grid_points must be <= hard_grid_points")
        if self.length_tol <= 0:
            raise ValueError("length_tolerance must be positive")
        if self.angle_tol <= 0:
            raise ValueError("angle_tolerance must be positive")
        if self.min_rdf_ratio < 0:
            raise ValueError("min_rdf_resolution_ratio must be non-negative")
        if self.min_coverage < 0:
            raise ValueError("min_coverage_estimate must be non-negative")
        if self.V_min_manual is not None and self.V_max_manual is not None:
            if self.V_min_manual > self.V_max_manual:
                raise ValueError("V_min must be <= V_max")
        if self.space_group_policy not in ("representative_subset", "full_range", "user_selected"):
            raise ValueError("unknown space group policy mode")
        if not self.crystal_systems:
            raise ValueError("crystal_systems must not be empty")
        if self.space_group_policy == "user_selected":
            if not isinstance(self.user_selected_space_groups, dict):
                raise ValueError("user_selected space groups must be a mapping")
            for system in self._configured_crystal_systems():
                selected = self.user_selected_space_groups.get(system)
                if not selected:
                    raise ValueError(
                        "user_selected space groups must include a non-empty list for %s"
                        % system
                    )

    @staticmethod
    def _max_volume_mode(max_volume_per_atom):
        if isinstance(max_volume_per_atom, dict):
            return str(max_volume_per_atom.get("mode", "manual")).strip().lower().replace("_", "")
        return "manual"

    def _validate_max_volume_per_atom_config(self):
        mode = self._max_volume_mode(self.max_volume_per_atom)
        if mode in ("manual", "dataset"):
            resolve_max_volume_per_atom(
                self.max_volume_per_atom,
                config_root=self.ProgramRoot,
            )
            return
        raise ValueError("max_volume_per_atom mode must be manual or dataset")

    @staticmethod
    def _formula_name(formula_dict):
        return "".join("%s%d" % (s, int(n)) for s, n in sorted(formula_dict.items()))

    @staticmethod
    def _phase_name(species):
        return "".join(sorted(str(s) for s in species))

    @staticmethod
    def _expand_formula_dict(reduced_formula_dict, multiplicity):
        return {
            str(s): int(n) * int(multiplicity)
            for s, n in reduced_formula_dict.items()
        }

    def _formula_jobs(self, stoichiometric_dict_list, multiplicities):
        if multiplicities is None:
            multiplicities = [1]

        jobs = []
        for reduced in stoichiometric_dict_list:
            reduced_clean = {str(s): int(n) for s, n in reduced.items()}
            for mult in multiplicities:
                mult = int(mult)
                if mult <= 0:
                    raise ValueError("multiplicity must be positive")
                full = self._expand_formula_dict(reduced_clean, mult)
                jobs.append({
                    "reduced_formula_dict": dict(sorted(reduced_clean.items())),
                    "full_formula_dict": dict(sorted(full.items())),
                    "multiplicity": mult,
                    "Multiplicity": mult,
                    "PhaseName": self._phase_name(reduced_clean.keys()),
                    "ReducedFormula": self._formula_name(reduced_clean),
                    "FullFormula": self._formula_name(full),
                })
        return jobs

    def _configured_crystal_systems(self):
        if self.crystal_systems == "all":
            return list(self.CRYSTAL_SYSTEM_NAMES)
        if isinstance(self.crystal_systems, str):
            return [self.crystal_systems]
        return list(self.crystal_systems)

    @staticmethod
    def _parse_generate_entries_args(args, kwargs):
        kwargs = dict(kwargs)
        entry_dir = kwargs.pop("EntryDir", None)
        name = kwargs.pop("name", None)
        stoichiometric_dict_list = kwargs.pop("stoichiometric_dict_list", None)
        multiplicities = kwargs.pop("multiplicities", None)
        if "Multiplicities" in kwargs:
            if multiplicities is not None:
                raise TypeError("multiplicities and Multiplicities cannot both be provided")
            multiplicities = kwargs.pop("Multiplicities")
        dry_run = kwargs.pop("dry_run", False)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError("unexpected keyword argument(s): %s" % unexpected)

        args = list(args)
        if args and len(args) >= 4 and isinstance(args[1], str):
            if entry_dir is None:
                entry_dir = args[0]
            if name is None:
                name = args[1]
            if multiplicities is None:
                multiplicities = args[2]
            if stoichiometric_dict_list is None:
                stoichiometric_dict_list = args[3]
            args = args[4:]
        else:
            if args and name is None:
                name = args.pop(0)
            if args and stoichiometric_dict_list is None:
                stoichiometric_dict_list = args.pop(0)
            if args and multiplicities is None:
                multiplicities = args.pop(0)
            if args:
                dry_run = args.pop(0)

        if args:
            raise TypeError("too many positional arguments")
        if name is None:
            raise TypeError("name is required")
        if stoichiometric_dict_list is None:
            raise TypeError("stoichiometric_dict_list is required")

        return str(entry_dir) if entry_dir is not None else None, name, stoichiometric_dict_list, multiplicities, dry_run

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(k): LatticeGridEngine._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [LatticeGridEngine._json_safe(v) for v in value]
        return value

    @staticmethod
    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                LatticeGridEngine._json_safe(payload),
                f,
                indent=4,
                ensure_ascii=False,
            )

    @staticmethod
    def _write_csv(path, rows):
        rows = [LatticeGridEngine._json_safe(row) for row in rows]
        fieldnames = sorted({key for row in rows for key in row.keys()})
        if not fieldnames:
            fieldnames = ["message"]
            rows = [{"message": "no rows"}]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                })

    @staticmethod
    def _grid_rejection_reason(grid_status, quality):
        if grid_status in ("anisotropic", "too_many_points"):
            return "grid_%s" % grid_status
        return None

    def _grid_resolution(self, lattice_matrix, grid_size):
        lattice = np.asarray(lattice_matrix, dtype=float)
        lengths = np.array([np.linalg.norm(lattice[i]) for i in range(3)])
        spacings = [
            float(lengths[i] / int(grid_size[i]))
            for i in range(3)
        ]
        return round(max(spacings), 4)

    def _resolve_grid_space_group_compatibility(
            self,
            space_group,
            grid_size,
            lattice_type,
            lattice_matrix):
        return ensure_grid_space_group_compatibility(
            space_group=space_group,
            grid_size=grid_size,
            lattice_type=lattice_type,
            lattice_matrix=lattice_matrix,
            warning_grid_points=self.warning_grid_points,
            hard_grid_points=self.hard_grid_points,
            max_grid_axis_ratio=self.max_grid_axis_ratio,
            validation_mode=self.grid_compatibility_validation_mode,
        )

    def generate_entries(self, *args, **kwargs):
        entry_dir, name, stoichiometric_dict_list, multiplicities, dry_run = (
            self._parse_generate_entries_args(args, kwargs)
        )
        output_root = entry_dir or self.EntryDir
        entry_dir_phase = os.path.join(output_root, name)
        os.makedirs(entry_dir_phase, exist_ok=True)
        summaries = []

        for job in self._formula_jobs(stoichiometric_dict_list, multiplicities):
            sto_dict = dict(sorted(job["full_formula_dict"].items()))
            n_atoms = sum(sto_dict.values())

            # Step 1: bounds
            print("[SearchBounds] %d atoms ..." % n_atoms)
            bounds = compute_search_bounds(
                n_atoms=n_atoms,
                effective_pair_dir=self.EffectivePairDir,
                r_min=self.grid_r_min,
                V_min=self.V_min_manual,
                V_max=self.V_max_manual,
                max_volume_per_atom=self.max_volume_per_atom,
                volume_per_atom_bounds=self.volume_per_atom_bounds,
                config_root=self.ProgramRoot,
                formula_dict=sto_dict,
                volume_policy=self.volume_policy,
            )
            print("  V=[%.1f, %.1f] A^3  spacing=%.4f A  source=%s" % (
                bounds["V_min"], bounds["V_max"],
                bounds["min_atom_spacing"], bounds.get("volume_source", bounds.get("source"))))

            # Step 2: lattices
            print("[IntegerLatticeEnumerator] systems=%s ..." % self.crystal_systems)
            lattices, rejected_lattices, diagnostics = enumerate_integer_lattices(
                V_min=bounds["V_min"],
                V_max=bounds["V_max"],
                crystal_systems=self.crystal_systems,
                min_lattice_length=int(self.min_lattice_length or 4),
                integer_step=int(self.integer_step),
                rhombohedral_angles=self.rhombohedral_angles,
                include_flat_hexagonal=self.include_flat_hexagonal,
                reduce_sequences=self.reduce_lattice_sequences,
                max_axis_ratio=self.max_axis_ratio,
                return_rejected=True,
            )
            print("  %d integer lattices after sequence reduction" % len(lattices))

            # Step 3: grids + validate
            entry_data = []
            accepted_lattice_rows = []
            rejected_rows = list(rejected_lattices)
            entry_idx = 1
            grid_status_counts = Counter()
            quality_status_counts = Counter()
            grid_compatibility_status_counts = Counter()
            space_group_filter_cache = {}

            for lat in lattices:
                lat_matrix = np.array(lat["lattice_matrix"])

                grid_size, resolution, grid_status, grid_diagnostics = generate_grid(
                    lattice_matrix=lat_matrix,
                    r_min=self.grid_r_min,
                    safety_factor=self.safety_factor,
                    max_points=self.max_grid_points,
                    min_grid_per_axis=self.min_grid_per_axis,
                    max_grid_axis_ratio=self.max_grid_axis_ratio,
                    return_diagnostics=True,
                )
                grid_status_counts[grid_status] += 1

                quality = validate_coverage(
                    lattice_matrix=lat_matrix,
                    grid_size=grid_size,
                    r_min=self.grid_r_min,
                    r_max=self.r_max,
                    min_grid_per_axis=self.min_grid_per_axis,
                    max_grid_axis_ratio=self.max_grid_axis_ratio,
                    max_rdf_resolution_ratio=self.max_rdf_ratio,
                    max_spacing_anisotropy=self.max_spacing_anisotropy,
                )
                quality_status_counts[quality["status"]] += 1

                rejection_reason = self._grid_rejection_reason(grid_status, quality)
                if quality["rdf_resolution_ratio"] < self.min_rdf_ratio:
                    rejection_reason = "rdf_resolution_ratio_below_minimum"
                if quality["coverage_estimate"] < self.min_coverage:
                    rejection_reason = "coverage_estimate_below_minimum"
                if self.reject_poor_quality and quality["status"] == "poor":
                    rejection_reason = rejection_reason or "poor_discretization_quality"

                candidate_row = {
                    **lat,
                    "Grid Size": [int(x) for x in grid_size],
                    "GridResolution": resolution,
                    "GridStatus": grid_status,
                    "GridDiagnostics": grid_diagnostics,
                    "DiscretizationQuality": quality,
                }

                if rejection_reason:
                    rejected = dict(candidate_row)
                    rejected["reason"] = rejection_reason
                    rejected_rows.append(rejected)
                    continue

                sg_source_list = self._space_groups(lat["lattice_type"])
                sg_list = list(sg_source_list)
                if self.wyckoff_filter_enabled:
                    cache_key = lat["lattice_type"]
                    if cache_key not in space_group_filter_cache:
                        kept_sg, rejected_sg = filter_space_groups_by_wyckoff(
                            sg_source_list,
                            sto_dict,
                            return_rejected=True,
                        )
                        space_group_filter_cache[cache_key] = {
                            "input_space_groups": list(sg_source_list),
                            "kept_space_groups": kept_sg,
                            "rejected_space_groups": rejected_sg,
                        }
                    sg_list = space_group_filter_cache[cache_key]["kept_space_groups"]

                candidate_row["WyckoffFilterEnabled"] = bool(self.wyckoff_filter_enabled)
                candidate_row["SpaceGroupsBeforeWyckoff"] = len(sg_source_list)
                candidate_row["SpaceGroupsAfterWyckoff"] = len(sg_list)
                accepted_lattice_rows.append(candidate_row)

                for sg in sg_list:
                    grid_compatibility = self._resolve_grid_space_group_compatibility(
                        space_group=sg,
                        grid_size=grid_size,
                        lattice_type=lat["lattice_type"],
                        lattice_matrix=lat_matrix,
                    )
                    grid_compatibility_status_counts[grid_compatibility["status"]] += 1

                    if grid_compatibility["status"] == "rejected":
                        rejected = dict(candidate_row)
                        rejected["Space Group"] = int(sg)
                        rejected["reason"] = grid_compatibility["reason"]
                        rejected["GridCompatibility"] = grid_compatibility
                        rejected_rows.append(rejected)
                        continue

                    entry_grid_size = [int(x) for x in grid_compatibility["grid"]]
                    entry_resolution = self._grid_resolution(lat_matrix, entry_grid_size)
                    entry_quality = validate_coverage(
                        lattice_matrix=lat_matrix,
                        grid_size=entry_grid_size,
                        r_min=self.grid_r_min,
                        r_max=self.r_max,
                        min_grid_per_axis=self.min_grid_per_axis,
                        max_grid_axis_ratio=self.max_grid_axis_ratio,
                        max_rdf_resolution_ratio=self.max_rdf_ratio,
                        max_spacing_anisotropy=self.max_spacing_anisotropy,
                    )
                    entry_data.append({
                        "Index": entry_idx,
                        "PhaseName": job["PhaseName"],
                        "ReducedFormula": job["ReducedFormula"],
                        "FullFormula": job["FullFormula"],
                        "Multiplicity": job["multiplicity"],
                        "Chemical Stoichiometric": job["ReducedFormula"],
                        "Full Formula Dictionary": sto_dict,
                        "Atom Number": n_atoms,
                        "Full Formula": job["FullFormula"],
                        "Lattice Type": lat["lattice_type"],
                        "Lattice": lat["lattice_matrix"],
                        "Grid Size": entry_grid_size,
                        "Original Grid Size": [int(x) for x in grid_size],
                        "Space Group": int(sg),
                        "LatticeNiggliKey": lat["niggli_key"],
                        "SamplingMethod": lat["SamplingMethod"],
                        "TargetVolume": lat["TargetVolume"],
                        "ActualVolume": lat["ActualVolume"],
                        "VolumeRelativeError": lat["VolumeRelativeError"],
                        "GridResolution": entry_resolution,
                        "DiscretizationQuality": entry_quality,
                        "GridCompatibility": grid_compatibility,
                    })
                    entry_idx += 1

            # Step 4: write
            formula = job["FullFormula"]
            entries_path = os.path.join(entry_dir_phase, "%s.json" % formula)
            candidates_path = os.path.join(entry_dir_phase, "%s_lattice_candidates.csv" % formula)
            rejected_path = os.path.join(entry_dir_phase, "%s_rejected_lattices.csv" % formula)
            summary_path = os.path.join(entry_dir_phase, "%s_summary.json" % formula)

            self._write_csv(candidates_path, accepted_lattice_rows)
            self._write_csv(rejected_path, rejected_rows)
            stale_solver_entries_removed = False
            if dry_run and os.path.exists(entries_path):
                os.remove(entries_path)
                stale_solver_entries_removed = True

            wyckoff_filter_summary = {
                lattice_type: {
                    "input_count": len(data["input_space_groups"]),
                    "kept_count": len(data["kept_space_groups"]),
                    "rejected_count": len(data["rejected_space_groups"]),
                    "kept_space_groups": [int(x) for x in data["kept_space_groups"]],
                    "rejected_space_groups": [
                        int(row["space_group"])
                        for row in data["rejected_space_groups"]
                    ],
                }
                for lattice_type, data in sorted(space_group_filter_cache.items())
            }

            summary = {
                **job,
                "n_atoms": n_atoms,
                "bounds": bounds,
                "dry_run": bool(dry_run),
                "solver_entries_written": not bool(dry_run),
                "stale_solver_entries_removed": stale_solver_entries_removed,
                "lattice_diagnostics": diagnostics,
                "accepted_lattice_count": len(accepted_lattice_rows),
                "rejected_lattice_count": len(rejected_rows),
                "expected_entry_count": len(entry_data),
                "space_group_policy": self.space_group_policy,
                "wyckoff_filter_enabled": bool(self.wyckoff_filter_enabled),
                "wyckoff_filter_summary": wyckoff_filter_summary,
                "grid_status_counts": dict(grid_status_counts),
                "quality_status_counts": dict(quality_status_counts),
                "grid_compatibility_status_counts": dict(grid_compatibility_status_counts),
                "rejection_reasons": dict(Counter(
                    row.get("reason", "unknown") for row in rejected_rows
                )),
            }
            self._write_json(summary_path, summary)

            if not dry_run:
                self._write_json(entries_path, entry_data)

            if dry_run:
                print("[Output] dry-run manifests -> %s (solver entries not written)" % summary_path)
            else:
                print("[Output] %d entries -> %s" % (len(entry_data), entries_path))
            summaries.append(summary)

        return summaries

    def _space_groups(self, lattice_type):
        full = {
            "Triclinic": [1, 2],
            "Monoclinic": list(range(3, 16)),
            "Orthorhombic": list(range(16, 75)),
            "Tetragonal": list(range(75, 143)),
            "Hexagonal": [
                143, 144, 145, 147,
                149, 150, 151, 152, 153, 154,
                156, 157, 158, 159,
                162, 163, 164, 165,
                *range(168, 195),
            ],
            "Rhombohedral": [146, 148, 155, 160, 161, 166, 167],
            "Cubic": list(range(195, 231)),
        }
        representative = {
            "Triclinic": [1, 2],
            "Monoclinic": [3, 5, 12, 15],
            "Orthorhombic": [16, 19, 25, 47, 62, 71],
            "Tetragonal": [75, 87, 99, 123, 139],
            "Hexagonal": [143, 156, 164, 186, 194],
            "Rhombohedral": [146, 148, 155, 166],
            "Cubic": [195, 216, 221, 225, 227],
        }
        if self.space_group_policy == "full_range":
            return full.get(lattice_type, [1])
        if self.space_group_policy == "user_selected":
            selected = self.user_selected_space_groups.get(lattice_type)
            return [int(x) for x in selected] if selected else []
        return representative.get(lattice_type, [1])
