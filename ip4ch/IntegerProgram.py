import json
import os
import time
from contextlib import contextmanager
import ase
import ase.io
from ip4ch.Generate_alpha_coefficients import get_energy, _resolve_pair_file_for_atoms
from ip4ch.compact_backend import (
    build_orbit_data,
    build_pair_coefficients,
    conflict_cover,
)
from ip4ch.generate_crystal_orbit import (
    build_orbit_cache_metadata,
    generate_orbit_dict,
    generate_reference_orbit_dict,
    orbit_cache_metadata_matches,
    orbit_cache_metadata_path,
)
from ip4ch.Generate_alpha_coefficients import generate_orthorhombic, generate_hexagonal
from ip4ch.orbit_utils import assert_valid_orbit_partition
import gurobipy as gb
from matscipy.neighbours import neighbour_list
import numpy as np


def _available_gurobi_status_names():
    names = [
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    ]
    status_names = {}
    for name in names:
        code = getattr(gb.GRB, name, None)
        if code is not None:
            status_names[int(code)] = name
    return status_names


GUROBI_STATUS_NAMES = _available_gurobi_status_names()


def gurobi_status_name(status_code):
    if status_code is None:
        return "UNKNOWN"
    code = int(status_code)
    return GUROBI_STATUS_NAMES.get(code, f"UNKNOWN_{code}")


def _safe_model_attr(model, attr_name, default=None):
    try:
        return getattr(model, attr_name)
    except (AttributeError, gb.GurobiError):
        return default


def _safe_float(value):
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return converted


def _iter_pair_energy_terms(atom_pair, types, site_count, orbit_position_by_site, alpha_matrix):
    """Yield QUBO/IP variable index pairs for one chemical pair and all site pairs."""
    if not ((atom_pair[0] in types) and (atom_pair[1] in types)):
        return

    j1 = types.index(atom_pair[0])
    j2 = types.index(atom_pair[1])

    for i1 in range(site_count):
        orbit_i1 = orbit_position_by_site[i1]
        for i2 in range(i1 + 1, site_count):
            orbit_i2 = orbit_position_by_site[i2]
            coefficient = alpha_matrix[i1, i2]
            yield j1, orbit_i1, j2, orbit_i2, coefficient
            if atom_pair[0] != atom_pair[1]:
                yield j2, orbit_i1, j1, orbit_i2, coefficient


class Allocate:
    def __init__(self, ProgramRoot, DataDir, OutputDir, EntryDir, GridsDir, EffectivePairDir, IPResultsDir,
                 PlotDir, lattice, lattice_matrix, lattice_type, grid_size, full_formula_dict, r_min, r_max,
                 compound_type, pair=None, pair_type='SCEP', pair_file=None, pair_dir=None):
        # self.phase = phase

        self.model = None
        self.ProgramRoot = ProgramRoot

        self.DataDir = DataDir

        self.OutputDir = OutputDir

        self.EntryDir = EntryDir

        self.GridsDir = GridsDir

        self.EffectivePairDir = EffectivePairDir

        self.IPResultsDir = IPResultsDir

        self.PlotDir = PlotDir

        self.lattice = lattice

        self.lattice_matrix = lattice_matrix

        self.lattice_type = lattice_type

        self.grid_size = grid_size

        self.full_formula_dict = full_formula_dict

        self.species = self.generate_species()

        self.phase_name = ''.join(i for i in self.species)

        self.r_min = r_min

        self.r_max = r_max

        self.compound_type = compound_type

        if pair is not None:
            self.pair = pair
        else:
            self.pair = self.generate_atom_pairs()

        self.pair_type = pair_type  # Other Choices： SP,AveEP

        self.pair_file = pair_file
        self.pair_dir = pair_dir

        if lattice_type in ["Orthorhombic", "Tetragonal", "Cubic", "Rhombohedral"]:
            self.frac_pos = generate_orthorhombic(ions_on_sides=grid_size)
        elif lattice_type in ["Hexagonal"]:
            self.frac_pos = generate_hexagonal(ions_on_sides=grid_size)

    def generate_atom_pairs(self):
        return tuple((self.species[i], self.species[j]) for i in range(len(self.species)) for j in
                     range(i, len(self.species)))

    def generate_species(self):
        return sorted([specie for specie, num in self.full_formula_dict.items()])

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def _orbit_cache_lock_is_stale(self, lock_path, stale_after):
        if stale_after is None or float(stale_after) <= 0:
            return False

        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            created_at = float(payload.get("created_at"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            try:
                created_at = os.path.getmtime(lock_path)
            except OSError:
                return False

        return time.time() - created_at > float(stale_after)

    def _remove_stale_orbit_cache_lock(self, lock_path, stale_after):
        if not self._orbit_cache_lock_is_stale(lock_path, stale_after):
            return False
        try:
            os.unlink(lock_path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    @contextmanager
    def _orbit_cache_lock(self, orbit_path, timeout=600.0, poll_interval=0.25, stale_after=7200.0):
        lock_path = f"{orbit_path}.lock"
        lock_dir = os.path.dirname(lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        start_time = time.time()
        acquired = False
        while not acquired:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({
                        "pid": os.getpid(),
                        "created_at": time.time(),
                        "orbit_path": orbit_path,
                    }, handle)
                acquired = True
            except FileExistsError:
                if self._remove_stale_orbit_cache_lock(lock_path, stale_after):
                    continue
                if time.time() - start_time > float(timeout):
                    raise TimeoutError(f"Timed out waiting for orbit cache lock: {lock_path}")
                time.sleep(float(poll_interval))

        try:
            yield
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

    def _backup_invalid_orbit_file(self, orbit_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{orbit_path}.invalid_{stamp}"
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{orbit_path}.invalid_{stamp}_{counter}"
            counter += 1
        os.replace(orbit_path, backup_path)
        metadata_path = orbit_cache_metadata_path(orbit_path)
        if os.path.exists(metadata_path):
            os.replace(metadata_path, orbit_cache_metadata_path(backup_path))
        return backup_path

    def _load_or_generate_valid_orbits(self, group, orbit_filename, symmetry_reference_file=None):
        orbit_path = os.path.join(self.GridsDir, orbit_filename)
        reference_path = None
        if symmetry_reference_file:
            reference_path = symmetry_reference_file
            if not os.path.isabs(reference_path):
                reference_path = os.path.join(self.ProgramRoot, reference_path)
        expected_metadata = build_orbit_cache_metadata(
            target_space_group=group,
            grid_size=self.grid_size,
            lattice_matrix=None if reference_path else self.lattice_matrix,
            reference_structure_file=reference_path,
        )

        with self._orbit_cache_lock(orbit_path):
            if os.path.exists(orbit_path):
                try:
                    with open(orbit_path, 'r') as f:
                        orbits = json.load(f)
                    assert_valid_orbit_partition(
                        orbits=orbits,
                        grid_size=self.grid_size,
                        context=orbit_filename,
                    )
                    metadata_ok, metadata_reason = orbit_cache_metadata_matches(
                        orbit_path=orbit_path,
                        expected_metadata=expected_metadata,
                    )
                    if not metadata_ok:
                        raise ValueError(f"orbit cache metadata mismatch: {metadata_reason}")
                    return orbits
                except (json.JSONDecodeError, ValueError) as exc:
                    backup_path = self._backup_invalid_orbit_file(orbit_path)
                    print(
                        f"Invalid cached orbit file {orbit_path}; "
                        f"backed up to {backup_path}; regenerating. Reason: {exc}",
                        flush=True,
                    )

            if symmetry_reference_file:
                generate_reference_orbit_dict(
                    target_space_group=group,
                    frac_pos_list=self.frac_pos,
                    grid_size=self.grid_size,
                    GridsDir=self.GridsDir,
                    reference_structure_file=reference_path,
                    orbit_filename=orbit_filename,
                )
            else:
                generate_orbit_dict(target_space_group=group,
                                    frac_pos_list=self.frac_pos,
                                    grid_size=self.grid_size, GridsDir=self.GridsDir,
                                    lattice_matrix=self.lattice_matrix)
            with open(orbit_path, 'r') as f:
                orbits = json.load(f)
            assert_valid_orbit_partition(
                orbits=orbits,
                grid_size=self.grid_size,
                context=orbit_filename,
            )
            metadata_ok, metadata_reason = orbit_cache_metadata_matches(
                orbit_path=orbit_path,
                expected_metadata=expected_metadata,
            )
            if not metadata_ok:
                raise ValueError(f"Generated orbit cache metadata mismatch: {metadata_reason}")
            return orbits

    def _find_min_distance_orbit_conflicts(self, o_pos, min_distance, tolerance=1e-8):
        # Variables are orbit-level: selecting one orbit occupies every equivalent
        # position in that orbit. If one orbit contains a short pair internally,
        # the whole orbit must be disabled because partial orbit filling would
        # lower the requested space-group symmetry.
        min_distance = float(min_distance)
        if min_distance <= 0:
            return set(), set()

        n_positions = len(o_pos)
        tmp_struc = ase.Atoms(
            symbols='H' * n_positions,
            scaled_positions=self.frac_pos,
            pbc=True,
            cell=self.lattice_matrix,
        )
        i_list, j_list, distances = neighbour_list(
            quantities='ijd',
            atoms=tmp_struc,
            cutoff=min_distance,
        )

        blocked_orbits = set()
        blocked_orbit_pairs = set()
        for i0, j0, distance in zip(i_list, j_list, distances):
            if float(distance) >= min_distance - tolerance:
                continue

            orbit_i = o_pos[int(i0)]
            orbit_j = o_pos[int(j0)]
            if orbit_i == orbit_j:
                blocked_orbits.add(orbit_i)
            else:
                blocked_orbit_pairs.add(tuple(sorted((orbit_i, orbit_j))))

        return blocked_orbits, blocked_orbit_pairs

    def _add_min_distance_constraints(self, model, Vars, blocked_orbits, blocked_orbit_pairs):
        constraint_count = 0
        type_count = len(Vars)

        for orbit_index in sorted(blocked_orbits):
            orbit_vars = [Vars[j][orbit_index] for j in range(type_count)]
            model.addConstr(
                gb.LinExpr([1.0] * len(orbit_vars), orbit_vars) <= 0,
                name=f"min_distance_block_orbit_{orbit_index}",
            )
            constraint_count += 1

        for orbit_i, orbit_j in sorted(blocked_orbit_pairs):
            pair_vars = (
                [Vars[j][orbit_i] for j in range(type_count)]
                + [Vars[j][orbit_j] for j in range(type_count)]
            )
            model.addConstr(
                gb.LinExpr([1.0] * len(pair_vars), pair_vars) <= 1,
                name=f"min_distance_orbits_{orbit_i}_{orbit_j}",
            )
            constraint_count += 1

        return constraint_count

    def _add_compact_min_distance_constraints(self, model, Vars, orbit_data):
        """Add blocked-orbit constraints and a clique cover of pair conflicts."""
        constraint_count = 0
        type_count = len(Vars)
        for original in sorted(orbit_data.blocked_original):
            variables = [Vars[j][original] for j in range(type_count)]
            model.addConstr(
                gb.LinExpr([1.0] * len(variables), variables) <= 0,
                name=f"min_distance_block_orbit_{original}",
            )
            constraint_count += 1

        cover = conflict_cover(orbit_data)
        for clique_index, clique in enumerate(cover.cliques):
            variables = []
            for active in clique:
                original = orbit_data.active_original[active]
                variables.extend(Vars[j][original] for j in range(type_count))
            model.addConstr(
                gb.LinExpr([1.0] * len(variables), variables) <= 1,
                name=f"min_distance_clique_{clique_index}",
            )
            constraint_count += 1

        for active_left, active_right in cover.residual_edges:
            original_left = orbit_data.active_original[active_left]
            original_right = orbit_data.active_original[active_right]
            variables = (
                [Vars[j][original_left] for j in range(type_count)]
                + [Vars[j][original_right] for j in range(type_count)]
            )
            model.addConstr(
                gb.LinExpr([1.0] * len(variables), variables) <= 1,
                name=f"min_distance_edge_{original_left}_{original_right}",
            )
            constraint_count += 1
        return constraint_count, cover

    @staticmethod
    def _add_compact_pair_objective(
            energy, Vars, types, atom_pair, orbit_data, coefficients):
        left_type = types.index(atom_pair[0])
        right_type = types.index(atom_pair[1])
        same_species = atom_pair[0] == atom_pair[1]

        if same_species:
            for active, coefficient in enumerate(coefficients.linear):
                if coefficient == 0.0:
                    continue
                original = orbit_data.active_original[active]
                energy.add(Vars[left_type][original] * float(coefficient))

        for (active_left, active_right), coefficient in coefficients.quadratic.items():
            original_left = orbit_data.active_original[active_left]
            original_right = orbit_data.active_original[active_right]
            energy.add(
                Vars[left_type][original_left]
                * Vars[right_type][original_right]
                * float(coefficient)
            )
            if not same_species:
                energy.add(
                    Vars[right_type][original_left]
                    * Vars[left_type][original_right]
                    * float(coefficient)
                )

    def _normalize_orbit_size_signature(self, orbit_size_signature):
        if orbit_size_signature in (None, {}, []):
            return None
        if not isinstance(orbit_size_signature, dict):
            raise ValueError("orbit-size signature must be a mapping of orbit_size -> count")

        signature = {}
        for raw_size, raw_count in orbit_size_signature.items():
            size = int(raw_size)
            count = int(raw_count)
            if size <= 0:
                raise ValueError("orbit-size signature sizes must be positive")
            if count < 0:
                raise ValueError("orbit-size signature counts must be non-negative")
            if count == 0:
                continue
            signature[size] = signature.get(size, 0) + count

        if not signature:
            return None

        occupied_atoms = sum(size * count for size, count in signature.items())
        expected_atoms = sum(int(count) for count in self.full_formula_dict.values())
        if occupied_atoms != expected_atoms:
            raise ValueError(
                f"orbit-size signature occupies {occupied_atoms} atoms, "
                f"but formula requires {expected_atoms}"
            )
        return dict(sorted(signature.items()))

    def _add_orbit_size_signature_constraints(self, model, Vars, orb_size, orbit_size_signature):
        signature = self._normalize_orbit_size_signature(orbit_size_signature)
        if signature is None:
            return 0

        type_count = len(Vars)
        constraints_added = 0
        for size, required_count in signature.items():
            matching_orbits = [i for i, actual_size in enumerate(orb_size) if int(actual_size) == size]
            if len(matching_orbits) < required_count:
                raise ValueError(
                    f"orbit-size signature requires {required_count} orbits of size {size}, "
                    f"but only {len(matching_orbits)} are available"
                )

            orbit_vars = []
            for orbit_index in matching_orbits:
                orbit_vars.extend(Vars[j][orbit_index] for j in range(type_count))
            model.addConstr(
                gb.LinExpr([1.0] * len(orbit_vars), orbit_vars) == required_count,
                name=f"orbit_size_signature_size_{size}",
            )
            constraints_added += 1

        return constraints_added

    def _solver_status_summary(self, model, runtime):
        raw_status = _safe_model_attr(model, "Status", _safe_model_attr(model, "status"))
        status_code = int(raw_status) if raw_status is not None else None
        sol_count = int(_safe_model_attr(model, "SolCount", 0) or 0)
        best_objective = _safe_float(
            _safe_model_attr(model, "ObjVal", _safe_model_attr(model, "objVal"))
        )
        objective_bound = _safe_float(_safe_model_attr(model, "ObjBound"))
        mip_gap = _safe_float(_safe_model_attr(model, "MIPGap"))

        return {
            "gurobi_status_code": status_code,
            "gurobi_status_name": gurobi_status_name(status_code),
            "has_incumbent": sol_count > 0,
            "is_proven_optimal": status_code == int(gb.GRB.OPTIMAL),
            "solution_count_raw": sol_count,
            "solution_count_written": 0,
            "solution_count_skipped": 0,
            "runtime_seconds": _safe_float(runtime),
            "best_objective": best_objective,
            "objective_bound": objective_bound,
            "mip_gap": mip_gap,
        }

    def _selected_variables_for_solution(self, solution_index):
        solution_index = int(solution_index)
        self.model.params.SolutionNumber = solution_index
        selected = []
        for variable in self.model.getVars():
            value = _safe_model_attr(variable, "Xn")
            if value is None and solution_index == 0:
                value = _safe_model_attr(variable, "x")
            if value is not None and float(value) > 0.5:
                selected.append(variable.varName)
        return selected

    def _validate_solution_atoms(self, atoms, solution_index, tolerance=1e-8):
        expected_counts = {str(element): int(count) for element, count in self.full_formula_dict.items()}
        actual_counts = {}
        for symbol in atoms.get_chemical_symbols():
            actual_counts[symbol] = actual_counts.get(symbol, 0) + 1

        if actual_counts != expected_counts:
            raise ValueError(
                f"atom count mismatch for solution {solution_index}: "
                f"expected {expected_counts} total={sum(expected_counts.values())}, "
                f"got {actual_counts} total={len(atoms)}"
            )

        seen = {}
        for atom_index, frac in enumerate(atoms.get_scaled_positions(wrap=True)):
            wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
            wrapped[np.isclose(wrapped, 1.0, atol=tolerance)] = 0.0
            wrapped[np.isclose(wrapped, 0.0, atol=tolerance)] = 0.0
            key = tuple(np.round(wrapped, 10))
            if key in seen:
                raise ValueError(
                    f"duplicate fractional coordinates for solution {solution_index}: "
                    f"atoms {seen[key]} and {atom_index} at {key}"
                )
            seen[key] = atom_index

        min_distance = float(self.r_min)
        if min_distance > 0.0 and len(atoms) > 1:
            i_list, j_list, distances = neighbour_list(
                quantities='ijd',
                atoms=atoms,
                cutoff=min_distance,
            )
            violating_distances = [
                float(distance)
                for i0, j0, distance in zip(i_list, j_list, distances)
                if int(i0) < int(j0) and float(distance) < min_distance - tolerance
            ]
            if violating_distances:
                d_min = min(violating_distances)
                raise ValueError(
                    f"minimum distance {d_min:.10g} below r_min {min_distance:.10g} "
                    f"for solution {solution_index}"
                )

    def solution_to_Atoms(self, i, orbits):
        """
        Converts the solution to an ASE Atoms object.
        """
        self.model.params.SolutionNumber = i

        grid_positions = self.frac_pos
        symbols = ''
        positions = []

        try:
            for v in self.model.getVars():
                if v.Xn > 0.5:
                    t, o = v.varName.split(sep='_')
                    positions.append(grid_positions[int(o)])
                    symbols += t

                    # Add orbit positions
                    for pos in orbits[o]:
                        positions.append(grid_positions[pos])
                        symbols += t

            # Validate positions
            if len(positions) == 0:
                raise ValueError("Positions list is empty. Please check the input data.")

            if any(len(pos) != 3 for pos in positions):
                raise ValueError("Each position should have 3 coordinates. Please check the positions.")

            atoms = ase.Atoms(symbols=symbols, scaled_positions=positions, cell=self.lattice_matrix, pbc=True)
            self._validate_solution_atoms(atoms, solution_index=i)
            return atoms

        except Exception as e:
            raise RuntimeError(f"Error in `solution_to_Atoms` for solution {i}: {str(e)}")

    def optimize_symmetry_ase(self, group=None, PoolSolutions=10, TimeLimit=0, threads=None,
                              verbose=True, write_model=False, model_file=None,
                              orbit_size_signature=None, symmetry_reference_file=None,
                              backend="legacy_dense", mip_gap=None, mip_focus=None,
                              seed=None, pool_search_mode=None):

        backend = str(backend)
        if backend not in {"legacy_dense", "compact_quadratic"}:
            raise ValueError(
                "backend must be 'legacy_dense' or 'compact_quadratic'"
            )

        variable_list = []
        N = 1
        for size in self.grid_size:
            N *= size
        T = len(self.full_formula_dict)
        # Loading the crystal orbit file
        orbit_filename = f'SG{group}_G{self.grid_size}.json'
        if symmetry_reference_file:
            reference_stem = os.path.splitext(os.path.basename(symmetry_reference_file))[0]
            orbit_filename = f'SG{group}_G{self.grid_size}_ref_{reference_stem}.json'
        orbits = self._load_or_generate_valid_orbits(
            group=group,
            orbit_filename=orbit_filename,
            symmetry_reference_file=symmetry_reference_file,
        )
        # orb_key是等效轨道的索引
        orb_key = list(orbits.keys())

        # orb_size为列表，其中每个元素对应了每个等效轨道对应的等效点数目
        orb_size = [len(orbits[k]) + 1 for k in orb_key]

        o_pos = []
        for i in range(N):  # i是位置索引
            for orb, pos in orbits.items():  # orb代表某个等效轨道的索引，pos是该等效轨道索引对应的值，为一个列表，需要遍历
                if int(orb) == i:  # 表示该位置索引恰好为等效轨道的索引
                    o_pos.append(orb_key.index(orb))
                    break
                if i in pos:  # 表示该位置是某个等效轨道内的元素
                    o_pos.append(orb_key.index(orb))
                    break
        if len(o_pos) != N:
            raise ValueError(f"Orbit mapping incomplete: expected {N} positions, got {len(o_pos)}")
        # O表示在该空间群下等效轨道的数目
        O = len(orb_key)

        types = self.species
        # counts 表示每种元素的原子个数，类型为列表,example:[12,4]
        counts = [self.full_formula_dict[element] for element in types]

        # Create a new model
        m = gb.Model('Crystal Structure Prediction')

        # Create Variables
        Vars = [[] for _ in range(T)]
        for i in range(O):
            tmp_var = []
            for j in range(T):
                Vars[j] += [m.addVar(vtype=gb.GRB.BINARY, name=str(types[j]) + '_' + orb_key[i])]
                variable_list.append(str(types[j]) + '_' + orb_key[i])
                tmp_var += [Vars[j][-1]]

            if i == 0:
                # m.addConstr(gb.LinExpr([ip4ch.0] * T, tmp_var) == ip4ch, 'first_orbit_has_ion')
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, 'first_orbit_has_ion')
            else:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, f'one_per_orbit_{i}')

        for j in range(T):
            tmp_expr = gb.LinExpr()
            for i in range(O):
                tmp_expr.add(Vars[j][i], orb_size[i])
            m.addConstr(tmp_expr == counts[j], name=f"number_of_ions_type_{j}")

        blocked_orbits, blocked_orbit_pairs = self._find_min_distance_orbit_conflicts(
            o_pos=o_pos,
            min_distance=self.r_min,
        )
        orbit_data = None
        compact_cover = None
        if backend == "compact_quadratic":
            orbit_data = build_orbit_data(
                grid_size=self.grid_size,
                orbits=orbits,
                blocked_orbits=blocked_orbits,
                blocked_orbit_pairs=blocked_orbit_pairs,
            )
            min_distance_constraint_count, compact_cover = (
                self._add_compact_min_distance_constraints(
                    model=m, Vars=Vars, orbit_data=orbit_data
                )
            )
        else:
            min_distance_constraint_count = self._add_min_distance_constraints(
                model=m,
                Vars=Vars,
                blocked_orbits=blocked_orbits,
                blocked_orbit_pairs=blocked_orbit_pairs,
            )
        orbit_signature_constraint_count = self._add_orbit_size_signature_constraints(
            model=m,
            Vars=Vars,
            orb_size=orb_size,
            orbit_size_signature=orbit_size_signature,
        )

        print("Variables and constraints were generated")
        print(f"Model backend: {backend}")
        print(f"Min-distance constraints were generated: {min_distance_constraint_count} (r_min={self.r_min})")
        if compact_cover is not None:
            print(
                "Compact conflict cover: "
                f"cliques={len(compact_cover.cliques)} "
                f"residual_edges={len(compact_cover.residual_edges)}"
            )
        if orbit_size_signature:
            print(
                "Orbit-size signature constraints were generated: "
                f"{orbit_signature_constraint_count} {self._normalize_orbit_size_signature(orbit_size_signature)}"
            )
        print("*" * 30, 'Now Initializing the Objective Function', '*' * 30)

        # Create Objective Function
        energy = gb.QuadExpr()

        for atom_pair in self.pair:
            if not ((atom_pair[0] in types) and (atom_pair[1] in types)):
                continue

            if backend == "compact_quadratic":
                resolved_pair_file = _resolve_pair_file_for_atoms(
                    atom_pair[0], atom_pair[1],
                    pair_file=self.pair_file, pair_dir=self.pair_dir,
                )
                coefficients, coefficient_source = build_pair_coefficients(
                    orbit_data=orbit_data,
                    lattice_type=self.lattice_type,
                    grid_size=self.grid_size,
                    lattice_matrix=self.lattice_matrix,
                    fractional_positions=self.frac_pos,
                    r_min=self.r_min,
                    r_max=self.r_max,
                    pair_file=resolved_pair_file,
                )
                self._add_compact_pair_objective(
                    energy, Vars, types, atom_pair, orbit_data, coefficients
                )
                print(
                    f"Compact coefficients for {atom_pair}: "
                    f"source={coefficient_source} "
                    f"linear={int(np.count_nonzero(coefficients.linear))} "
                    f"quadratic={len(coefficients.quadratic)}"
                )
            else:
                alpha_matrix = get_energy(
                    t1=atom_pair[0], t2=atom_pair[1], phase_name=self.phase_name,
                    lattice=self.lattice, lattice_matrix=self.lattice_matrix,
                    r_min=self.r_min, r_max=self.r_max,
                    grid_size=self.grid_size, frac_pos=self.frac_pos,
                    pair_type=self.pair_type, DataPath=self.DataDir,
                    compound_type=self.compound_type, pair_file=self.pair_file,
                    pair_dir=self.pair_dir,
                )
                for j_left, orbit_left, j_right, orbit_right, coefficient in _iter_pair_energy_terms(
                        atom_pair=atom_pair,
                        types=types,
                        site_count=N,
                        orbit_position_by_site=o_pos,
                        alpha_matrix=alpha_matrix):
                    energy.add(Vars[j_left][orbit_left] * Vars[j_right][orbit_right] * coefficient)
                del alpha_matrix

        print("Objective function was generated!")

        m.setObjective(energy, gb.GRB.MINIMIZE)
        self.model = m

        if TimeLimit > 0:
            m.params.TimeLimit = TimeLimit

        if not verbose:
            m.params.OutputFlag = 0

        if PoolSolutions > 1:
            m.params.PoolSolutions = PoolSolutions
        if pool_search_mode is not None:
            m.params.PoolSearchMode = int(pool_search_mode)
        elif PoolSolutions > 1:
            m.params.PoolSearchMode = 2

        if mip_gap is not None:
            m.params.MIPGap = float(mip_gap)
        if mip_focus is not None:
            m.params.MIPFocus = int(mip_focus)
        if seed is not None:
            m.params.Seed = int(seed)

        nodefile_dir = os.environ.get("GUROBI_NODEFILE_DIR", "/tmp/gurobi_nodes")
        os.makedirs(nodefile_dir, exist_ok=True)
        m.Params.NodefileStart = 1
        m.Params.NodefileDir = nodefile_dir
        if threads is not None:
            m.Params.Threads = threads

        if write_model:
            if model_file is None:
                model_file = "model.lp"
            model_dir = os.path.dirname(model_file)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            print(f"Writing model file: {model_file}")
            m.write(model_file)

        # m.optimize(my_callback)
        m.optimize()
        runtime = m.Runtime
        solver_summary = self._solver_status_summary(m, runtime)
        solver_summary["model_backend"] = backend
        status_code = solver_summary["gurobi_status_code"]
        is_optimal = bool(solver_summary["is_proven_optimal"])
        best_obj = solver_summary["best_objective"]

        res = []
        obj = []
        solution_records = []
        skipped_solutions = []

        if status_code == int(gb.GRB.CUTOFF) and int(m.SolCount) == 0:
            print("Cutoff! No solution with negative energy.")

        if int(m.SolCount) > 0:
            print("There are", m.SolCount, "solutions")
            for solution_index in range(int(m.SolCount)):
                m.params.SolutionNumber = solution_index
                pool_obj = _safe_float(_safe_model_attr(m, "PoolObjVal"))
                selected_for_solution = self._selected_variables_for_solution(solution_index)
                record = {
                    "solution_index": int(solution_index),
                    "pool_obj": pool_obj,
                    "selected_variables": selected_for_solution,
                    "status": "candidate",
                }

                try:
                    atoms = self.solution_to_Atoms(solution_index, orbits)
                except RuntimeError as exc:
                    record["status"] = "skipped_invalid_structure"
                    record["error"] = str(exc)
                    skipped_solutions.append(record)
                    print(f"Skipping invalid solution {solution_index}: {exc}", flush=True)
                    continue

                record["status"] = "written"
                record["atom_count"] = int(len(atoms))
                record["chemical_formula"] = atoms.get_chemical_formula()
                res.append(atoms)
                obj.append(pool_obj)
                solution_records.append(record)

        solver_summary["solution_count_written"] = int(len(res))
        solver_summary["solution_count_skipped"] = int(len(skipped_solutions))

        if solution_records:
            var_list = list(solution_records[0]["selected_variables"])
        else:
            var_list = []

        if var_list:
            print(" ".join(var_list))
        if best_obj is not None:
            print("Minimal energy via optimizer: %g" % best_obj)
            print(f"Minimal per atom energy via optimizer:{best_obj / sum(counts)}")
        print(
            "Gurobi status: "
            f"{solver_summary['gurobi_status_name']} "
            f"(code={solver_summary['gurobi_status_code']}), "
            f"raw_solutions={solver_summary['solution_count_raw']}, "
            f"written={solver_summary['solution_count_written']}, "
            f"skipped={solver_summary['solution_count_skipped']}, "
            f"gap={solver_summary['mip_gap']}"
        )

        if PoolSolutions <= 1 and len(res) > 1:
            res = res[:1]
            obj = obj[:1]
            solution_records = solution_records[:1]
            solver_summary["solution_count_written"] = int(len(res))

        return (
            res,
            runtime,
            best_obj,
            obj,
            var_list,
            is_optimal,
            status_code,
            solver_summary,
            solution_records,
            skipped_solutions,
        )


def generate_lattice_matrix(lattice, lattice_type):
    """
    Generate a lattice matrix based on the lattice parameters and lattice type.
    """
    lattice_matrix = np.zeros((3, 3))
    if lattice_type in ["Orthorhombic", "Tetragonal", "Cubic"]:
        for i in range(3):
            lattice_matrix[i][i] = lattice[i]
    elif lattice_type in ["Hexagonal"]:
        a = lattice[0]
        c = lattice[2]
        lattice_matrix[0][0] = a
        lattice_matrix[1][0] = -0.5 * a
        lattice_matrix[1][1] = np.sqrt(3) / 2 * a
        lattice_matrix[2][2] = c
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")
    return lattice_matrix


if __name__ == '__main__':
    print('')
    """lattice = [6, 6, 5]
    lattice_type = "Hexagonal"
    lattice_matrix = generate_lattice_matrix(lattice_type=lattice_type, lattice=lattice)
    allocate = Allocate(lattice=lattice, lattice_matrix=lattice_matrix, lattice_type=lattice_type,
                        grid_size=[8, 8, 4],
                        full_formula_dict={"Al": 2, "Ti": 6}, r_min=2.0, r_max=5.0)
    print(allocate.pair)
    results, runtime, incumbent, obj, v, per_atom_energy = allocate.optimize_symmetry_ase(group=str(163))
    subdir_path = ''
    for i in range(len(results)):
        ase.io.write(filename=f'trial{i}.vasp', images=results[i], format='vasp')"""
