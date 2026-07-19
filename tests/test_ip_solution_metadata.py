import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

try:
    import gurobipy as gb
except Exception:
    gb = None

from ip4ch.IntegerProgram import Allocate, gurobi_status_name


class FakeParams:
    def __init__(self):
        self.SolutionNumber = 0


class FakeVar:
    def __init__(self, name, values, params):
        self.varName = name
        self._values = list(values)
        self._params = params

    @property
    def Xn(self):
        return self._values[self._params.SolutionNumber]

    @property
    def x(self):
        return self._values[0]


class FakeModel:
    def __init__(self):
        self.params = FakeParams()
        self.Status = gb.GRB.TIME_LIMIT if gb is not None else 9
        self.status = self.Status
        self.SolCount = 2
        self.Runtime = 1.25
        self.ObjVal = -4.0
        self.objVal = -4.0
        self.ObjBound = -5.0
        self.MIPGap = 0.2
        self._vars = [
            FakeVar("B_0", [1.0, 0.0], self.params),
            FakeVar("B_1", [0.0, 1.0], self.params),
            FakeVar("B_2", [0.0, 0.0], self.params),
        ]

    def getVars(self):
        return self._vars


class IPSolutionMetadataTests(unittest.TestCase):
    def make_boron_allocator(self, root, grid_size, formula, pair_energy=0.0):
        root = Path(root)
        pair_file = root / "pair.csv"
        pair_file.write_text(f"r_ij,energy\n1.0,{pair_energy}\n", encoding="utf-8")
        grids_dir = root / "Data" / "Grids"
        grids_dir.mkdir(parents=True, exist_ok=True)

        return Allocate(
            ProgramRoot=str(root),
            DataDir=str(root / "Data"),
            OutputDir=str(root / "Output"),
            EntryDir=str(root / "Data" / "Entries"),
            GridsDir=str(grids_dir),
            EffectivePairDir=str(root / "Data" / "EffectivePair"),
            IPResultsDir=str(root / "Results" / "ip_result"),
            PlotDir=str(root / "Results" / "plot"),
            lattice=[2.0, 2.0, 2.0],
            lattice_matrix=np.diag([2.0, 2.0, 2.0]),
            lattice_type="Orthorhombic",
            grid_size=grid_size,
            full_formula_dict=formula,
            r_min=0.0,
            r_max=1.5,
            compound_type="test",
            pair_type="unit",
            pair_file=str(pair_file),
        )

    def test_orbit_cache_lock_creates_and_removes_lock_file(self):
        allocation = object.__new__(Allocate)
        with tempfile.TemporaryDirectory() as tmpdir:
            orbit_path = str(Path(tmpdir) / "SG1_G[1, 1, 1].json")
            with allocation._orbit_cache_lock(orbit_path, timeout=1.0, poll_interval=0.01):
                self.assertTrue(Path(f"{orbit_path}.lock").is_file())
            self.assertFalse(Path(f"{orbit_path}.lock").exists())

    def test_orbit_cache_lock_reclaims_stale_lock_file(self):
        allocation = object.__new__(Allocate)
        with tempfile.TemporaryDirectory() as tmpdir:
            orbit_path = str(Path(tmpdir) / "SG1_G[1, 1, 1].json")
            lock_path = Path(f"{orbit_path}.lock")
            stale_created_at = time.time() - 3600.0
            lock_path.write_text(
                json.dumps({
                    "pid": 999999,
                    "created_at": stale_created_at,
                    "orbit_path": orbit_path,
                }),
                encoding="utf-8",
            )

            with allocation._orbit_cache_lock(
                orbit_path,
                timeout=1.0,
                poll_interval=0.01,
                stale_after=0.01,
            ):
                replacement = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(replacement["orbit_path"], orbit_path)
                self.assertGreater(replacement["created_at"], stale_created_at)

            self.assertFalse(lock_path.exists())

    def test_orbit_cache_lock_creates_parent_directory(self):
        allocation = object.__new__(Allocate)
        with tempfile.TemporaryDirectory() as tmpdir:
            orbit_path = str(Path(tmpdir) / "missing" / "SG1_G[1, 1, 1].json")
            with allocation._orbit_cache_lock(orbit_path, timeout=1.0, poll_interval=0.01):
                self.assertTrue(Path(f"{orbit_path}.lock").is_file())
            self.assertFalse(Path(f"{orbit_path}.lock").exists())

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_gurobi_status_name_maps_known_and_unknown_codes(self):
        self.assertEqual(gurobi_status_name(gb.GRB.OPTIMAL), "OPTIMAL")
        self.assertEqual(gurobi_status_name(gb.GRB.TIME_LIMIT), "TIME_LIMIT")
        self.assertEqual(gurobi_status_name(999999), "UNKNOWN_999999")

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_solver_summary_contains_time_limit_gap_and_bound(self):
        allocation = object.__new__(Allocate)
        summary = allocation._solver_status_summary(FakeModel(), runtime=1.25)

        self.assertEqual(summary["gurobi_status_code"], int(gb.GRB.TIME_LIMIT))
        self.assertEqual(summary["gurobi_status_name"], "TIME_LIMIT")
        self.assertTrue(summary["has_incumbent"])
        self.assertFalse(summary["is_proven_optimal"])
        self.assertEqual(summary["solution_count_raw"], 2)
        self.assertEqual(summary["runtime_seconds"], 1.25)
        self.assertEqual(summary["best_objective"], -4.0)
        self.assertEqual(summary["objective_bound"], -5.0)
        self.assertTrue(math.isclose(summary["mip_gap"], 0.2))

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_selected_variables_are_read_from_each_pool_solution(self):
        allocation = object.__new__(Allocate)
        allocation.model = FakeModel()

        self.assertEqual(allocation._selected_variables_for_solution(0), ["B_0"])
        self.assertEqual(allocation._selected_variables_for_solution(1), ["B_1"])

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_equal_objective_pool_solutions_are_all_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocation = self.make_boron_allocator(
                root=tmpdir,
                grid_size=[2, 1, 1],
                formula={"B": 1},
            )
            result = allocation.optimize_symmetry_ase(
                group=1,
                PoolSolutions=2,
                TimeLimit=30,
                threads=1,
                verbose=False,
            )

        structures, runtime, best_obj, pool_objectives, selected_variables, is_optimal, status_code = result[:7]
        solver_summary = result[7]
        solution_records = result[8]
        skipped_solutions = result[9]

        self.assertEqual(len(structures), 2)
        self.assertEqual(pool_objectives, [0.0, 0.0])
        self.assertEqual(solver_summary["solution_count_written"], 2)
        self.assertEqual(solver_summary["solution_count_skipped"], 0)
        self.assertEqual(skipped_solutions, [])
        self.assertEqual(
            sorted(tuple(record["selected_variables"]) for record in solution_records),
            [("B_0",), ("B_1",)],
        )
        self.assertIn(tuple(selected_variables), [("B_0",), ("B_1",)])

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_high_objective_pool_solution_is_not_screened_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocation = self.make_boron_allocator(
                root=tmpdir,
                grid_size=[2, 1, 1],
                formula={"B": 2},
                pair_energy=1001.0,
            )
            result = allocation.optimize_symmetry_ase(
                group=1,
                PoolSolutions=1,
                TimeLimit=30,
                threads=1,
                verbose=False,
            )

        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[7]["solution_count_written"], 1)
        self.assertEqual(result[7]["solution_count_skipped"], 0)
        self.assertEqual(result[9], [])
        self.assertGreaterEqual(result[3][0], 1000.0)

    @unittest.skipIf(gb is None, "gurobipy is not available")
    def test_infeasible_model_returns_solver_status_without_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocation = self.make_boron_allocator(
                root=tmpdir,
                grid_size=[1, 1, 1],
                formula={"B": 2},
            )
            result = allocation.optimize_symmetry_ase(
                group=1,
                PoolSolutions=1,
                TimeLimit=30,
                threads=1,
                verbose=False,
            )

        self.assertEqual(result[0], [])
        self.assertEqual(result[3], [])
        self.assertEqual(result[4], [])
        self.assertFalse(result[5])
        self.assertEqual(result[7]["gurobi_status_name"], "INFEASIBLE")
        self.assertFalse(result[7]["has_incumbent"])
        self.assertEqual(result[7]["solution_count_written"], 0)


if __name__ == "__main__":
    unittest.main()
