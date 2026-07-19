import unittest
from unittest.mock import patch

import numpy as np

from ipcss.GridSymmetryCompatibility import (
    ensure_grid_space_group_compatibility,
    evaluate_grid_space_group_compatibility,
)
from ipcss.LatticeGridEngine import LatticeGridEngine


def minimal_engine_config(root="/tmp/boron_allotropes_test"):
    return {
        "base": {
            "root": root,
            "data": f"{root}/Data",
            "output": f"{root}/Results",
        },
        "input": {
            "entry": f"{root}/Data/Entry",
            "effective_pair": f"{root}/Data/EffectivePair/B_pair",
            "grids": f"{root}/Data/Grids",
        },
        "output": {
            "ip_results": f"{root}/Results/ip_result",
            "plot_results": f"{root}/Results/plot_result",
        },
        "MLIP": {
            "r_min": 1.5,
            "r_max": 5.0,
        },
        "LatticeEnumeration": {
            "crystal_systems": ["Tetragonal"],
            "volume_step_factor": 0.1,
            "max_volume_per_atom": {"mode": "manual", "value": 50.0},
            "min_lattice_length": 4,
            "max_axis_ratio": 4.0,
            "integer_step": 1,
            "rhombohedral_angles": [55, 60, 65, 70],
            "include_flat_hexagonal": True,
            "reduce_sequences": True,
        },
        "GridGeneration": {
            "safety_factor": 3.0,
            "max_grid_points": None,
            "min_grid_per_axis": 3,
            "max_grid_axis_ratio": 6.0,
            "warning_grid_points": 5000,
            "hard_grid_points": 12000,
        },
        "Deduplication": {
            "length_tolerance": 0.05,
            "angle_tolerance": 0.5,
        },
        "VolumeConstraints": {
            "V_min": None,
            "V_max": None,
        },
        "VolumePerAtomBounds": None,
        "VolumePolicy": {
            "mode": "boron_bulk",
            "boron_bulk": {
                "min_volume_per_atom": 7.0,
                "max_volume_per_atom": 8.05,
            },
        },
        "QualityFilter": {
            "min_rdf_resolution_ratio": 0.1,
            "max_rdf_resolution_ratio": 0.75,
            "max_spacing_anisotropy": 2.0,
            "min_coverage_estimate": 0.0,
            "reject_poor_quality": True,
        },
        "SpaceGroupPolicy": {
            "mode": "user_selected",
            "wyckoff_filter": True,
            "user_selected": {"Tetragonal": [139]},
        },
    }


class GridSymmetryCompatibilityTests(unittest.TestCase):
    def test_fast_mode_adjusts_translation_grid_without_exhaustive_grid_walk(self):
        with patch(
            "ipcss.GridSymmetryCompatibility._frac_positions",
            side_effect=AssertionError("exhaustive grid walk should not run in fast mode"),
        ):
            result = ensure_grid_space_group_compatibility(
                space_group=78,
                grid_size=[30, 30, 10],
                lattice_type="Tetragonal",
                lattice_matrix=np.diag([15.0, 15.0, 5.0]),
                hard_grid_points=12000,
                validation_mode="fast",
            )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["grid"], [30, 30, 12])
        self.assertEqual(result["method"], "fast_spacegroup_operation_closure")

    def test_fast_mode_adjusts_axis_mapping_without_exhaustive_grid_walk(self):
        with patch(
            "ipcss.GridSymmetryCompatibility._frac_positions",
            side_effect=AssertionError("exhaustive grid walk should not run in fast mode"),
        ):
            result = ensure_grid_space_group_compatibility(
                space_group=75,
                grid_size=[8, 10, 24],
                lattice_type="Tetragonal",
                lattice_matrix=np.diag([4.0, 5.0, 12.0]),
                hard_grid_points=12000,
                validation_mode="fast",
            )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["grid"], [10, 10, 24])
        self.assertEqual(result["method"], "fast_spacegroup_operation_closure")

    def test_fast_mode_handles_hexagonal_offset_grid_without_exhaustive_grid_walk(self):
        with patch(
            "ipcss.GridSymmetryCompatibility._frac_positions",
            side_effect=AssertionError("exhaustive grid walk should not run in fast mode"),
        ):
            result = ensure_grid_space_group_compatibility(
                space_group=194,
                grid_size=[8, 10, 10],
                lattice_type="Hexagonal",
                lattice_matrix=np.array([
                    [10.0, 0.0, 0.0],
                    [-5.0, 8.6602540378, 0.0],
                    [0.0, 0.0, 10.0],
                ]),
                hard_grid_points=12000,
                validation_mode="fast",
            )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["grid"], [10, 10, 10])
        self.assertEqual(result["method"], "fast_spacegroup_operation_closure")

    def test_detects_sg139_grid_that_cannot_represent_half_translation(self):
        result = evaluate_grid_space_group_compatibility(
            space_group=139,
            grid_size=[3, 3, 10],
            lattice_type="Tetragonal",
            lattice_matrix=np.diag([5.0, 5.0, 7.0]),
        )

        self.assertFalse(result["compatible"])
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual(result["reason"], "grid_space_group_incompatible")
        self.assertIn("failed_site", result)
        self.assertIn("nearest_frac", result)

    def test_adjusts_sg139_grid_until_symmetry_sites_land_on_grid(self):
        result = ensure_grid_space_group_compatibility(
            space_group=139,
            grid_size=[3, 3, 10],
            lattice_type="Tetragonal",
            lattice_matrix=np.diag([5.0, 5.0, 7.0]),
            hard_grid_points=12000,
        )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["original_grid"], [3, 3, 10])
        self.assertEqual(result["grid"], [4, 4, 10])
        self.assertGreater(result["adjustment_iterations"], 0)

    def test_adjusts_real_b160_tetragonal_grid_to_represent_three_quarter_positions(self):
        result = ensure_grid_space_group_compatibility(
            space_group=78,
            grid_size=[30, 30, 10],
            lattice_type="Tetragonal",
            lattice_matrix=np.diag([15.0, 15.0, 5.0]),
            hard_grid_points=12000,
        )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["grid"], [30, 30, 12])
        self.assertGreaterEqual(result["grid_points"], 10000)
        self.assertIn("large_grid_warning", result["warnings"])

    def test_rejects_adjusted_grid_above_hard_limit(self):
        result = ensure_grid_space_group_compatibility(
            space_group=78,
            grid_size=[30, 30, 10],
            lattice_type="Tetragonal",
            lattice_matrix=np.diag([15.0, 15.0, 5.0]),
            hard_grid_points=10000,
        )

        self.assertFalse(result["compatible"])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "grid_space_group_adjustment_exceeds_hard_limit")
        self.assertEqual(result["suggested_grid"], [30, 30, 12])

    def test_lattice_grid_engine_exposes_compatibility_resolution(self):
        engine = LatticeGridEngine(minimal_engine_config())

        result = engine._resolve_grid_space_group_compatibility(
            space_group=139,
            grid_size=[3, 3, 10],
            lattice_type="Tetragonal",
            lattice_matrix=np.diag([5.0, 5.0, 7.0]),
        )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["status"], "adjusted")
        self.assertEqual(result["grid"], [4, 4, 10])


if __name__ == "__main__":
    unittest.main()
