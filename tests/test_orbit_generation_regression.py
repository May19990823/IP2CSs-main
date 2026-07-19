import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ip4ch import generate_crystal_orbit as orbit_module
from ip4ch.Generate_alpha_coefficients import generate_hexagonal
from ip4ch.orbit_utils import assert_valid_orbit_partition
from ip4ch.generate_crystal_orbit import (
    generate_fractional_coordiantion,
    generate_orbit_dict,
)


def orbit_sizes(orbits):
    return sorted(len(members) + 1 for members in orbits.values())


def hexagonal_lattice_matrix(a=5.0, c=7.0):
    return np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, np.sqrt(3.0) / 2.0 * a, 0.0],
        [0.0, 0.0, c],
    ], dtype=float)


class OrbitGenerationRegressionTests(unittest.TestCase):
    def test_hexagonal_offset_grid_spglib_branch_uses_actual_fractional_positions(self):
        grid = [3, 3, 4]
        frac_pos = generate_hexagonal(ions_on_sides=grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            orbits, total_sites = generate_orbit_dict(
                target_space_group=191,
                frac_pos_list=frac_pos,
                grid_size=grid,
                GridsDir=tmpdir,
                lattice_matrix=hexagonal_lattice_matrix(a=5.0, c=7.0),
            )

        assert_valid_orbit_partition(
            orbits=orbits,
            grid_size=grid,
            context="SG191 hexagonal offset grid [3, 3, 4]",
        )
        self.assertEqual(sum(len(site_group) for site_group in total_sites), 36)
        self.assertEqual(
            sum(len(members) + 1 for members in orbits.values()),
            36,
        )
        self.assertEqual(len(orbits), 9)
        self.assertEqual(orbit_sizes(orbits), [1, 1, 2, 2, 2, 4, 6, 6, 12])

    def test_sg1_with_lattice_matrix_generates_single_site_orbits(self):
        grid = [3, 3, 3]
        frac_pos, _ = generate_fractional_coordiantion(grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            orbits, _ = generate_orbit_dict(
                target_space_group=1,
                frac_pos_list=frac_pos,
                grid_size=grid,
                GridsDir=tmpdir,
                lattice_matrix=np.diag([5.0, 6.0, 7.0]),
            )

            self.assertEqual(len(orbits), 27)
            self.assertEqual(orbit_sizes(orbits), [1] * 27)

            metadata_path = Path(tmpdir) / "SG1_G[3, 3, 3].json.meta"
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["target_space_group"], 1)
            self.assertEqual(metadata["grid_size"], grid)

    def test_target_space_group_controls_orbits_with_lattice_matrix(self):
        grid = [3, 3, 3]
        frac_pos, _ = generate_fractional_coordiantion(grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            sg1_orbits, _ = generate_orbit_dict(
                target_space_group=1,
                frac_pos_list=frac_pos,
                grid_size=grid,
                GridsDir=tmpdir,
                lattice_matrix=np.diag([5.0, 6.0, 7.0]),
            )
            sg47_orbits, _ = generate_orbit_dict(
                target_space_group=47,
                frac_pos_list=frac_pos,
                grid_size=grid,
                GridsDir=tmpdir,
                lattice_matrix=np.diag([5.0, 6.0, 7.0]),
            )

        self.assertNotEqual(orbit_sizes(sg1_orbits), orbit_sizes(sg47_orbits))
        self.assertEqual(orbit_sizes(sg1_orbits), [1] * 27)
        self.assertEqual(orbit_sizes(sg47_orbits), [1, 2, 2, 2, 4, 4, 4, 8])

    def test_debug_demo_uses_hardcoded_small_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = orbit_module.run_debug_demo(grids_dir=tmpdir)

            self.assertEqual(summary["grid_size"], [3, 3, 3])
            cases = {case["label"]: case for case in summary["cases"]}
            self.assertEqual(cases["p1_singletons"]["orbit_count"], 27)
            self.assertEqual(cases["p1_singletons"]["orbit_size_histogram"], {1: 27})
            self.assertEqual(cases["sg47_orthorhombic_demo"]["orbit_count"], 8)


if __name__ == "__main__":
    unittest.main()
