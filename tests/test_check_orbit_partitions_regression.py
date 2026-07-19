import unittest
from pathlib import Path

from tools.check_orbit_partitions import grid_size_from_filename


class CheckOrbitPartitionsRegressionTests(unittest.TestCase):
    def test_grid_size_from_reference_orbit_filename(self):
        self.assertEqual(
            grid_size_from_filename(Path("SG166_G[18, 18, 18]_ref_beta_B105.json")),
            [18, 18, 18],
        )


if __name__ == "__main__":
    unittest.main()
