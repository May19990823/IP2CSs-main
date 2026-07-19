import unittest

import numpy as np

from ip4ch.IntegerProgram import _iter_pair_energy_terms


class MultiSpeciesPairTermTests(unittest.TestCase):
    def test_same_species_pair_uses_one_orientation_per_site_pair(self):
        alpha = np.array([
            [0.0, 12.0, 13.0],
            [21.0, 0.0, 23.0],
            [31.0, 32.0, 0.0],
        ])

        terms = list(_iter_pair_energy_terms(
            atom_pair=("Al", "Al"),
            types=["Al", "N"],
            site_count=3,
            orbit_position_by_site=[0, 0, 1],
            alpha_matrix=alpha,
        ))

        self.assertEqual(terms, [
            (0, 0, 0, 0, 12.0),
            (0, 0, 0, 1, 13.0),
            (0, 0, 0, 1, 23.0),
        ])

    def test_different_species_pair_uses_both_site_orientations(self):
        alpha = np.array([
            [0.0, 12.0, 13.0],
            [21.0, 0.0, 23.0],
            [31.0, 32.0, 0.0],
        ])

        terms = list(_iter_pair_energy_terms(
            atom_pair=("Al", "N"),
            types=["Al", "N"],
            site_count=3,
            orbit_position_by_site=[0, 0, 1],
            alpha_matrix=alpha,
        ))

        self.assertEqual(terms, [
            (0, 0, 1, 0, 12.0),
            (1, 0, 0, 0, 12.0),
            (0, 0, 1, 1, 13.0),
            (1, 0, 0, 1, 13.0),
            (0, 0, 1, 1, 23.0),
            (1, 0, 0, 1, 23.0),
        ])

    def test_pair_with_missing_species_generates_no_terms(self):
        alpha = np.ones((2, 2))

        terms = list(_iter_pair_energy_terms(
            atom_pair=("Al", "Sc"),
            types=["Al", "N"],
            site_count=2,
            orbit_position_by_site=[0, 1],
            alpha_matrix=alpha,
        ))

        self.assertEqual(terms, [])


if __name__ == "__main__":
    unittest.main()
