import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ip4ch.Generate_alpha_coefficients import (
    _alpha_cache_metadata,
    _resolve_pair_file_for_atoms,
    alpha_cache_path,
)
from multiprocessing_main import resolve_mlip_pair_paths


class PairFileResolutionTests(unittest.TestCase):
    def write_pair(self, root, name):
        path = Path(root) / name
        path.write_text("r_ij,E_ij\n1.5,0.0\n", encoding="utf-8")
        return path

    def test_resolves_canonical_pair_file_from_configured_pair_file_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configured = self.write_pair(tmpdir, "Al-N.csv")
            expected = self.write_pair(tmpdir, "N-Sc.csv")

            resolved = _resolve_pair_file_for_atoms(
                t1="Sc",
                t2="N",
                pair_file=str(configured),
            )

            self.assertEqual(Path(resolved), expected)

    def test_resolves_canonical_pair_file_from_pair_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self.write_pair(tmpdir, "N-Sc.csv")

            resolved = _resolve_pair_file_for_atoms(
                t1="Sc",
                t2="N",
                pair_dir=tmpdir,
            )

            self.assertEqual(Path(resolved), expected)

    def test_resolves_same_species_pair_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configured = self.write_pair(tmpdir, "Al-N.csv")
            expected = self.write_pair(tmpdir, "Al-Al.csv")

            resolved = _resolve_pair_file_for_atoms(
                t1="Al",
                t2="Al",
                pair_file=str(configured),
            )

            self.assertEqual(Path(resolved), expected)

    def test_missing_canonical_pair_file_fails_instead_of_reusing_wrong_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configured = self.write_pair(tmpdir, "Al-N.csv")

            with self.assertRaises(FileNotFoundError):
                _resolve_pair_file_for_atoms(
                    t1="Al",
                    t2="Sc",
                    pair_file=str(configured),
                )

    def test_generic_legacy_pair_file_still_works_as_single_file_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configured = self.write_pair(tmpdir, "pair.csv")

            resolved = _resolve_pair_file_for_atoms(
                t1="Al",
                t2="Sc",
                pair_file=str(configured),
            )

            self.assertEqual(Path(resolved), configured)

    def test_pair_dir_is_preferred_over_legacy_pair_file_config(self):
        class FakeConfig:
            resolved_config = {
                "MLIP": {
                    "pair_dir": "/data/AlScN_pair",
                    "pair_file": "/data/AlScN_pair/Al-N.csv",
                }
            }

        pair_file, pair_dir = resolve_mlip_pair_paths(FakeConfig())

        self.assertIsNone(pair_file)
        self.assertEqual(pair_dir, "/data/AlScN_pair")

    def test_legacy_pair_file_config_is_still_supported(self):
        class FakeConfig:
            resolved_config = {
                "MLIP": {
                    "pair_file": "/data/B_pair/B-B.csv",
                }
            }

        pair_file, pair_dir = resolve_mlip_pair_paths(FakeConfig())

        self.assertEqual(pair_file, "/data/B_pair/B-B.csv")
        self.assertIsNone(pair_dir)

    def test_alpha_cache_accepts_pathlike_pair_type(self):
        path = alpha_cache_path(
            t1="Al",
            t2="N",
            phase_name="AlNSc",
            pair_type=Path("AlScN_pair"),
            DataPath="/tmp/data",
            lattice_matrix=np.eye(3),
            r_min=1.5,
            r_max=5.0,
            grid_size=[2, 2, 2],
        )
        metadata = _alpha_cache_metadata(
            t1="Al",
            t2="N",
            phase_name="AlNSc",
            pair_type=Path("AlScN_pair"),
            lattice_matrix=np.eye(3),
            r_min=1.5,
            r_max=5.0,
            grid_size=[2, 2, 2],
        )

        self.assertIn("/AlScN_pair/", str(path))
        self.assertEqual(metadata["pair_type"], "AlScN_pair")
        json.dumps(metadata)


if __name__ == "__main__":
    unittest.main()
