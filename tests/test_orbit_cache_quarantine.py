import tempfile
import unittest
from pathlib import Path

from tools.quarantine_orbit_cache import collect_cache_files, quarantine_orbit_cache


class OrbitCacheQuarantineTests(unittest.TestCase):
    def test_collects_orbit_json_and_matching_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grids_dir = Path(tmpdir)
            orbit = grids_dir / "SG191_G[3, 3, 4].json"
            meta = grids_dir / "SG191_G[3, 3, 4].json.meta"
            unrelated = grids_dir / "notes.json"
            orbit.write_text("{}", encoding="utf-8")
            meta.write_text("{}", encoding="utf-8")
            unrelated.write_text("{}", encoding="utf-8")

            files = collect_cache_files(grids_dir)

        self.assertEqual([path.name for path in files], [
            "SG191_G[3, 3, 4].json",
            "SG191_G[3, 3, 4].json.meta",
        ])

    def test_dry_run_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grids_dir = Path(tmpdir)
            orbit = grids_dir / "SG47_G[3, 3, 3].json"
            orbit.write_text("{}", encoding="utf-8")

            result = quarantine_orbit_cache(grids_dir, apply=False)

            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual(result["file_count"], 1)
            self.assertTrue(orbit.is_file())
            self.assertIsNone(result["quarantine_dir"])

    def test_missing_grids_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing_grids"
            with self.assertRaises(FileNotFoundError):
                quarantine_orbit_cache(missing, apply=False)
            with self.assertRaises(FileNotFoundError):
                quarantine_orbit_cache(missing, apply=True)
            self.assertFalse(missing.exists())

    def test_apply_moves_files_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grids_dir = Path(tmpdir)
            orbit = grids_dir / "SG47_G[3, 3, 3].json"
            meta = grids_dir / "SG47_G[3, 3, 3].json.meta"
            orbit.write_text("{}", encoding="utf-8")
            meta.write_text("{}", encoding="utf-8")

            result = quarantine_orbit_cache(grids_dir, apply=True)
            quarantine_dir = Path(result["quarantine_dir"])

            self.assertEqual(result["mode"], "apply")
            self.assertEqual(result["file_count"], 2)
            self.assertFalse(orbit.exists())
            self.assertFalse(meta.exists())
            self.assertTrue((quarantine_dir / orbit.name).is_file())
            self.assertTrue((quarantine_dir / meta.name).is_file())
            self.assertTrue((quarantine_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
