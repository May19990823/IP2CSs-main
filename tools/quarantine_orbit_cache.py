#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path


def collect_cache_files(grids_dir):
    grids_dir = Path(grids_dir)
    files = []
    for orbit_path in sorted(grids_dir.glob("SG*_G*.json")):
        if orbit_path.name.endswith(".json.meta"):
            continue
        files.append(orbit_path)
        metadata_path = Path(f"{orbit_path}.meta")
        if metadata_path.exists():
            files.append(metadata_path)
    return files


def build_manifest(grids_dir, files, mode, quarantine_dir):
    entries = []
    for path in files:
        stat = path.stat()
        entries.append({
            "source": str(path),
            "name": path.name,
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        })
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "grids_dir": str(Path(grids_dir).resolve()),
        "quarantine_dir": None if quarantine_dir is None else str(Path(quarantine_dir).resolve()),
        "file_count": len(files),
        "files": entries,
    }


def quarantine_orbit_cache(grids_dir, apply=False):
    grids_dir = Path(grids_dir)
    if not grids_dir.is_dir():
        raise FileNotFoundError(f"grids_dir does not exist or is not a directory: {grids_dir}")

    files = collect_cache_files(grids_dir)
    if not apply:
        manifest = build_manifest(
            grids_dir=grids_dir,
            files=files,
            mode="dry-run",
            quarantine_dir=None,
        )
        return manifest

    stamp = time.strftime("%Y%m%d_%H%M%S")
    quarantine_dir = grids_dir / f"quarantine_{stamp}"
    counter = 1
    while quarantine_dir.exists():
        quarantine_dir = grids_dir / f"quarantine_{stamp}_{counter}"
        counter += 1
    quarantine_dir.mkdir(parents=True, exist_ok=False)

    manifest = build_manifest(
        grids_dir=grids_dir,
        files=files,
        mode="apply",
        quarantine_dir=quarantine_dir,
    )
    for source in files:
        target = quarantine_dir / source.name
        shutil.move(str(source), str(target))

    manifest_path = quarantine_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Quarantine generated orbit cache JSON files before a fresh run."
    )
    parser.add_argument("--grids-dir", default="Data/Grids")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    result = quarantine_orbit_cache(args.grids_dir, apply=bool(args.apply))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
