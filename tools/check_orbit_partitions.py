#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ip4ch.orbit_utils import validate_orbit_partition  # noqa: E402


def grid_size_from_filename(path: Path) -> list[int]:
    match = re.search(r"_G\[([^\]]+)\](?:_ref_[^.]*)?\.json$", path.name)
    if not match:
        raise ValueError(f"cannot infer grid size from filename: {path.name}")
    return [int(part.strip()) for part in match.group(1).split(",")]


def check_file(path: Path, grid_size: list[int] | None = None) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        orbits = json.load(handle)
    grid = grid_size or grid_size_from_filename(path)
    report = validate_orbit_partition(orbits=orbits, grid_size=grid)
    status = "OK" if report.ok else "INVALID"
    print(
        "\t".join(
            [
                status,
                str(path),
                f"grid={list(report.grid_size)}",
                f"N={report.total_grid_points}",
                f"orbits={report.orbit_count}",
                f"duplicates={len(report.duplicate_points)}",
                f"missing={len(report.missing_points)}",
                f"out_of_range={len(report.out_of_range_points)}",
            ]
        )
    )
    if not report.ok:
        print(f"  {report.error_summary()}")
    return report.ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate orbit JSON files as grid partitions.")
    parser.add_argument("paths", nargs="*", help="orbit JSON files to validate")
    parser.add_argument("--all", action="store_true", help="scan Data/Grids/SG*_G*.json")
    parser.add_argument("--grid-size", nargs=3, type=int, default=None, help="explicit grid size for all paths")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    if args.all:
        paths.extend(sorted((PROJECT_ROOT / "Data" / "Grids").glob("SG*_G*.json")))
    if not paths:
        parser.error("provide orbit JSON paths or --all")

    ok = True
    for path in paths:
        if not check_file(path, args.grid_size):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
