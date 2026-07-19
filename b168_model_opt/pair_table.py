from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


def infer_precision(distances: list[float]) -> int:
    """Infer the decimal lookup precision exactly as the production code does."""
    keys = sorted({round(float(value), 8) for value in distances})
    if len(keys) < 2:
        return 1
    differences = [
        right - left
        for left, right in zip(keys, keys[1:])
        if right - left > 1e-8
    ]
    if not differences:
        return 1
    step = min(differences)
    for decimals in range(5):
        if abs(step - round(step, decimals)) < 1e-8:
            return decimals
    return 4


@dataclass(frozen=True)
class PairTable:
    values: Mapping[float, float]
    precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.precision, int) or isinstance(self.precision, bool):
            raise ValueError("precision must be an integer")
        if not 0 <= self.precision <= 4:
            raise ValueError("precision must be between 0 and 4")

        copied: dict[float, float] = {}
        for distance, energy in self.values.items():
            normalized_distance = float(distance)
            normalized_energy = float(energy)
            if not math.isfinite(normalized_distance) or not math.isfinite(
                normalized_energy
            ):
                raise ValueError("pair-table distances and energies must be finite")
            key = round(normalized_distance, self.precision)
            if key in copied:
                raise ValueError(f"duplicate rounded distance: {key}")
            copied[key] = normalized_energy
        if not copied:
            raise ValueError("pair table must contain at least one value")
        object.__setattr__(self, "values", MappingProxyType(copied))

    @classmethod
    def from_csv(cls, path: str | Path) -> "PairTable":
        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                fieldnames = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV pair table has no header: {path}") from exc

            if not fieldnames or any(not name for name in fieldnames):
                raise ValueError(f"CSV pair table has an invalid header: {path}")
            duplicates = sorted(
                {name for name in fieldnames if fieldnames.count(name) > 1}
            )
            if duplicates:
                raise ValueError(
                    f"duplicate column name(s) in pair table: {', '.join(duplicates)}"
                )
            distance_column = "r_ij" if "r_ij" in fieldnames else fieldnames[0]
            production_priority = (
                "E_ij",
                "median_smoothed",
                "median_effective_pair",
                "q10_smoothed",
                "q10_interpolated",
                "energy",
            )
            energy_column = next(
                (name for name in production_priority if name in fieldnames), None
            )
            if energy_column is None:
                energy_column = fieldnames[1]

            distance_index = fieldnames.index(distance_column)
            energy_index = fieldnames.index(energy_column)
            raw: list[tuple[float, float]] = []
            for row_number, row in enumerate(reader, start=2):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(fieldnames):
                    raise ValueError(
                        f"row {row_number} has {len(row)} columns; "
                        f"expected {len(fieldnames)}"
                    )
                try:
                    distance = float(row[distance_index])
                    energy = float(row[energy_index])
                except ValueError as exc:
                    raise ValueError(
                        f"non-numeric pair-table value on row {row_number}"
                    ) from exc
                if not math.isfinite(distance) or not math.isfinite(energy):
                    raise ValueError(
                        f"pair-table values must be finite on row {row_number}"
                    )
                raw.append((distance, energy))

        if not raw:
            raise ValueError(f"CSV pair table has no usable rows: {path}")

        precision = infer_precision([distance for distance, _ in raw])
        normalized: dict[float, float] = {}
        for distance, energy in raw:
            key = round(distance, precision)
            if key in normalized:
                raise ValueError(f"duplicate rounded distance: {key}")
            normalized[key] = energy
        return cls(values=normalized, precision=precision)

    def energy(self, distance: float) -> float | None:
        value = float(distance)
        if not math.isfinite(value):
            return None
        return self.values.get(round(value, self.precision))
