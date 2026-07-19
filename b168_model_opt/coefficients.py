from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from b168_model_opt.legacy_eval import _validated_assignment, _validated_kernel
from b168_model_opt.types import OrbitData, PairCoefficients


_CACHE_SCHEMA_VERSION = 2
_CACHE_FIELDS = frozenset(
    {
        "schema",
        "linear",
        "quadratic",
        "metadata",
        "content_hash",
        "source_fingerprint",
        "input_fingerprint",
    }
)
_AGGREGATION_BLOCK_SIZE = 32


class CacheError(ValueError):
    """A malformed, incompatible, or unverifiable coefficient cache."""


def aggregate_orbit_coefficients(
    orbit_data: OrbitData,
    kernel: np.ndarray,
) -> PairCoefficients:
    """Aggregate the production i < j site-pair objective by active orbit.

    The periodic pair construction is expected to yield an inversion-symmetric
    kernel for physical pair potentials.  This aggregation deliberately does
    not use that property: it retains the production site-index ordering so a
    non-symmetric diagnostic kernel cannot introduce a factor-of-two error.
    """
    displacement_kernel = _validated_kernel(orbit_data, kernel)
    active_count = len(orbit_data.active_original)
    site_count = math.prod(orbit_data.grid_size)
    coordinates = np.column_stack(
        np.unravel_index(np.arange(site_count, dtype=np.int64), orbit_data.grid_size)
    )
    grid = np.asarray(orbit_data.grid_size, dtype=np.int64)
    linear = np.zeros(active_count, dtype=np.float64)
    quadratic_by_code: dict[int, float] = {}
    active_by_site = np.asarray(orbit_data.site_to_active)
    first_sites = np.flatnonzero(active_by_site >= 0)
    all_sites = np.arange(site_count, dtype=np.int64)
    conflict_codes = frozenset(
        left * active_count + right for left, right in orbit_data.conflict_edges
    )

    # The production objective emits alpha[i, j] in ascending site-index order.
    # Process bounded blocks of i values, vectorizing every j > i without a site-pair loop.
    for start in range(0, len(first_sites), _AGGREGATION_BLOCK_SIZE):
        block_sites = first_sites[start : start + _AGGREGATION_BLOCK_SIZE]
        block_orbits = active_by_site[block_sites]
        targets = np.broadcast_to(active_by_site, (len(block_sites), site_count))
        valid = (all_sites[None, :] > block_sites[:, None]) & (targets >= 0)
        if not np.any(valid):
            continue

        displacement = (
            coordinates[None, :, :] - coordinates[block_sites, None, :]
        ) % grid
        weights = displacement_kernel[
            displacement[..., 0], displacement[..., 1], displacement[..., 2]
        ]
        sources = np.broadcast_to(block_orbits[:, None], targets.shape)
        source = sources[valid]
        target = targets[valid]
        weight = weights[valid]

        same = source == target
        if np.any(same):
            linear += np.bincount(
                source[same], weights=weight[same], minlength=active_count
            )

        if np.any(~same):
            left = np.minimum(source[~same], target[~same])
            right = np.maximum(source[~same], target[~same])
            weight = weight[~same]
            nonzero = weight != 0.0
            if np.any(nonzero):
                pair_codes = left[nonzero] * active_count + right[nonzero]
                pair_weights = weight[nonzero]
                order = np.argsort(pair_codes, kind="stable")
                sorted_codes = pair_codes[order]
                starts = np.r_[
                    0, np.flatnonzero(np.diff(sorted_codes)) + 1
                ]
                block_codes = sorted_codes[starts]
                block_values = np.add.reduceat(pair_weights[order], starts)
                for code, value in zip(block_codes, block_values):
                    normalized_code = int(code)
                    if normalized_code not in conflict_codes:
                        quadratic_by_code[normalized_code] = (
                            quadratic_by_code.get(normalized_code, 0.0) + float(value)
                        )

    quadratic = {
        divmod(code, active_count): value
        for code, value in sorted(quadratic_by_code.items())
        if value != 0.0
    }

    return PairCoefficients(
        linear=linear,
        quadratic=quadratic,
        metadata={
            "grid_size": orbit_data.grid_size,
            "active_orbits": active_count,
            "quadratic_terms": len(quadratic),
            "pair_order": "site-index-i<j",
        },
    )


def evaluate_compact(assignment: np.ndarray, coefficients: PairCoefficients) -> float:
    _validate_coefficients(coefficients)
    values = _validated_assignment(assignment, len(coefficients.linear))
    total = float(coefficients.linear @ values)
    for (left, right), coefficient in sorted(coefficients.quadratic.items()):
        total += float(coefficient) * values[left] * values[right]
    return float(total)


def _is_index(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _validate_coefficients(coefficients: PairCoefficients) -> None:
    if not isinstance(coefficients, PairCoefficients):
        raise TypeError("coefficients must be a PairCoefficients")
    linear = coefficients.linear
    if linear.ndim != 1 or linear.dtype.kind != "f":
        raise ValueError("coefficient linear vector must have a floating-point dtype")
    if not np.isfinite(linear).all():
        raise ValueError("coefficient linear vector must contain only finite values")
    for edge, value in coefficients.quadratic.items():
        try:
            left, right = edge
        except (TypeError, ValueError) as exc:
            raise ValueError("quadratic keys must be index pairs") from exc
        if (
            not _is_index(left)
            or not _is_index(right)
            or not 0 <= left < right < len(linear)
        ):
            raise ValueError("quadratic keys must be canonical indices in bounds")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("quadratic coefficients must be finite real values")


def _metadata_to_wire(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite numeric values")
        return float(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("metadata arrays must have finite boolean, integer, or float values")
        return {
            "type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.reshape(-1).tolist(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata mapping keys must be strings")
        return {
            "type": "mapping",
            "items": [
                [key, _metadata_to_wire(nested)]
                for key, nested in sorted(value.items())
            ],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_metadata_to_wire(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_metadata_to_wire(item) for item in value]
        return {
            "type": "frozenset",
            "items": sorted(
                items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
            ),
        }
    raise ValueError(f"metadata value cannot be cached: {type(value).__name__}")


def _metadata_from_wire(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite numeric values")
        return value
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("metadata has an invalid encoded value")

    kind = value["type"]
    if kind == "mapping":
        if set(value) != {"type", "items"} or not isinstance(value["items"], list):
            raise ValueError("metadata mapping has an invalid shape")
        restored: dict[str, object] = {}
        for item in value["items"]:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or item[0] in restored
            ):
                raise ValueError("metadata mapping has invalid items")
            restored[item[0]] = _metadata_from_wire(item[1])
        return restored
    if kind in {"tuple", "frozenset"}:
        if set(value) != {"type", "items"} or not isinstance(value["items"], list):
            raise ValueError("metadata sequence has an invalid shape")
        items = tuple(_metadata_from_wire(item) for item in value["items"])
        return frozenset(items) if kind == "frozenset" else items
    if kind == "ndarray":
        if set(value) != {"type", "dtype", "shape", "data"}:
            raise ValueError("metadata ndarray has invalid fields")
        if not isinstance(value["dtype"], str) or not isinstance(value["shape"], list):
            raise ValueError("metadata ndarray has invalid dtype or shape")
        try:
            dtype = np.dtype(value["dtype"])
        except TypeError as exc:
            raise ValueError("metadata ndarray has an invalid dtype") from exc
        if dtype.kind not in "biuf" or any(
            not isinstance(dimension, int) or dimension < 0
            for dimension in value["shape"]
        ):
            raise ValueError("metadata ndarray has an invalid dtype or shape")
        if not isinstance(value["data"], list):
            raise ValueError("metadata ndarray data must be a list")
        expected_size = math.prod(value["shape"])
        if len(value["data"]) != expected_size:
            raise ValueError("metadata ndarray data has an invalid length")
        try:
            array = np.asarray(value["data"], dtype=dtype).reshape(value["shape"])
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata ndarray data is invalid") from exc
        if not np.isfinite(array).all():
            raise ValueError("metadata arrays must contain only finite values")
        return array
    raise ValueError("metadata has an unknown encoded value type")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_fingerprint(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheError(f"{field} must be a nonempty string")
    return value


def _scalar_cache_string(value: np.ndarray, field: str) -> str:
    if value.shape != () or value.dtype.kind not in "US":
        raise CacheError(f"invalid coefficient cache {field} field")
    return _validated_fingerprint(str(value.item()), field)


def _content_hash(linear: np.ndarray, rows: np.ndarray, metadata: str) -> str:
    digest = hashlib.sha256()
    for label, payload in (
        (b"linear", np.ascontiguousarray(linear, dtype=np.dtype("<f8")).tobytes()),
        (b"quadratic", np.ascontiguousarray(rows, dtype=np.dtype("<f8")).tobytes()),
        (b"metadata", metadata.encode("utf-8")),
    ):
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _cache_error(exc: Exception) -> CacheError:
    if isinstance(exc, CacheError):
        return exc
    return CacheError(f"invalid coefficient cache: {exc}")


def save_atomic_cache(
    path: str | Path,
    coefficients: PairCoefficients,
    *,
    source_fingerprint: str,
    input_fingerprint: str,
) -> None:
    """Durably replace a cache file without exposing an incomplete archive."""
    _validate_coefficients(coefficients)
    source_fingerprint = _validated_fingerprint(source_fingerprint, "source fingerprint")
    input_fingerprint = _validated_fingerprint(input_fingerprint, "input fingerprint")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(
        [
            (left, right, coefficient)
            for (left, right), coefficient in sorted(coefficients.quadratic.items())
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    metadata = json.dumps(
        _metadata_to_wire(coefficients.metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    content_hash = _content_hash(coefficients.linear, rows, metadata)
    temporary: Path | None = None
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                schema=np.asarray(_CACHE_SCHEMA_VERSION, dtype=np.int64),
                linear=coefficients.linear,
                quadratic=rows,
                metadata=np.asarray(metadata),
                content_hash=np.asarray(content_hash),
                source_fingerprint=np.asarray(source_fingerprint),
                input_fingerprint=np.asarray(input_fingerprint),
            )
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        except Exception as exc:
            raise CacheError(f"unable to save coefficient cache: {exc}") from exc
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)


def load_cache(
    path: str | Path,
    *,
    source_fingerprint: str,
    input_fingerprint: str,
) -> PairCoefficients:
    """Load and validate a complete versioned coefficient cache archive."""
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            fields = frozenset(payload.files)
            if fields != _CACHE_FIELDS:
                missing = sorted(_CACHE_FIELDS - fields)
                unexpected = sorted(fields - _CACHE_FIELDS)
                raise ValueError(
                    f"cache has invalid required fields; missing={missing}, unexpected={unexpected}"
                )
            schema = payload["schema"]
            linear = payload["linear"].copy()
            rows = payload["quadratic"].copy()
            metadata_field = payload["metadata"]
            content_hash_field = payload["content_hash"]
            source_fingerprint_field = payload["source_fingerprint"]
            input_fingerprint_field = payload["input_fingerprint"]
    except Exception as exc:
        raise _cache_error(exc) from exc

    try:
        expected_source = _validated_fingerprint(source_fingerprint, "source fingerprint")
        expected_input = _validated_fingerprint(input_fingerprint, "input fingerprint")
        if (
            schema.shape != ()
            or schema.dtype.kind not in "iu"
            or int(schema.item()) != _CACHE_SCHEMA_VERSION
        ):
            raise CacheError("invalid coefficient cache schema")
        if linear.ndim != 1 or linear.dtype != np.dtype(np.float64):
            raise CacheError("invalid coefficient cache linear vector")
        if not np.isfinite(linear).all():
            raise CacheError("invalid coefficient cache: linear values must be finite")
        if rows.ndim != 2 or rows.shape[1:] != (3,) or rows.dtype != np.dtype(np.float64):
            raise CacheError("invalid coefficient cache quadratic rows")
        if not np.isfinite(rows).all():
            raise CacheError("invalid coefficient cache: quadratic values must be finite")
        if metadata_field.shape != () or metadata_field.dtype.kind not in "US":
            raise CacheError("invalid coefficient cache metadata field")
        content_hash = _scalar_cache_string(content_hash_field, "content hash")
        if content_hash != _content_hash(linear, rows, str(metadata_field.item())):
            raise CacheError("invalid coefficient cache content hash")
        if _scalar_cache_string(source_fingerprint_field, "source fingerprint") != expected_source:
            raise CacheError("invalid coefficient cache source fingerprint")
        if _scalar_cache_string(input_fingerprint_field, "input fingerprint") != expected_input:
            raise CacheError("invalid coefficient cache input fingerprint")
        try:
            wire_metadata = json.loads(str(metadata_field.item()))
            metadata = _metadata_from_wire(wire_metadata)
        except Exception as exc:
            raise CacheError(f"invalid coefficient cache metadata: {exc}") from exc
    except Exception as exc:
        raise _cache_error(exc) from exc
    if not isinstance(metadata, dict):
        raise CacheError("invalid coefficient cache metadata: root must be a mapping")

    quadratic: dict[tuple[int, int], float] = {}
    for left_raw, right_raw, coefficient_raw in rows:
        if (
            not float(left_raw).is_integer()
            or not float(right_raw).is_integer()
        ):
            raise CacheError("invalid coefficient cache quadratic indices")
        left, right = int(left_raw), int(right_raw)
        if not 0 <= left < right < len(linear) or (left, right) in quadratic:
            raise CacheError("invalid coefficient cache quadratic keys")
        quadratic[(left, right)] = float(coefficient_raw)
    try:
        coefficients = PairCoefficients(
            linear=linear, quadratic=quadratic, metadata=metadata
        )
        _validate_coefficients(coefficients)
        return coefficients
    except Exception as exc:
        raise _cache_error(exc) from exc
