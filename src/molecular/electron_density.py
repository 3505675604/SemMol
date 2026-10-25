"""Read externally computed QM density; never synthesize or normalize density.

Producer-supplied safe NPZ artifacts must satisfy the contract below. Atomic
Gaussian/promolecular proxies are not accepted as a quantum modality.
"""
from __future__ import annotations

import hashlib
import io
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ExternalDensityError(ValueError):
    """An external density artifact is invalid or mismatched."""


EXTERNAL_DENSITY_SCHEMA = "semmol.external_qm_density.v1"
EXTERNAL_DENSITY_REQUIRED_FIELDS = frozenset({
    "schema", "grid", "origin", "spacing", "density_kind", "units",
    "length_unit", "calculation_method", "basis_set", "source_software",
    "source_uri", "charge", "spin_multiplicity", "electron_count",
    "integrated_electrons", "source_index", "canonical_smiles",
})


def _scalar(value: Any, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind == "O":
            raise ExternalDensityError(f"{name} must be a safe scalar")
        return value.item()
    return value.item() if isinstance(value, np.generic) else value


def _real(value: Any, name: str, *, positive: bool = False) -> float:
    value = _scalar(value, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExternalDensityError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ExternalDensityError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    value = _scalar(value, name)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ExternalDensityError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ExternalDensityError(f"{name} must be >= {minimum}")
    return result


def validate_external_density(
    density: Mapping[str, Any],
    *,
    expected_source_index: int | None = None,
    expected_smiles: str | None = None,
    atomic_numbers: Sequence[int] | np.ndarray | None = None,
    grid_size: int = 32,
    voxel_spacing: float = 0.75,
    max_relative_integration_error: float = 0.05,
) -> dict[str, Any]:
    """Validate physical values/provenance without resampling or rescaling.

    The fixed grid/spacing must match QMEncoder. Integral metadata is checked
    against the grid; numerical integration error relative to the molecular
    electron count is retained explicitly, never hidden by normalization. The
    default numerical QC rejects losses/excesses above 5%; it is an engineering
    check, not a claim of quantum-calculation accuracy.
    """
    if not isinstance(density, Mapping):
        raise ExternalDensityError("external density must be a mapping")
    missing = EXTERNAL_DENSITY_REQUIRED_FIELDS - set(density)
    if missing:
        raise ExternalDensityError(f"external QM density missing fields: {sorted(missing)}")
    parsed = {key: value if key in {"grid", "origin"} else _scalar(value, key)
              for key, value in density.items()}
    for key in (
        "schema", "density_kind", "units", "length_unit", "calculation_method",
        "basis_set", "source_software", "source_uri", "canonical_smiles",
    ):
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise ExternalDensityError(f"{key} must be a non-empty string")
        parsed[key] = parsed[key].strip()
    if parsed["schema"] != EXTERNAL_DENSITY_SCHEMA:
        raise ExternalDensityError("unsupported external QM density schema")
    if parsed["density_kind"] != "quantum_mechanical":
        raise ExternalDensityError("QM requires external quantum_mechanical density; proxies are forbidden")
    if any(word in parsed["calculation_method"].lower() for word in ("promolecular", "proxy", "atomic_gaussian")):
        raise ExternalDensityError("promolecular/atomic Gaussian proxies are not QM densities")
    if parsed["length_unit"] != "angstrom" or parsed["units"] != "electron/angstrom^3":
        raise ExternalDensityError("external QM units must be angstrom and electron/angstrom^3; convert externally")
    parsed["source_index"] = _integer(parsed["source_index"], "source_index", minimum=0)
    parsed["charge"] = _integer(parsed["charge"], "charge")
    parsed["spin_multiplicity"] = _integer(parsed["spin_multiplicity"], "spin_multiplicity", minimum=1)
    parsed["spacing"] = _real(parsed["spacing"], "spacing", positive=True)
    parsed["electron_count"] = _real(parsed["electron_count"], "electron_count", positive=True)
    parsed["integrated_electrons"] = _real(parsed["integrated_electrons"], "integrated_electrons", positive=True)
    if not math.isclose(parsed["electron_count"], round(parsed["electron_count"]), abs_tol=1e-6):
        raise ExternalDensityError("electron_count must be integral for a molecular QM state")
    unpaired = parsed["spin_multiplicity"] - 1
    if unpaired > round(parsed["electron_count"]) or (round(parsed["electron_count"]) - unpaired) % 2:
        raise ExternalDensityError("spin_multiplicity is incompatible with electron_count")
    if expected_source_index is not None and parsed["source_index"] != expected_source_index:
        raise ExternalDensityError("external QM source_index does not match the molecule")
    if expected_smiles is not None and parsed["canonical_smiles"] != expected_smiles:
        raise ExternalDensityError("external QM canonical_smiles does not match the molecule")
    if not math.isclose(parsed["spacing"], voxel_spacing, rel_tol=0, abs_tol=1e-7):
        raise ExternalDensityError(f"external QM spacing must match QMEncoder ({voxel_spacing} angstrom)")
    grid = np.asarray(parsed["grid"])
    if grid.shape == (grid_size, grid_size, grid_size):
        grid = grid[None, ...]
    if grid.shape != (1, grid_size, grid_size, grid_size) or grid.dtype.kind != "f":
        raise ExternalDensityError(f"external QM grid must be floating [1,{grid_size},{grid_size},{grid_size}]")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ExternalDensityError("external QM grid must be finite and nonnegative")
    origin = np.asarray(parsed["origin"])
    if origin.shape != (3,) or origin.dtype.kind not in "fiu" or not np.all(np.isfinite(origin)):
        raise ExternalDensityError("external QM origin must contain three finite angstrom coordinates")
    integral = float(grid.sum(dtype=np.float64) * parsed["spacing"] ** 3)
    if not math.isclose(integral, parsed["integrated_electrons"], rel_tol=1e-6, abs_tol=1e-7):
        raise ExternalDensityError("integrated_electrons disagrees with the stored external QM grid")
    tolerance = _real(max_relative_integration_error, "max_relative_integration_error")
    if not 0 <= tolerance < 1:
        raise ExternalDensityError("max_relative_integration_error must be in [0,1)")
    if abs(integral - parsed["electron_count"]) / parsed["electron_count"] > tolerance:
        raise ExternalDensityError("external QM grid electron integral exceeds the allowed numerical integration error")
    if atomic_numbers is not None:
        numbers = np.asarray(atomic_numbers)
        if numbers.ndim != 1 or numbers.dtype.kind not in "iu" or np.any((numbers < 1) | (numbers > 118)):
            raise ExternalDensityError("atomic_numbers must be integer Z values in [1,118]")
        expected_count = int(numbers.sum(dtype=np.int64)) - parsed["charge"]
        if not math.isclose(parsed["electron_count"], expected_count, rel_tol=0, abs_tol=1e-6):
            raise ExternalDensityError("electron_count must equal sum(atomic_numbers) - charge")
    with np.errstate(over="ignore", invalid="ignore"):
        parsed["grid"] = grid.astype(np.float32, copy=False)
        parsed["origin"] = origin.astype(np.float32, copy=False)
    if not np.all(np.isfinite(parsed["grid"])) or not np.all(np.isfinite(parsed["origin"])):
        raise ExternalDensityError("external QM values exceed float32 range")
    parsed["integration_error_electrons"] = integral - parsed["electron_count"]
    return parsed


def load_external_density(path: str | Path, **validation_options: Any) -> dict[str, Any]:
    """Read a producer-supplied safe NPZ and record SHA256 provenance."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"external QM density is required but missing: {source}; no density is synthesized")
    payload = source.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        density = {key: arrays[key] for key in arrays.files}
    parsed = validate_external_density(density, **validation_options)
    parsed["source_sha256"] = hashlib.sha256(payload).hexdigest()
    return parsed


__all__ = ["ExternalDensityError", "EXTERNAL_DENSITY_SCHEMA", "EXTERNAL_DENSITY_REQUIRED_FIELDS", "validate_external_density", "load_external_density"]
