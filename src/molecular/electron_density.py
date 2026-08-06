"""Element-aware promolecular electron-density approximation.

This module does not compute a quantum-mechanical/DFT density. It approximates
the promolecular density as a sum of normalized, element-dependent Gaussian
atomic densities on a fixed Cartesian grid.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Optional, Union

import numpy as np
from rdkit import Chem


class DensityError(ValueError):
    """Base class for promolecular density failures."""


class DensityInputError(DensityError):
    """Raised for invalid atom, coordinate, or conformer input."""


class GridOverflowError(DensityError):
    """Raised when the requested fixed grid cannot contain the padded molecule."""

    def __init__(
        self,
        required_extent: np.ndarray,
        available_extent: np.ndarray,
    ) -> None:
        self.required_extent = np.asarray(required_extent, dtype=np.float64)
        self.available_extent = np.asarray(available_extent, dtype=np.float64)
        super().__init__(
            "fixed grid is too small: required extent "
            f"{self.required_extent.tolist()}, available "
            f"{self.available_extent.tolist()}"
        )


class DensityConfigError(DensityError):
    """Raised when the unified density configuration violates its schema."""


@dataclass(frozen=True)
class DensityGridResult:
    """Density grid plus the physical and numerical metadata needed downstream."""

    grid: np.ndarray
    origin: np.ndarray
    spacing: float
    electron_count: float
    integrated_electrons: float
    overflow: bool
    overflow_axes: np.ndarray
    atomic_sigmas: np.ndarray
    method: str
    box_padding: float
    conformers_used: np.ndarray
    conformer_reduction: str
    conformer_alignment: str
    normalization_requested: str
    normalization_applied: str

    def to_storage_dict(self) -> dict[str, np.ndarray]:
        """Return a safe, non-object mapping suitable for compressed NPZ."""
        stored_grid = np.asarray(self.grid, dtype=np.float32)
        stored_spacing = np.asarray(self.spacing, dtype=np.float32)
        stored_integral = float(
            stored_grid.sum(dtype=np.float64) * float(stored_spacing) ** 3
        )
        return {
            "grid": stored_grid,
            "origin": np.asarray(self.origin, dtype=np.float32),
            "spacing": stored_spacing,
            "electron_count": np.asarray(self.electron_count, dtype=np.float64),
            "integrated_electrons": np.asarray(
                stored_integral,
                dtype=np.float64,
            ),
            "prequantization_integrated_electrons": np.asarray(
                self.integrated_electrons,
                dtype=np.float64,
            ),
            "overflow": np.asarray(self.overflow, dtype=np.bool_),
            "overflow_axes": np.asarray(self.overflow_axes, dtype=np.bool_),
            "atomic_sigmas": np.asarray(self.atomic_sigmas, dtype=np.float32),
            "method": np.asarray(self.method, dtype=np.str_),
            "box_padding": np.asarray(self.box_padding, dtype=np.float32),
            "conformers_used": np.asarray(self.conformers_used, dtype=np.int64),
            "conformer_reduction": np.asarray(
                self.conformer_reduction,
                dtype=np.str_,
            ),
            "conformer_alignment": np.asarray(
                self.conformer_alignment,
                dtype=np.str_,
            ),
            "normalization_requested": np.asarray(
                self.normalization_requested,
                dtype=np.str_,
            ),
            "normalization_applied": np.asarray(
                self.normalization_applied,
                dtype=np.str_,
            ),
        }


SigmaSpec = Optional[Union[float, Mapping[int, float]]]

_DENSITY_CONFIG_FIELDS = {
    "grid_size",
    "spacing",
    "box_padding",
    "atomic_sigma",
    "conformer_index",
    "strict",
    "discrete_normalize",
}


def _atomic_number_array(atomic_numbers: np.ndarray) -> np.ndarray:
    raw = np.asarray(atomic_numbers)
    if raw.ndim != 1 or raw.size == 0:
        raise DensityInputError("atomic_numbers must be a non-empty 1D array")
    if raw.dtype.kind not in {"i", "u"}:
        raise DensityInputError(
            "atomic_numbers must use an integer dtype; implicit truncation is forbidden"
        )
    if raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
        raise DensityInputError("atomic_numbers exceed the int64 range")
    numbers = raw.astype(np.int64, copy=False)
    if np.any(numbers <= 0) or np.any(numbers > 118):
        raise DensityInputError("atomic_numbers must be in the range 1..118")
    return numbers


def validate_density_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the single flat schema used by APIs and preprocessing."""
    if not isinstance(config, MappingABC):
        raise DensityConfigError("density config must be a mapping")
    fields = set(config)
    missing = _DENSITY_CONFIG_FIELDS - fields
    unknown = fields - _DENSITY_CONFIG_FIELDS
    if missing or unknown:
        raise DensityConfigError(
            f"density config fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    grid_size = config["grid_size"]
    if isinstance(grid_size, bool) or not isinstance(grid_size, Integral):
        raise DensityConfigError("grid_size must be an integer")
    if int(grid_size) < 2:
        raise DensityConfigError("grid_size must be at least 2")

    spacing = config["spacing"]
    padding = config["box_padding"]
    if (
        isinstance(spacing, bool)
        or not isinstance(spacing, Real)
        or not np.isfinite(float(spacing))
        or float(spacing) <= 0
    ):
        raise DensityConfigError("spacing must be a finite positive number")
    if (
        isinstance(padding, bool)
        or not isinstance(padding, Real)
        or not np.isfinite(float(padding))
        or float(padding) < 0
    ):
        raise DensityConfigError("box_padding must be finite and non-negative")

    atomic_sigma = config["atomic_sigma"]
    if atomic_sigma is not None:
        if isinstance(atomic_sigma, MappingABC):
            for atomic_number, sigma in atomic_sigma.items():
                if (
                    isinstance(atomic_number, bool)
                    or not isinstance(atomic_number, Integral)
                    or not 1 <= int(atomic_number) <= 118
                    or isinstance(sigma, bool)
                    or not isinstance(sigma, Real)
                    or not np.isfinite(float(sigma))
                    or float(sigma) <= 0
                ):
                    raise DensityConfigError(
                        "atomic_sigma mapping requires positive integer keys "
                        "and finite positive numeric values"
                    )
        elif (
            isinstance(atomic_sigma, bool)
            or not isinstance(atomic_sigma, Real)
            or not np.isfinite(float(atomic_sigma))
            or float(atomic_sigma) <= 0
        ):
            raise DensityConfigError(
                "atomic_sigma must be None, a positive scalar, or an element mapping"
            )

    conformer_index = config["conformer_index"]
    if conformer_index is not None and (
        isinstance(conformer_index, bool)
        or not isinstance(conformer_index, Integral)
        or int(conformer_index) < 0
    ):
        raise DensityConfigError(
            "conformer_index must be None or a non-negative integer"
        )
    if not isinstance(config["strict"], bool):
        raise DensityConfigError("strict must be a boolean")
    if not isinstance(config["discrete_normalize"], bool):
        raise DensityConfigError("discrete_normalize must be a boolean")

    return {
        "grid_size": int(grid_size),
        "spacing": float(spacing),
        "box_padding": float(padding),
        "atomic_sigma": atomic_sigma,
        "conformer_index": (
            None if conformer_index is None else int(conformer_index)
        ),
        "strict": config["strict"],
        "discrete_normalize": config["discrete_normalize"],
    }


def element_sigmas(
    atomic_numbers: np.ndarray,
    atomic_sigma: SigmaSpec = None,
    covalent_radius_scale: float = 0.45,
    min_sigma: float = 0.20,
) -> np.ndarray:
    """Derive Gaussian widths in Å from RDKit covalent radii or overrides."""
    numbers = _atomic_number_array(atomic_numbers)
    for name, value in (
        ("min_sigma", min_sigma),
        ("covalent_radius_scale", covalent_radius_scale),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise DensityInputError(f"{name} must be a finite positive number")
    min_sigma = float(min_sigma)
    covalent_radius_scale = float(covalent_radius_scale)

    if isinstance(atomic_sigma, MappingABC):
        for atomic_number, sigma in atomic_sigma.items():
            if (
                isinstance(atomic_number, bool)
                or not isinstance(atomic_number, Integral)
                or not 1 <= int(atomic_number) <= 118
                or isinstance(sigma, bool)
                or not isinstance(sigma, Real)
                or not np.isfinite(float(sigma))
                or float(sigma) <= 0
            ):
                raise DensityInputError(
                    "atomic_sigma overrides require element numbers in 1..118 "
                    "and finite positive widths"
                )
        periodic_table = Chem.GetPeriodicTable()
        values = [
            float(
                atomic_sigma.get(
                    int(number),
                    max(
                        min_sigma,
                        periodic_table.GetRcovalent(int(number))
                        * covalent_radius_scale,
                    ),
                )
            )
            for number in numbers
        ]
    elif atomic_sigma is not None:
        if (
            isinstance(atomic_sigma, bool)
            or not isinstance(atomic_sigma, Real)
            or not np.isfinite(float(atomic_sigma))
            or float(atomic_sigma) <= 0
        ):
            raise DensityInputError(
                "atomic_sigma must be None, a finite positive scalar, "
                "or an element mapping"
            )
        values = [float(atomic_sigma)] * numbers.size
    else:
        periodic_table = Chem.GetPeriodicTable()
        values = [
            max(
                min_sigma,
                periodic_table.GetRcovalent(int(number))
                * covalent_radius_scale,
            )
            for number in numbers
        ]
    sigmas = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise DensityInputError("all Gaussian sigma values must be finite and positive")
    return sigmas


def _select_conformers(
    coords: np.ndarray,
    conformer_index: Optional[int],
    conformer_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_coordinates = np.asarray(coords)
    if raw_coordinates.dtype.kind not in {"i", "u", "f"}:
        raise DensityInputError("coords must use a real numeric dtype")
    coordinates = raw_coordinates.astype(np.float64, copy=False)
    if coordinates.ndim == 2:
        if coordinates.shape[1:] != (3,):
            raise DensityInputError("coords must have shape (A, 3) or (C, A, 3)")
        coordinates = coordinates[None, ...]
    elif coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise DensityInputError("coords must have shape (A, 3) or (C, A, 3)")
    if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
        raise DensityInputError("coords must not contain an empty conformer or atom axis")
    if not np.all(np.isfinite(coordinates)):
        raise DensityInputError("coords must contain only finite values")

    if conformer_mask is None:
        mask = np.ones(coordinates.shape[0], dtype=np.bool_)
    else:
        raw_mask = np.asarray(conformer_mask)
        if raw_mask.dtype.kind != "b":
            raise DensityInputError(
                "conformer_mask must use a boolean dtype"
            )
        mask = raw_mask.astype(np.bool_, copy=False)
        if mask.shape != (coordinates.shape[0],):
            raise DensityInputError("conformer_mask must have shape (C,)")

    if conformer_index is None:
        selected_indices = np.flatnonzero(mask)
        reduction = "mean"
    else:
        if not isinstance(conformer_index, (int, np.integer)):
            raise DensityInputError("conformer_index must be an integer or None")
        if conformer_index < 0 or conformer_index >= coordinates.shape[0]:
            raise DensityInputError("conformer_index is out of range")
        if not mask[int(conformer_index)]:
            raise DensityInputError("selected conformer is masked invalid")
        selected_indices = np.asarray([int(conformer_index)], dtype=np.int64)
        reduction = "single"
    if selected_indices.size == 0:
        raise DensityInputError("no valid conformer was selected")
    return coordinates[selected_indices], selected_indices, reduction


def _align_conformers_on_heavy_atoms(
    conformers: np.ndarray,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    """Rigidly align conformers to the first one using heavy-atom Kabsch."""
    heavy_indices = np.flatnonzero(atomic_numbers != 1)
    if heavy_indices.size == 0:
        raise DensityInputError(
            "multi-conformer Kabsch alignment requires at least one heavy atom"
        )
    aligned = np.asarray(conformers, dtype=np.float64).copy()
    reference_heavy = aligned[0, heavy_indices]
    reference_centroid = reference_heavy.mean(axis=0)
    reference_centered = reference_heavy - reference_centroid
    for index in range(1, aligned.shape[0]):
        mobile_heavy = aligned[index, heavy_indices]
        mobile_centroid = mobile_heavy.mean(axis=0)
        mobile_centered = mobile_heavy - mobile_centroid
        covariance = mobile_centered.T @ reference_centered
        left, _, right_transpose = np.linalg.svd(covariance)
        if np.linalg.det(left @ right_transpose) < 0:
            right_transpose[-1, :] *= -1
        rotation = left @ right_transpose
        aligned[index] = (
            (aligned[index] - mobile_centroid) @ rotation
            + reference_centroid
        )
    return aligned


def build_promolecular_density(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    grid_size: int = 32,
    spacing: float = 0.75,
    box_padding: float = 4.0,
    atomic_sigma: SigmaSpec = None,
    conformer_index: Optional[int] = 0,
    conformer_mask: Optional[np.ndarray] = None,
    strict: bool = True,
    discrete_normalize: bool = True,
) -> DensityGridResult:
    """Construct an element-aware Gaussian promolecular density.

    Passing ``conformer_index=None`` averages density grids over all conformers
    selected by ``conformer_mask``. In strict mode, insufficient fixed-grid
    coverage raises ``GridOverflowError`` instead of clipping or rescaling.
    """
    validated = validate_density_config(
        {
            "grid_size": grid_size,
            "spacing": spacing,
            "box_padding": box_padding,
            "atomic_sigma": atomic_sigma,
            "conformer_index": conformer_index,
            "strict": strict,
            "discrete_normalize": discrete_normalize,
        }
    )
    grid_size = validated["grid_size"]
    spacing = validated["spacing"]
    box_padding = validated["box_padding"]
    atomic_sigma = validated["atomic_sigma"]
    conformer_index = validated["conformer_index"]
    strict = validated["strict"]
    discrete_normalize = validated["discrete_normalize"]

    numbers = _atomic_number_array(atomic_numbers)
    selected, selected_indices, reduction = _select_conformers(
        coords,
        conformer_index,
        conformer_mask,
    )
    if selected.shape[1] != numbers.size:
        raise DensityInputError(
            "atomic_numbers length must equal the coordinate atom dimension"
        )
    alignment = "none"
    if reduction == "mean" and selected.shape[0] > 1:
        selected = _align_conformers_on_heavy_atoms(selected, numbers)
        alignment = "heavy_atom_kabsch"

    sigmas = element_sigmas(numbers, atomic_sigma=atomic_sigma)
    coord_min = selected.min(axis=(0, 1))
    coord_max = selected.max(axis=(0, 1))
    required_extent = coord_max - coord_min + 2.0 * box_padding
    available_extent = np.full(3, (grid_size - 1) * spacing, dtype=np.float64)
    overflow_axes = required_extent > available_extent + 1e-10
    overflow = bool(np.any(overflow_axes))
    if overflow and strict:
        raise GridOverflowError(required_extent, available_extent)

    center = 0.5 * (coord_min + coord_max)
    origin = center - 0.5 * available_extent
    axes = [
        origin[axis] + np.arange(grid_size, dtype=np.float64) * spacing
        for axis in range(3)
    ]
    voxel_volume = float(spacing**3)
    conformer_grids: list[np.ndarray] = []
    normalization_requested = (
        "discrete_electron_count"
        if discrete_normalize
        else "continuous_gaussian"
    )
    normalization_applied = (
        "discrete_electron_count"
        if discrete_normalize and not overflow
        else "continuous_gaussian"
    )

    for conformer in selected:
        # Isotropic Gaussians are separable.  Building three (A, N) axis
        # kernels avoids three full N^3 coordinate meshes and avoids evaluating
        # exp() once per voxel per atom, which is material at PCQM scale.
        axis_kernels = [
            np.exp(
                -0.5
                * (
                    (
                        axis[None, :]
                        - conformer[:, axis_index, None]
                    )
                    / sigmas[:, None]
                )
                ** 2
            )
            for axis_index, axis in enumerate(axes)
        ]
        if discrete_normalize and not overflow:
            discrete_integrals = (
                axis_kernels[0].sum(axis=1, dtype=np.float64)
                * axis_kernels[1].sum(axis=1, dtype=np.float64)
                * axis_kernels[2].sum(axis=1, dtype=np.float64)
                * voxel_volume
            )
            if np.any(~np.isfinite(discrete_integrals)) or np.any(
                discrete_integrals <= 0
            ):
                raise DensityInputError(
                    "an atomic Gaussian has no representable mass on the grid"
                )
            weights = numbers.astype(np.float64) / discrete_integrals
        else:
            weights = numbers.astype(np.float64) / (
                (2.0 * np.pi) ** 1.5 * sigmas**3
            )
        grid = np.einsum(
            "ai,aj,ak,a->ijk",
            axis_kernels[0],
            axis_kernels[1],
            axis_kernels[2],
            weights,
            optimize=True,
        )
        conformer_grids.append(grid)

    density = np.mean(np.stack(conformer_grids, axis=0), axis=0)
    electron_count = float(numbers.sum(dtype=np.int64))
    integrated_electrons = float(density.sum(dtype=np.float64) * voxel_volume)
    return DensityGridResult(
        grid=density.astype(np.float32),
        origin=origin.astype(np.float32),
        spacing=float(spacing),
        electron_count=electron_count,
        integrated_electrons=integrated_electrons,
        overflow=overflow,
        overflow_axes=overflow_axes,
        atomic_sigmas=sigmas.astype(np.float32),
        method="promolecular_gaussian",
        box_padding=float(box_padding),
        conformers_used=selected_indices,
        conformer_reduction=reduction,
        conformer_alignment=alignment,
        normalization_requested=normalization_requested,
        normalization_applied=normalization_applied,
    )


def build_density_grid(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    grid_size: int = 32,
    resolution: float = 0.75,
    box_padding: float = 4.0,
    atomic_sigma: SigmaSpec = None,
    normalize: bool = True,
    strict: bool = True,
    conformer_index: Optional[int] = 0,
    conformer_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Backward-compatible grid-only wrapper.

    ``normalize`` now means discrete electron-count normalization; it no longer
    performs peak normalization to ``[0, 1]``.
    """
    return build_promolecular_density(
        atomic_numbers,
        coords,
        grid_size=grid_size,
        spacing=resolution,
        box_padding=box_padding,
        atomic_sigma=atomic_sigma,
        conformer_index=conformer_index,
        conformer_mask=conformer_mask,
        strict=strict,
        discrete_normalize=normalize,
    ).grid


def generate_electron_density_grid(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    grid_size: int = 32,
    extent: Optional[float] = None,
    **kwargs: object,
) -> np.ndarray:
    """Compatibility wrapper supporting physical ``extent`` when provided."""
    if extent is not None:
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("extent must be finite and positive")
        resolution = float(extent) / float(grid_size - 1)
    else:
        resolution = float(kwargs.pop("resolution", 0.75))
    return build_density_grid(
        atomic_numbers,
        coords,
        grid_size=grid_size,
        resolution=resolution,
        **kwargs,
    )


def density_grid_to_dict(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    **kwargs: object,
) -> dict[str, np.ndarray]:
    """Return the preferred grid-plus-metadata safe storage contract."""
    if "resolution" in kwargs and "spacing" not in kwargs:
        kwargs["spacing"] = kwargs.pop("resolution")
    result = build_promolecular_density(atomic_numbers, coords, **kwargs)
    return result.to_storage_dict()
