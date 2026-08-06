import numpy as np
import pytest

from src.molecular.electron_density import (
    DensityConfigError,
    DensityInputError,
    GridOverflowError,
    build_promolecular_density,
    validate_density_config,
)


def test_promolecular_grid_integrates_to_electron_count():
    result = build_promolecular_density(
        np.array([1, 8], dtype=np.int64),
        np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32),
        grid_size=32,
        spacing=0.35,
        box_padding=3.0,
    )

    numerical_integral = float(result.grid.sum(dtype=np.float64) * 0.35**3)
    assert result.electron_count == 9.0
    assert result.integrated_electrons == pytest.approx(9.0, abs=1e-4)
    assert numerical_integral == pytest.approx(9.0, abs=1e-4)
    assert result.origin.shape == (3,)
    assert result.spacing == 0.35
    assert result.overflow is False
    assert result.method == "promolecular_gaussian"


def test_fixed_grid_overflow_is_strict_by_default_and_can_be_flagged():
    atomic_numbers = np.array([6, 6], dtype=np.int64)
    coords = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    with pytest.raises(GridOverflowError):
        build_promolecular_density(
            atomic_numbers,
            coords,
            grid_size=8,
            spacing=1.0,
            box_padding=2.0,
        )

    flagged = build_promolecular_density(
        atomic_numbers,
        coords,
        grid_size=8,
        spacing=1.0,
        box_padding=2.0,
        strict=False,
    )
    assert flagged.overflow is True
    assert flagged.overflow_axes.tolist() == [True, False, False]
    assert flagged.integrated_electrons < flagged.electron_count


def test_multi_conformer_mean_rejects_an_empty_selection():
    coords = np.zeros((2, 1, 3), dtype=np.float32)
    with pytest.raises(DensityInputError):
        build_promolecular_density(
            np.array([6]),
            coords,
            conformer_index=None,
            conformer_mask=np.array([False, False]),
        )


def test_multi_conformer_mean_kabsch_aligns_on_heavy_atoms_before_gridding():
    first = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    second = first @ rotation + np.array([10.0, -7.0, 2.0])
    result = build_promolecular_density(
        np.array([6, 8, 1], dtype=np.int64),
        np.stack([first, second]),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
        conformer_index=None,
    )

    assert result.overflow is False
    assert result.conformer_alignment == "heavy_atom_kabsch"
    assert result.conformers_used.tolist() == [0, 1]


def test_density_configuration_rejects_unknown_or_wrong_typed_values():
    valid = {
        "grid_size": 32,
        "spacing": 0.5,
        "box_padding": 4.0,
        "atomic_sigma": None,
        "conformer_index": 0,
        "strict": True,
        "discrete_normalize": True,
    }
    assert validate_density_config(valid) == valid

    with pytest.raises(DensityConfigError):
        validate_density_config({**valid, "unknown": 1})
    with pytest.raises(DensityConfigError):
        validate_density_config({**valid, "strict": "true"})
