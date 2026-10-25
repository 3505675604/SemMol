"""Pure-NumPy external QM contract tests; no density generator is exercised."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "src/molecular/electron_density.py"
_SPEC = importlib.util.spec_from_file_location("semmol_external_density_contract", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def external_fixture() -> dict:
    # Numerical test fixture only, not a purported physical QM calculation.
    grid = np.full((1, 32, 32, 32), 5.0 / (32 ** 3 * 0.75 ** 3), dtype=np.float32)
    return {
        "schema": module.EXTERNAL_DENSITY_SCHEMA,
        "grid": grid, "origin": np.zeros(3, dtype=np.float32), "spacing": 0.75,
        "density_kind": "quantum_mechanical", "units": "electron/angstrom^3",
        "length_unit": "angstrom", "calculation_method": "DFT/PBE0",
        "basis_set": "def2-SVP", "source_software": "unit-test-only",
        "source_uri": "fixture://not-a-real-calculation", "charge": 1,
        "spin_multiplicity": 2, "electron_count": 5.0,
        "integrated_electrons": float(grid.sum(dtype=np.float64) * 0.75 ** 3),
        "source_index": 9, "canonical_smiles": "[C+]",
    }


class TestExternalDensity(unittest.TestCase):
    def test_values_are_not_rescaled_to_electron_count(self):
        density = external_fixture()
        density["grid"] *= 0.99
        density["integrated_electrons"] = float(density["grid"].sum(dtype=np.float64) * 0.75 ** 3)
        parsed = module.validate_external_density(density, atomic_numbers=np.array([6]))
        np.testing.assert_array_equal(parsed["grid"], density["grid"])
        self.assertNotEqual(parsed["integration_error_electrons"], 0)

    def test_charge_is_included_in_electron_count(self):
        density = external_fixture()
        density["electron_count"] = 6.0
        density["spin_multiplicity"] = 1
        density["grid"] *= 6.0 / 5.0
        density["integrated_electrons"] = float(density["grid"].sum(dtype=np.float64) * 0.75 ** 3)
        with self.assertRaisesRegex(module.ExternalDensityError, "sum"):
            module.validate_external_density(density, atomic_numbers=np.array([6]))

    def test_proxy_and_missing_provenance_are_rejected(self):
        density = external_fixture()
        density["calculation_method"] = "promolecular_gaussian"
        with self.assertRaisesRegex(module.ExternalDensityError, "proxies"):
            module.validate_external_density(density)
        density = external_fixture()
        del density["basis_set"]
        with self.assertRaisesRegex(module.ExternalDensityError, "missing fields"):
            module.validate_external_density(density)

    def test_identity_and_units_are_checked(self):
        for options in ({"expected_source_index": 10}, {"expected_smiles": "C"}):
            with self.subTest(options=options), self.assertRaises(module.ExternalDensityError):
                module.validate_external_density(external_fixture(), **options)
        density = external_fixture()
        density["units"] = "electron/bohr^3"
        with self.assertRaisesRegex(module.ExternalDensityError, "units"):
            module.validate_external_density(density)

    def test_spin_integral_and_encoder_spacing_are_checked(self):
        for key, value in (("spin_multiplicity", 1), ("integrated_electrons", 1.0), ("spacing", 0.5)):
            density = external_fixture()
            density[key] = value
            with self.subTest(key=key), self.assertRaises(module.ExternalDensityError):
                module.validate_external_density(density)

    def test_missing_external_file_does_not_generate_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.npz"
            with self.assertRaisesRegex(FileNotFoundError, "no density is synthesized"):
                module.load_external_density(path)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_safe_npz_roundtrip_records_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "9.npz"
            density = external_fixture()
            np.savez(path, **density)
            loaded = module.load_external_density(path, expected_source_index=9)
            self.assertEqual(len(loaded["source_sha256"]), 64)
            np.testing.assert_array_equal(loaded["grid"], density["grid"])

    def test_no_generation_api_remains(self):
        self.assertFalse(hasattr(module, "build_promolecular_density"))
        self.assertFalse(hasattr(module, "build_density_grid"))


if __name__ == "__main__":
    unittest.main()
