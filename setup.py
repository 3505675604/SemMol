"""SemMol installation script.

Install on the server (editable mode, so source changes take effect immediately):
    pip install -e .

This package only registers the source as importable modules; it does not package data or weights.
"""

from setuptools import setup, find_packages

setup(
    name="semmol",
    version="0.1.0",
    description="SemMol: Multimodal Molecular Representation Learning Framework (1D SMILES + 2D Graph + 3D Geometry + QM Electron Density)",
    author="SemMol",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "rdkit>=2023.9.5",
        "ogb>=1.3.6",
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=14.0.0",
        "lmdb>=1.4.1",
        "msgpack>=1.0.7",
        "zstandard>=0.22.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.8.0", "seaborn>=0.13.0"],
        "explain": ["shap>=0.44.0"],
        "log": ["tensorboard>=2.15.0", "wandb>=0.16.0"],
        "config": ["omegaconf>=2.3.0", "hydra-core>=1.3.0"],
    },
)
