"""SemMol: Multimodal molecular representation learning framework.

Supports parallel encoding of four modalities:
    - 1D: SMILES sequence (ChemBERTa-MLM Transformer)
    - 2D: Molecular topology graph (sparse edge-aware GatedGCN / GINE)
    - 3D: Conformer geometry (DimeNet directional message passing)
    - QM: Quantum-mechanical electron-density grid (f_Q)

Core modules:
    - DCL  (Dynamic Central Library)       : Mini-batch K-means + EMA semantic centers
    - ACSM (Anchor-Centers Soft Matching)  : Anchor from any modality, Top-M weighted fusion and denoising

Subpackages: datasets, molecular, models, losses, trainers, evaluation, utils
"""

__version__ = "0.1.0"
__author__ = "SemMol"
