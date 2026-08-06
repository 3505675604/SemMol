"""Shared-space projection modules."""

from .bank import ProjectionBank
from .sap_projection import SemanticAttentionProjection
from .unified_projection import ProjectionOutput, UnifiedProjectionMLP

__all__ = [
    "ProjectionBank",
    "ProjectionOutput",
    "SemanticAttentionProjection",
    "UnifiedProjectionMLP",
]
