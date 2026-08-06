"""Per-modality projection modules with a shared output dimensionality."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from src.models.encoders.common import EncoderOutput, validate_encoder_output

from .sap_projection import SemanticAttentionProjection
from .unified_projection import ProjectionOutput, UnifiedProjectionMLP


class ProjectionBank(nn.Module):
    """Own independent projection weights for 1D, 2D, 3D, and QM features."""

    MODALITIES = ("1d", "2d", "3d", "qm")

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 256,
        *,
        modalities: Sequence[str] | None = None,
        projection_types: Mapping[str, str] | None = None,
        mlp: Mapping[str, Any] | None = None,
        sap: Mapping[str, Any] | None = None,
        normalize_eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        if modalities is None:
            selected_modalities = self.MODALITIES
        else:
            if isinstance(modalities, (str, bytes)) or not isinstance(
                modalities, Sequence
            ):
                raise TypeError("modalities must be a sequence of modality names")
            normalized_modalities = tuple(
                str(modality).strip().lower() for modality in modalities
            )
            if not normalized_modalities:
                raise ValueError("modalities must contain at least one modality")
            if any(not modality for modality in normalized_modalities):
                raise ValueError("modality names must be non-empty")
            if len(set(normalized_modalities)) != len(normalized_modalities):
                raise ValueError("modalities must be unique")
            unknown_modalities = set(normalized_modalities) - set(self.MODALITIES)
            if unknown_modalities:
                raise ValueError(
                    f"Unknown projection modalities: {sorted(unknown_modalities)}"
                )
            selected_modalities = normalized_modalities

        selected_types = {
            "1d": "mlp",
            "2d": "mlp",
            "3d": "sap",
            "qm": "sap",
        }
        if projection_types is not None:
            unknown = set(projection_types) - set(self.MODALITIES)
            if unknown:
                raise ValueError(
                    f"Unknown projection modalities: {sorted(unknown)}"
                )
            selected_types.update(
                {name: kind.strip().lower() for name, kind in projection_types.items()}
            )

        if normalize_eps <= 0.0:
            raise ValueError("normalize_eps must be positive")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        mlp_options = dict(mlp or {})
        sap_options = dict(sap or {})
        reserved = {"input_dim", "output_dim"}
        if reserved.intersection(mlp_options):
            raise ValueError("mlp options cannot override input_dim or output_dim")
        if reserved.intersection(sap_options):
            raise ValueError("sap options cannot override input_dim or output_dim")
        mlp_options.setdefault("normalize_eps", normalize_eps)
        mlp_options.setdefault("validate_values", validate_values)
        sap_options.setdefault("normalize_eps", normalize_eps)
        sap_options.setdefault("validate_values", validate_values)

        modules: dict[str, nn.Module] = {}
        for modality in selected_modalities:
            projection_type = selected_types[modality]
            if projection_type == "mlp":
                modules[modality] = UnifiedProjectionMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **mlp_options,
                )
            elif projection_type == "sap":
                modules[modality] = SemanticAttentionProjection(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **sap_options,
                )
            else:
                raise ValueError(
                    f"Unsupported projection type {projection_type!r} for {modality}; "
                    "expected 'mlp' or 'sap'"
                )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modalities = tuple(selected_modalities)
        self.projection_types = {
            modality: selected_types[modality]
            for modality in self.modalities
        }
        self.validate_values = validate_values
        self.projections = nn.ModuleDict(modules)

    def forward(
        self,
        modality: str,
        encoder_output: EncoderOutput,
        *,
        full_batch_size: int | None = None,
    ) -> ProjectionOutput:
        normalized_modality = modality.strip().lower()
        if normalized_modality not in self.projections:
            raise KeyError(
                f"Unknown modality {modality!r}; expected one of {self.modalities}"
            )
        validate_encoder_output(
            encoder_output,
            embedding_dim=self.input_dim,
            batch_size=full_batch_size,
            check_values=self.validate_values,
        )

        projection = self.projections[normalized_modality]
        projection_type = self.projection_types[normalized_modality]
        if projection_type == "mlp":
            if not isinstance(projection, UnifiedProjectionMLP):
                raise RuntimeError("ProjectionBank module/type registry is inconsistent")
            return projection(encoder_output.global_embedding)
        if not isinstance(projection, SemanticAttentionProjection):
            raise RuntimeError("ProjectionBank module/type registry is inconsistent")
        return projection(encoder_output.tokens, encoder_output.token_mask)

    def forward_all(
        self,
        encoder_outputs: Mapping[str, EncoderOutput],
        *,
        full_batch_size: int | None = None,
    ) -> dict[str, ProjectionOutput]:
        unknown = set(encoder_outputs) - set(self.modalities)
        if unknown:
            raise KeyError(f"Unknown encoder output modalities: {sorted(unknown)}")
        return {
            modality: self.forward(
                modality,
                encoder_output,
                full_batch_size=full_batch_size,
            )
            for modality, encoder_output in encoder_outputs.items()
        }
