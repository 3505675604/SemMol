"""Shared MLP projection and explicit modality-specific SAP variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from torch import nn

from src.models.encoders.common import EncoderOutput, validate_encoder_output

from .sap_projection import SemanticAttentionProjection
from .unified_projection import ProjectionOutput, UnifiedProjectionMLP


class ProjectionBank(nn.Module):
    """Map the molecular modalities through one shared MLP by default.

    ``share_mlp_weights=False`` selects independent MLPs. A modality whose
    type is ``sap`` instead receives its own local-token attention projector.
    Modules are registered exactly once, so a shared projector has one set
    of checkpoint keys and cannot be overwritten through modality aliases.
    """

    MODALITIES = ("1d", "2d", "3d", "qm")

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 256,
        *,
        modalities: Sequence[str] | None = None,
        projection_types: Mapping[str, str] | None = None,
        share_mlp_weights: bool = True,
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
            "3d": "mlp",
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

        if not math.isfinite(normalize_eps) or normalize_eps <= 0.0:
            raise ValueError("normalize_eps must be finite and positive")
        if not isinstance(share_mlp_weights, bool):
            raise TypeError("share_mlp_weights must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        mlp_options = dict(mlp or {})
        sap_options = dict(sap or {})
        reserved = {"input_dim", "output_dim"}
        if reserved.intersection(mlp_options):
            raise ValueError("mlp options cannot override input_dim or output_dim")
        if reserved.intersection(sap_options):
            raise ValueError("sap options cannot override input_dim or output_dim")
        for name, options in (("mlp", mlp_options), ("sap", sap_options)):
            if options.get("normalize_eps", normalize_eps) != normalize_eps:
                raise ValueError(
                    f"{name}.normalize_eps must equal projection.normalize_eps"
                )
        mlp_options.setdefault("normalize_eps", normalize_eps)
        mlp_options.setdefault("validate_values", validate_values)
        sap_options.setdefault("normalize_eps", normalize_eps)
        sap_options.setdefault("validate_values", validate_values)

        modules: dict[str, nn.Module] = {}
        projection_keys: dict[str, str] = {}
        for modality in selected_modalities:
            projection_type = selected_types[modality]
            if projection_type == "mlp":
                key = "shared_mlp" if share_mlp_weights else modality
                if key not in modules:
                    modules[key] = UnifiedProjectionMLP(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        **mlp_options,
                    )
            elif projection_type == "sap":
                key = modality
                modules[key] = SemanticAttentionProjection(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **sap_options,
                )
            else:
                raise ValueError(
                    f"Unsupported projection type {projection_type!r} for {modality}; "
                    "expected 'mlp' or 'sap'"
                )
            projection_keys[modality] = key

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_eps = float(normalize_eps)
        self.share_mlp_weights = share_mlp_weights
        self.modalities = tuple(selected_modalities)
        self.projection_types = {
            modality: selected_types[modality]
            for modality in self.modalities
        }
        self.validate_values = validate_values
        self.projectors = nn.ModuleDict(modules)
        self._projection_keys = projection_keys

    @property
    def projections(self) -> Mapping[str, nn.Module]:
        """Expose modality lookup without registering shared module aliases."""

        return {
            modality: self.projectors[key]
            for modality, key in self._projection_keys.items()
        }

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
