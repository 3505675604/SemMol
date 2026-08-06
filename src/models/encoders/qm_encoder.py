"""Residual 3D encoder for fixed-grid electron-density fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import (
    EncoderOutput,
    compact_sample_index,
    validate_encoder_output,
)


def _group_count(channels: int, maximum: int = 8) -> int:
    if channels <= 0:
        raise ValueError("channels must be positive")
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    """Group-normalized residual block with optional spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            _group_count(out_channels),
            out_channels,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            _group_count(out_channels),
            out_channels,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(out_channels),
                    out_channels,
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.dropout(features)
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class ChannelAttention3D(nn.Module):
    """Low-resolution squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        hidden_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.gate(self.pool(inputs))


@dataclass(frozen=True)
class QMEncoderOutput(EncoderOutput):
    """Compact density features and the pre-transform density integral."""

    electron_count: Tensor


class QMEncoder(nn.Module):
    """Encode nonnegative 32-cubed density grids with an early-downsample ResNet."""

    def __init__(
        self,
        in_channels: int = 1,
        grid_size: int = 32,
        base_channels: int = 64,
        hidden_size: int = 512,
        num_res_blocks: int = 4,
        use_attention: bool = True,
        dropout: float = 0.1,
        log1p_input: bool = True,
        voxel_spacing: float = 0.75,
        density_channel: int = 0,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0 or hidden_size <= 0:
            raise ValueError(
                "in_channels, base_channels, and hidden_size must be positive"
            )
        if grid_size != 32:
            raise ValueError(
                "QMEncoder requires grid_size=32 for the 32->16->8->4 "
                "feature hierarchy"
            )
        if num_res_blocks < 3:
            raise ValueError(
                "num_res_blocks must be at least 3 to populate all resolutions"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(use_attention, bool):
            raise TypeError("use_attention must be bool")
        if not isinstance(log1p_input, bool):
            raise TypeError("log1p_input must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        if not math.isfinite(voxel_spacing) or voxel_spacing <= 0.0:
            raise ValueError("voxel_spacing must be positive and finite")
        if not 0 <= density_channel < in_channels:
            raise ValueError("density_channel must index an input channel")

        self.in_channels = int(in_channels)
        self.grid_size = int(grid_size)
        self.hidden_size = int(hidden_size)
        self.log1p_input = bool(log1p_input)
        self.voxel_volume = float(voxel_spacing) ** 3
        self.density_channel = int(density_channel)
        self.validate_values = validate_values

        self.stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                base_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(base_channels),
                base_channels,
            ),
            nn.SiLU(),
        )
        self.stage_16 = ResidualBlock3D(
            base_channels,
            base_channels,
            dropout=dropout,
        )
        self.stage_8 = ResidualBlock3D(
            base_channels,
            base_channels * 2,
            stride=2,
            dropout=dropout,
        )
        stage_4_blocks = [
            ResidualBlock3D(
                base_channels * 2,
                base_channels * 4,
                stride=2,
                dropout=dropout,
            )
        ]
        stage_4_blocks.extend(
            ResidualBlock3D(
                base_channels * 4,
                base_channels * 4,
                dropout=dropout,
            )
            for _ in range(num_res_blocks - 3)
        )
        self.stage_4 = nn.Sequential(*stage_4_blocks)

        final_channels = base_channels * 4
        self.channel_attention = (
            ChannelAttention3D(final_channels)
            if use_attention
            else nn.Identity()
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(final_channels),
            nn.Linear(final_channels, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(final_channels),
            nn.Linear(final_channels, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.apply(self._initialize_module)
        for module in self.modules():
            if isinstance(module, ResidualBlock3D):
                nn.init.zeros_(module.norm2.weight)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_inputs(self, qm_grid: Tensor, qm_mask: Tensor) -> None:
        if qm_grid.ndim != 5:
            raise ValueError(
                "qm_grid must have shape "
                f"[batch, {self.in_channels}, {self.grid_size}, "
                f"{self.grid_size}, {self.grid_size}], got "
                f"{tuple(qm_grid.shape)}"
            )
        expected_shape = (
            qm_grid.shape[0],
            self.in_channels,
            self.grid_size,
            self.grid_size,
            self.grid_size,
        )
        if tuple(qm_grid.shape) != expected_shape:
            raise ValueError(
                f"qm_grid must have shape [batch, {self.in_channels}, "
                f"{self.grid_size}, {self.grid_size}, {self.grid_size}], got "
                f"{tuple(qm_grid.shape)}"
            )
        if qm_grid.dtype != torch.float32:
            raise TypeError(f"qm_grid must be torch.float32, got {qm_grid.dtype}")
        expected_mask_shape = (qm_grid.shape[0],)
        if tuple(qm_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"qm_mask must have shape {expected_mask_shape}, got "
                f"{tuple(qm_mask.shape)}"
            )
        if qm_mask.dtype != torch.bool:
            raise TypeError(f"qm_mask must be bool, got {qm_mask.dtype}")
        if qm_grid.device != qm_mask.device:
            raise ValueError("qm_grid and qm_mask must be on the same device")
        parameter_device = self.stem[0].weight.device
        if qm_grid.device != parameter_device:
            raise ValueError(
                f"qm_grid is on {qm_grid.device}, but QMEncoder is on "
                f"{parameter_device}"
            )
        if self.validate_values:
            if not bool(torch.isfinite(qm_grid).all()):
                raise ValueError("qm_grid must contain only finite values")
            if bool(torch.any(qm_grid < 0)):
                raise ValueError(
                    "electron-density grids cannot contain negatives"
                )
            if bool(torch.any(qm_grid[~qm_mask] != 0)):
                raise ValueError(
                    "qm_grid rows outside qm_mask must be zero padded"
                )
            if bool(
                torch.any(
                    qm_grid[
                        qm_mask,
                        self.density_channel,
                    ].sum(dim=(1, 2, 3))
                    <= 0
                )
            ):
                raise ValueError(
                    "every valid density grid must have positive integrated mass"
                )

    def _empty_output(
        self,
        qm_grid: Tensor,
        sample_index: Tensor,
        batch_size: int,
    ) -> QMEncoderOutput:
        token_count = (self.grid_size // 8) ** 3
        output = QMEncoderOutput(
            global_embedding=qm_grid.new_empty((0, self.hidden_size)),
            sample_index=sample_index,
            tokens=qm_grid.new_empty(
                (0, token_count, self.hidden_size)
            ),
            token_mask=torch.empty(
                (0, token_count),
                dtype=torch.bool,
                device=qm_grid.device,
            ),
            electron_count=torch.empty(
                (0,),
                dtype=torch.float32,
                device=qm_grid.device,
            ),
        )
        validate_encoder_output(
            output,
            embedding_dim=self.hidden_size,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        return output

    def forward(
        self,
        qm_grid: Tensor,
        qm_mask: Tensor,
    ) -> QMEncoderOutput:
        """Return compact global and 4-cubed spatial density features."""

        self._validate_inputs(qm_grid, qm_mask)
        batch_size = int(qm_grid.shape[0])
        sample_index = compact_sample_index(qm_mask)
        if sample_index.numel() == 0:
            return self._empty_output(qm_grid, sample_index, batch_size)

        valid_grid = qm_grid.index_select(0, sample_index)
        electron_count = (
            valid_grid[:, self.density_channel]
            .float()
            .sum(dim=(1, 2, 3))
            * self.voxel_volume
        )
        features = torch.log1p(valid_grid) if self.log1p_input else valid_grid
        features = self.stem(features)
        features = self.stage_16(features)
        features = self.stage_8(features)
        features = self.stage_4(features)
        features = self.channel_attention(features)
        if tuple(features.shape[2:]) != (4, 4, 4):
            raise RuntimeError(
                "QM feature hierarchy must end at a 4x4x4 spatial grid, got "
                f"{tuple(features.shape[2:])}"
            )

        pooled_features = features.mean(dim=(2, 3, 4))
        global_embedding = self.global_projection(pooled_features)
        spatial_features = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.token_projection(spatial_features)
        token_mask = torch.ones(
            tokens.shape[:2],
            dtype=torch.bool,
            device=tokens.device,
        )

        output = QMEncoderOutput(
            global_embedding=global_embedding,
            sample_index=sample_index,
            tokens=tokens,
            token_mask=token_mask,
            electron_count=electron_count,
        )
        validate_encoder_output(
            output,
            embedding_dim=self.hidden_size,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        return output
