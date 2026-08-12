from __future__ import annotations

from typing import Any

import torch

from .common import (
    BaseSegmentationModel,
    SlicePositionEncoder,
    build_2d_decoder,
    build_2d_encoder,
    channels_from_config,
    normalise_2d_encoder_name,
    normalise_decoder_style,
)


class Experiment2DSegModel(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_type: str = "unet",
        encoder_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 16,
        decoder_style: str | None = None,
        normalization: str = "batchnorm",
        dropout: float = 0.0,
        use_position_encoder: bool = False,
        position_embedding_dim: int = 32,
        max_position_embeddings: int = 512,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_type = normalise_2d_encoder_name(encoder_type)
        self.channels = channels_from_config(encoder_channels or channels, base_channels)
        if decoder_style is None:
            decoder_style = {"unet": "same_scale", "unetpp": "nested_dense", "unet3plus": "full_scale", "nnunet": "same_scale"}[self.encoder_type]
        self.decoder_style = normalise_decoder_style(decoder_style)
        self.encoder_2d = build_2d_encoder(
            self.encoder_type,
            in_channels=in_channels,
            channels=self.channels,
            base_channels=base_channels,
            normalization=normalization,
            dropout=float(dropout),
        )
        self.decoder_2d = build_2d_decoder(
            self.decoder_style,
            decoder_model=self.encoder_type,
            channels=self.channels,
            num_classes=num_classes,
            normalization=normalization,
            dropout=float(dropout),
        )
        self.use_position_encoder = bool(use_position_encoder)
        self.expects_slice_indices = self.use_position_encoder
        self.position_encoder = (
            SlicePositionEncoder(self.channels, embedding_dim=position_embedding_dim, max_positions=max_position_embeddings)
            if self.use_position_encoder
            else None
        )
        self.model_name = f"proposal_exp_2d_{self.encoder_type}"
        self.backbone_name = f"{self.encoder_type}2d_encoder"
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            encoder_type=self.encoder_type,
            decoder_style=self.decoder_style,
            channels=self.channels,
            normalization=normalization,
            dropout=float(dropout),
            use_position_encoder=self.use_position_encoder,
        )

    def _position_injection(self, slice_indices: Any | None):
        if self.position_encoder is None:
            return None

        def inject(feature: torch.Tensor) -> torch.Tensor:
            return self.position_encoder.forward_feature(feature, slice_indices, projection_index=0)

        return inject

    def forward(self, x: torch.Tensor, return_features: bool = False, slice_indices: Any | None = None, slice_index: Any | None = None):
        if slice_indices is None:
            slice_indices = slice_index
        features = self.encoder_2d.forward_with_position(x, self._position_injection(slice_indices))
        logits = self.decoder_2d(features)
        if return_features:
            return self.build_output(logits, features={"encoder": features})
        return logits


FullExperiment2D = Experiment2DSegModel

__all__ = ["Experiment2DSegModel", "FullExperiment2D"]
