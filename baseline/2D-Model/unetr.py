from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from utils.model_output import BaseSegmentationModel


class SingleDeconv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SingleConv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2dBlock(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2dBlock(in_channels, out_channels),
            SingleConv2dBlock(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, image_size: Tuple[int, int], patch_size: int, dropout: float) -> None:
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for UNETR2D.")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.reference_grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.reference_grid[0] * self.reference_grid[1], embed_dim))
        self.dropout = nn.Dropout(dropout)
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        else:
            nn.init.normal_(self.position_embeddings, std=0.02)

    def _resize_position_embeddings(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        if grid_size == self.reference_grid:
            return self.position_embeddings
        pos_embed = self.position_embeddings.transpose(1, 2).reshape(1, self.embed_dim, *self.reference_grid)
        pos_embed = F.interpolate(pos_embed, size=grid_size, mode="bilinear", align_corners=False)
        return pos_embed.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError("Input size must be divisible by patch_size for UNETR2D.")
        x = self.proj(x)
        grid_size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self._resize_position_embeddings(grid_size)
        return self.dropout(x), grid_size


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        image_size: Tuple[int, int],
        patch_size: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float,
        extract_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.embeddings = PatchEmbedding2D(in_channels, embed_dim, image_size, patch_size, dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.extract_layers = tuple(extract_layers)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int]]:
        x, grid_size = self.embeddings(x)
        extracted = []
        for depth, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if depth in self.extract_layers:
                extracted.append(x)
        return tuple(extracted), grid_size


class UNETR2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        image_size: Sequence[int] = (256, 256),
        embed_dim: int = 384,
        patch_size: int = 16,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
        extract_layers: Sequence[int] = (3, 6, 9, 12),
    ) -> None:
        super().__init__()
        image_size = (int(image_size[0]), int(image_size[1]))
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.model_name = "unetr"
        self.backbone_name = "vit_encoder"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=list(image_size),
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            extract_layers=list(extract_layers),
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.transformer = TransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            extract_layers=extract_layers,
        )

        self.decoder0 = nn.Sequential(
            Conv2dBlock(in_channels, 32, 3),
            Conv2dBlock(32, 64, 3),
        )
        self.decoder3 = nn.Sequential(
            Deconv2dBlock(embed_dim, 512),
            Deconv2dBlock(512, 256),
            Deconv2dBlock(256, 128),
        )
        self.decoder6 = nn.Sequential(
            Deconv2dBlock(embed_dim, 512),
            Deconv2dBlock(512, 256),
        )
        self.decoder9 = Deconv2dBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2dBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2dBlock(1024, 512),
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            SingleDeconv2dBlock(512, 256),
        )
        self.decoder6_upsampler = nn.Sequential(
            Conv2dBlock(512, 256),
            Conv2dBlock(256, 256),
            SingleDeconv2dBlock(256, 128),
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv2dBlock(256, 128),
            Conv2dBlock(128, 128),
            SingleDeconv2dBlock(128, 64),
        )
        self.header = nn.Sequential(
            Conv2dBlock(128, 64),
            Conv2dBlock(64, 64),
            SingleConv2dBlock(64, num_classes, 1),
        )

    def _tokens_to_feature_map(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, grid_size[0], grid_size[1])

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        transformer_outputs, grid_size = self.transformer(x)
        z3, z6, z9, z12 = [self._tokens_to_feature_map(tokens, grid_size) for tokens in transformer_outputs]
        encoder_features = [z3, z6, z9, z12]

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))

        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))

        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))

        z0 = self.decoder0(x)
        logits = self.header(torch.cat([z0, z3], dim=1))
        return logits, {"encoder": encoder_features, "bottleneck": encoder_features[-1], "decoder": {"final": z0}}

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(logits, features=features)
        if return_features:
            return output.logits, output.features
        return output.logits


UNETR = UNETR2D
