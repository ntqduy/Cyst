"""Proposal segmentation models."""

from .mini_unet import MiniUnet3D
from .mobi_style_3D import UNet3D_Mobile

__all__ = ["MiniUnet3D", "UNet3D_Mobile"]
