try:
    from .hybrid_model import FullUNet3D, FullUNet3Plus2D, Hybrid3D2DUNet3Plus
    FullUnet3D_3_plus = FullUNet3D
    Hybrid3D2DUNet3PlusSliceInject = Hybrid3D2DUNet3Plus
except ModuleNotFoundError as error:
    if error.name != f"{__name__}.hybrid_model":
        raise
    from .hybrid_model_improve import (
        FullUNet3D,
        FullUnet3D_3_plus,
        FullUNet3Plus2D,
        Hybrid3D2DUNet3Plus,
        Hybrid3D2DUNet3PlusSliceInject,
    )

__all__ = [
    "FullUnet3D_3_plus",
    "FullUNet3D",
    "FullUNet3Plus2D",
    "Hybrid3D2DUNet3Plus",
    "Hybrid3D2DUNet3PlusSliceInject",
]
