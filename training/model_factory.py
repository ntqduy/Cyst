from __future__ import annotations

import importlib
import inspect
import sys
import types
from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn

from .utils import get_nested, project_root


@dataclass
class ModelBuildResult:
    model: nn.Module
    name: str
    backbone: str
    in_channels: int
    num_classes: int


_FIXED_BACKBONE_LABELS = {
    "default",
    "unet_encoder",
    "residual_encoder",
    "vnet_encoder",
    "vit_encoder",
    "transunet_encoder",
    "attention_unet_encoder",
    "recurrent_residual_unet_encoder",
    "nnunet2d_encoder",
    "unet3plus_encoder",
    "swin_transformer",
    "swin_unet_encoder",
    "unet3d_encoder",
    "vnet3d_encoder",
    "vit3d_encoder",
    "swin_transformer_3d",
    "mamba_3d_encoder",
    "nnunet3d_encoder",
    "mobile_style_2d3d_encoder",
    "mobi_style_3d_encoder",
    "mini_unet3d_encoder",
    "adaptive_kernel_moe_encoder",
    "adaptive_kernel_moe_add_encoder",
    "adaptive_kernel_moe_other_encoder",
    "adaptive_kernel_moe_dynamic_encoder",
    "adaptive_kernel_softmoe_encoder",
    "unet3plus2d_encoder",
    "hybrid_unet3plus_2d3d_encoder",
    "hybrid_unet3plus_2d3d_slice_inject_encoder",
    "proposal_experiment_2d_encoder",
    "proposal_experiment_3d_encoder",
    "proposal_experiment_hybrid_encoder",
    "unet3plus_slice_encoder",
}


def _encoder_backbone_or_default(backbone: str, default: str) -> str:
    value = str(backbone or "").strip()
    if not value or value.lower() in _FIXED_BACKBONE_LABELS:
        return default
    return value


def _backbone_or_default(backbone: str, default: str) -> str:
    value = str(backbone or "").strip()
    if not value or value.lower() == "default":
        return default
    return value


def _channels_from_base(base_channels: Any, stages: int = 5, cap: int | None = None) -> tuple[int, ...]:
    base = int(base_channels)
    channels = []
    for index in range(int(stages)):
        value = base * (2**index)
        channels.append(min(value, int(cap)) if cap is not None else value)
    return tuple(channels)


def _clear_network_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "networks"
            or module_name.startswith("networks.")
            or module_name == "utils"
            or module_name.startswith("utils.")
        ):
            del sys.modules[module_name]


def _prepare_imports(model_type: str) -> None:
    root = project_root()
    baseline_root = root / "baseline"
    network_parent = baseline_root / ("2D-Model" if model_type == "2D" else "3D-Model")
    for path in (str(network_parent), str(baseline_root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    _clear_network_modules()
    networks_pkg = types.ModuleType("networks")
    networks_pkg.__path__ = [str(network_parent)]  # type: ignore[attr-defined]
    networks_pkg.__package__ = "networks"
    sys.modules["networks"] = networks_pkg


def _filter_kwargs(cls, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _import_proposal_hybrid_module(prefer_improve: bool = False):
    module_names = (
        ("Proposal_hybrid_3D_2D.hybrid_model_improve", "Proposal_hybrid_3D_2D.hybrid_model")
        if prefer_improve
        else ("Proposal_hybrid_3D_2D.hybrid_model", "Proposal_hybrid_3D_2D.hybrid_model_improve")
    )
    last_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            last_error = error
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError("No Proposal_hybrid_3D_2D model module found.")


def _instantiate_2d(name: str, backbone: str, in_channels: int, num_classes: int, image_size, model_args: dict[str, Any]) -> nn.Module:
    proposal_2d_names = {
        "full_unet3plus_2d",
        "full_unet_3_plus_2d",
        "fullunet3plus2d",
        "proposal_full_unet3plus_2d",
    }
    proposal_experiment_2d_names = {
        "proposal_experiment_2d",
        "proposal_model_experiment_2d",
        "proposal_exp_2d",
    }
    proposal_method_2d_names = {
        "proposal_method_2d",
        "proposal_method_exp_2d",
    }
    unet_names = {"unet"}
    unet_plus_plus_names = {"unet_plus_plus", "unet++", "unet_plus_pluss", "unet_pluss_pluss", "unetplusplus"}
    unet_3_plus_names = {
        "unet_3_plus",
        "unet3plus",
        "unet_3plus",
        "unet+++",
        "unet_plus_plus_plus",
        "unet_3_plus_cgm",
        "unet3plus_cgm",
        "unet3plus_hybrid_cgm",
        "unet_3_plus_hybrid_cgm",
    }
    swin_unet_names = {"swin_unet", "swinunet", "swin-unet"}
    nnunet_names = {"nnunet", "nn_unet", "nnunet2d", "nn_unet2d"}
    deeplab_names = {"deeplab", "deeplabv3"}
    deeplab_plus_plus_names = {
        "deeplab_plus_plus",
        "deeplab++",
        "deeplabv3plus",
        "deeplabv3+",
        "deeplab_plus_pluss",
        "deeplabplusplus",
    }
    if name in proposal_2d_names:
        module = _import_proposal_hybrid_module()
        cls = getattr(module, "FullUNet3Plus2D")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_experiment_2d_names:
        module = importlib.import_module("Proposal_Model_Experiment.HybridModel.hybrid_model")
        cls = getattr(module, "Experiment2DSegModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_method_2d_names:
        module = importlib.import_module("Proposal_Method.HybridModel.hybrid_model")
        cls = getattr(module, "Experiment2DSegModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)

    aliases = {
        "unet": ("networks.unet", "UNet2D"),
        "resunet": ("networks.residual_unet", "ResidualUNet2D"),
        "residual_unet": ("networks.residual_unet", "ResidualUNet2D"),
        "vnet": ("networks.VNet", "VNet2D"),
        "unetr": ("networks.unetr", "UNETR2D"),
        "transunet": ("networks.transunet", "TransUNet2D"),
        "trans_unet": ("networks.transunet", "TransUNet2D"),
        "unet_resnet152": ("networks.Unet_restnet", "UNetResNet152"),
        "resnet152_unet": ("networks.Unet_restnet", "UNetResNet152"),
        "att_unet": ("networks.attention_unet", "AttentionUNet2D"),
        "attention_unet": ("networks.attention_unet", "AttentionUNet2D"),
        "r2unet": ("networks.attention_unet", "R2UNet2D"),
        "unet_plus_plus": ("networks.unet_plus_plus", "UNetPlusPlus2D"),
        "unet++": ("networks.unet_plus_plus", "UNetPlusPlus2D"),
        "unet_plus_pluss": ("networks.unet_plus_plus", "UNetPlusPlus2D"),
        "unet_pluss_pluss": ("networks.unet_plus_plus", "UNetPlusPlus2D"),
        "unetplusplus": ("networks.unet_plus_plus", "UNetPlusPlus2D"),
        "unet_3_plus": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet3plus": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet_3plus": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet+++": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet_plus_plus_plus": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet_3_plus_cgm": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet3plus_cgm": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet3plus_hybrid_cgm": ("networks.unet_3_plus", "UNet3Plus2D"),
        "unet_3_plus_hybrid_cgm": ("networks.unet_3_plus", "UNet3Plus2D"),
        "swin_unet": ("networks.swin_unet", "SwinUNet2D"),
        "swinunet": ("networks.swin_unet", "SwinUNet2D"),
        "swin-unet": ("networks.swin_unet", "SwinUNet2D"),
        "deeplab": ("networks.deeplab", "DeepLab2D"),
        "deeplabv3": ("networks.deeplab", "DeepLab2D"),
        "deeplab_plus_plus": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "deeplab++": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "deeplabv3plus": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "deeplabv3+": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "deeplab_plus_pluss": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "deeplabplusplus": ("networks.deeplab_plus_plus", "DeepLabPlusPlus2D"),
        "nnunet": ("networks.nnUnet", "NNUNet2D"),
        "nn_unet": ("networks.nnUnet", "NNUNet2D"),
        "nnunet2d": ("networks.nnUnet", "NNUNet2D"),
        "nn_unet2d": ("networks.nnUnet", "NNUNet2D"),
    }
    if name not in aliases:
        raise ValueError(f"Unsupported 2D model '{name}'. Available: {', '.join(sorted(aliases))}")
    module_name, class_name = aliases[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    kwargs = dict(model_args)
    if name == "unetr":
        kwargs.setdefault("image_size", image_size[:2])
    if name in {"transunet", "trans_unet"}:
        kwargs.setdefault("image_size", image_size[:2])
    if name in unet_names:
        kwargs.setdefault("backbone", _backbone_or_default(backbone, "unet_encoder"))
    if name in unet_plus_plus_names:
        kwargs.setdefault("encoder_name", _backbone_or_default(backbone, "unet_encoder"))
        if kwargs.get("encoder_pretrained") is False:
            kwargs.setdefault("encoder_weights", None)
    if name in unet_3_plus_names:
        kwargs.setdefault("backbone", "unet3plus_encoder")
        if "cgm" in name or "hybrid_cgm" in name:
            kwargs.setdefault("deep_supervision", True)
            kwargs.setdefault("cgm", True)
    if name in swin_unet_names:
        kwargs.setdefault("image_size", image_size[:2])
        kwargs.setdefault("backbone", _encoder_backbone_or_default(backbone, "swin_transformer"))
    if name in deeplab_names | deeplab_plus_plus_names:
        kwargs.setdefault("backbone", _encoder_backbone_or_default(backbone, "resnet50"))
        kwargs.setdefault("pretrained_backbone", True)
    if name in nnunet_names and "base_channels" in kwargs and "feature_channels" not in kwargs:
        kwargs["feature_channels"] = _channels_from_base(kwargs["base_channels"], cap=320)
    kwargs = _filter_kwargs(cls, kwargs)
    kwargs.pop("in_channels", None)
    kwargs.pop("num_classes", None)
    return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)


def _instantiate_3d(name: str, in_channels: int, num_classes: int, image_size, model_args: dict[str, Any]) -> nn.Module:
    nnunet_names = {"nnunet", "nn_unet", "nnunet3d", "nn_unet3d"}
    proposal_full_3d_names = {
        "full_unet3d",
        "full_unet3d_3_plus",
        "full_unet3d_3plus",
        "full_unet_3d_3_plus",
        "fullunet3d_3_plus",
        "fullunet3d_3plus",
        "fullunet3d3plus",
        "full_unet_3d",
        "fullunet3d",
        "proposal_full_unet3d",
        "proposal_full_unet3d_3_plus",
    }
    proposal_hybrid_names = {
        "hybrid_3d_2d",
        "hybrid3d2d",
        "hybrid_2d_3d",
        "hybrid3d2d_unet3plus",
        "hybrid_3d_2d_unet3plus",
        "proposal_hybrid_3d_2d",
        "proposal_hybrid_3d_2d_unet3plus",
    }
    proposal_hybrid_improve_names = {
        "hybrid_3d_2d_improve",
        "hybrid_3d_2d_slice_inject",
        "hybrid3d2dunet3plussliceinject",
        "hybrid_3d_2d_unet3plus_slice_inject",
        "hybrid3d2d_unet3plus_slice_inject",
        "proposal_hybrid_3d_2d_improve",
        "proposal_hybrid_3d_2d_slice_inject",
    }
    proposal_experiment_3d_names = {
        "proposal_experiment_3d",
        "proposal_model_experiment_3d",
        "proposal_exp_3d",
    }
    proposal_experiment_hybrid_names = {
        "proposal_experiment_hybrid",
        "proposal_model_experiment_hybrid",
        "proposal_exp_hybrid",
    }
    proposal_method_3d_names = {
        "proposal_method_3d",
        "proposal_method_exp_3d",
    }
    proposal_method_hybrid_names = {
        "proposal_method_hybrid",
        "proposal_method_exp_hybrid",
    }
    mini_unet_names = {
        "mini_unet",
        "mini_unet3d",
        "mini_unet_3d",
        "miniunet3d",
    }
    adaptive_kernel_moe_names = {
        "adaptive_kernel_moe",
        "adap_kernel_moe",
        "multi_kernel_moe",
        "multi_kenel_moe",
        "unet_moe",
        "unet3d_moe",
    }
    adaptive_kernel_moe_add_names = {
        "adaptive_kernel_moe_add",
        "adap_kernel_moe_add",
        "multi_kernel_moe_add",
        "multi_kenel_moe_add",
        "unet_moe_add",
        "unet3d_moe_add",
    }
    adaptive_kernel_moe_other_names = {
        "adaptive_kernel_moe_other",
        "adap_kernel_moe_other",
        "multi_kernel_moe_other",
        "multi_kenel_moe_other",
        "unet_moe_other",
        "unet3d_moe_other",
    }
    adaptive_kernel_moe_dynamic_names = {
        "adaptive_kernel_moe_dynamic",
        "adap_kernel_moe_dynamic",
        "multi_kernel_moe_dynamic",
        "multi_kenel_moe_dynamic",
        "unet_moe_dynamic",
        "unet3d_moe_dynamic",
    }
    adaptive_kernel_softmoe_names = {
        "adaptive_kernel_softmoe",
        "adap_kernel_softmoe",
        "multi_kernel_softmoe",
        "multi_kenel_softmoe",
        "unet_softmoe",
        "unet3d_softmoe",
    }
    unet3plus_slice_names = {
        "unet_3_plus_slice",
        "unet3plus_slice",
        "unet_3plus_slice",
        "unet+++_slice",
        "unet_3_plus_proposal_slice",
        "unet3plus_proposal_slice",
    }
    mobi_style_names = {
        "mobi_style_3d",
        "mobi_style_3d_3_plus",
        "mobi_style_3d_v3",
        "mobile_style_3d",
        "mobile_style_3d_3_plus",
        "mobile_style_3d_v3",
        "mobile_style_2d3d",
        "mobi_style_2d3d",
        "unet3d_mobile",
        "unet_3d_mobile",
    }
    if name == "vnet":
        module = importlib.import_module("networks.VNet")
        cls = getattr(module, "VNet")
        kwargs = dict(model_args)
        if "base_channels" in kwargs and "n_filters" not in kwargs:
            kwargs["n_filters"] = kwargs.pop("base_channels")
        kwargs.setdefault("n_channels", in_channels)
        kwargs.setdefault("n_classes", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in {"swin_unetr", "swinunetr", "swin-unetr"}:
        module = importlib.import_module("networks.swin_unetr")
        cls = getattr(module, "SwinUNETR3D")
        kwargs = dict(model_args)
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("num_classes", num_classes)
        kwargs.setdefault("image_size", tuple(image_size[:3]))
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in {"segmamba", "seg_mamba", "seg-mamba"}:
        module = importlib.import_module("networks.segmamba")
        cls = getattr(module, "SegMamba3D")
        kwargs = dict(model_args)
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("num_classes", num_classes)
        kwargs.setdefault("image_size", tuple(image_size[:3]))
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in {"unet", "unet3d", "unet_3d"}:
        module = importlib.import_module("networks.Unet3D")
        cls = getattr(module, "UNet")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name == "unetr":
        module = importlib.import_module("networks.unetr")
        cls = getattr(module, "UNETR")
        kwargs = dict(model_args)
        kwargs.setdefault("img_shape", tuple(image_size[:3]))
        kwargs.setdefault("input_dim", in_channels)
        kwargs.setdefault("output_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in nnunet_names:
        module = importlib.import_module("networks.nnUnet")
        cls = getattr(module, "NNUNet3D")
        kwargs = dict(model_args)
        if "base_channels" in kwargs and "feature_channels" not in kwargs:
            kwargs["feature_channels"] = _channels_from_base(kwargs["base_channels"], cap=320)
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("num_classes", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in proposal_full_3d_names:
        module = _import_proposal_hybrid_module(prefer_improve="3_plus" in name or "3plus" in name)
        cls = getattr(module, "FullUnet3D_3_plus", None)
        if cls is None:
            cls = getattr(module, "FullUNet3D")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_hybrid_names:
        module = _import_proposal_hybrid_module()
        cls = getattr(module, "Hybrid3D2DUNet3Plus")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_hybrid_improve_names:
        module = _import_proposal_hybrid_module(prefer_improve=True)
        cls = getattr(module, "Hybrid3D2DUNet3PlusSliceInject", None)
        if cls is None:
            cls = getattr(module, "Hybrid3D2DUNet3Plus")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_experiment_3d_names:
        module = importlib.import_module("Proposal_Model_Experiment.HybridModel.hybrid_model")
        cls = getattr(module, "Experiment3DSegModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_experiment_hybrid_names:
        module = importlib.import_module("Proposal_Model_Experiment.HybridModel.hybrid_model")
        cls = getattr(module, "ExperimentHybridModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_method_3d_names:
        module = importlib.import_module("Proposal_Method.HybridModel.hybrid_model")
        cls = getattr(module, "Experiment3DSegModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in proposal_method_hybrid_names:
        module = importlib.import_module("Proposal_Method.HybridModel.hybrid_model")
        cls = getattr(module, "ExperimentHybridModel")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in mini_unet_names:
        module = importlib.import_module("Proposal.mini_unet")
        cls = getattr(module, "MiniUnet3D")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in adaptive_kernel_moe_names:
        module = importlib.import_module("Multi_kenel_MoE.adap_kernel_moe")
        cls = getattr(module, "UNetMoE")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in adaptive_kernel_moe_add_names:
        module = importlib.import_module("Multi_kenel_MoE.adap_kernel_moe_add")
        cls = getattr(module, "UNetMoE")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in adaptive_kernel_moe_other_names:
        module = importlib.import_module("Multi_kenel_MoE.adap_kernel_moe_other")
        cls = getattr(module, "UNetMoE")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in adaptive_kernel_moe_dynamic_names:
        module = importlib.import_module("Multi_kenel_MoE.adap_kernel_moe_dynamic")
        cls = getattr(module, "UNetMoE")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in adaptive_kernel_softmoe_names:
        module = importlib.import_module("Multi_kenel_MoE.adap_kernel_softmoe")
        cls = getattr(module, "UNetMoE")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128, 256))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        return cls(**kwargs)
    if name in unet3plus_slice_names:
        module = importlib.import_module("Unet_3_plus_Slice.unet_3_plus_slice")
        cls = getattr(module, "UNet3PlusSlice")
        kwargs = _filter_kwargs(cls, model_args)
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if name in mobi_style_names:
        module_name = (
            "Proposal.mobi_style_3D_3_plus"
            if name in {"mobi_style_3d_3_plus", "mobile_style_3d_3_plus"}
            else "Proposal.mobi_style_3D"
        )
        module = importlib.import_module(module_name)
        cls = getattr(module, "UNet3D_Mobile")
        kwargs = dict(model_args)
        if "feature_channels" in kwargs and "feat_channels" not in kwargs:
            kwargs["feat_channels"] = tuple(kwargs.pop("feature_channels"))
        kwargs.setdefault("feat_channels", (16, 32, 64, 128))
        kwargs.setdefault("in_dim", in_channels)
        kwargs.setdefault("out_dim", num_classes)
        kwargs = _filter_kwargs(cls, kwargs)
        model = cls(**kwargs)
        if name in {"mobi_style_3d_v3", "mobile_style_3d_v3"}:
            setattr(model, "model_name", "mobi_style_3d_v3")
        if name in {"mobi_style_3d_3_plus", "mobile_style_3d_3_plus"}:
            setattr(model, "model_name", "mobi_style_3d_3_plus")
        return model
    raise ValueError(
        "Unsupported 3D model '{}'. Available: vnet, unet, unet3d, unetr, nnunet, nnunet3d, "
        "mini_unet3d, adaptive_kernel_moe, adaptive_kernel_softmoe, mobi_style_3d, "
        "mobi_style_3d_3_plus, mobi_style_3d_v3, full_unet3d, full_unet3d_3_plus, hybrid_3d_2d, "
        "unet_3_plus_slice, swin_unetr, segmamba".format(name)
    )


def _architecture_config_value(model: nn.Module, *keys: str) -> str | None:
    architecture_config = getattr(model, "architecture_config", None)
    if not isinstance(architecture_config, Mapping):
        return None
    for key in keys:
        value = architecture_config.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _resolved_backbone_name(model_type: str, name: str, model: nn.Module, configured_backbone: str) -> str:
    model_backbone = getattr(model, "backbone_name", None)
    if model_backbone:
        return str(model_backbone)
    architecture_backbone = _architecture_config_value(model, "backbone", "encoder_name", "encoder")
    if architecture_backbone:
        return architecture_backbone

    if model_type == "3D":
        fixed_backbones = {
            "vnet": "vnet3d_encoder",
            "unet": "unet3d_encoder",
            "unet3d": "unet3d_encoder",
            "unet_3d": "unet3d_encoder",
            "unetr": "vit3d_encoder",
            "swin_unetr": "swin_transformer_3d",
            "swinunetr": "swin_transformer_3d",
            "swin-unetr": "swin_transformer_3d",
            "segmamba": "mamba_3d_encoder",
            "seg_mamba": "mamba_3d_encoder",
            "seg-mamba": "mamba_3d_encoder",
            "nnunet": "nnunet3d_encoder",
            "nn_unet": "nnunet3d_encoder",
            "nnunet3d": "nnunet3d_encoder",
            "nn_unet3d": "nnunet3d_encoder",
            "mobi_style_3d": "mobile_style_2d3d_encoder",
            "mobi_style_3d_3_plus": "mobile_style_2d3d_unet3plus_encoder",
            "mobi_style_3d_v3": "mobile_style_2d3d_encoder",
            "mobile_style_3d": "mobile_style_2d3d_encoder",
            "mobile_style_3d_3_plus": "mobile_style_2d3d_unet3plus_encoder",
            "mobile_style_3d_v3": "mobile_style_2d3d_encoder",
            "mobile_style_2d3d": "mobile_style_2d3d_encoder",
            "mobi_style_2d3d": "mobile_style_2d3d_encoder",
            "unet3d_mobile": "mobile_style_2d3d_encoder",
            "unet_3d_mobile": "mobile_style_2d3d_encoder",
            "hybrid_3d_2d_improve": "hybrid_unet3plus_2d3d_slice_inject_encoder",
            "hybrid_3d_2d_slice_inject": "hybrid_unet3plus_2d3d_slice_inject_encoder",
            "proposal_hybrid_3d_2d_improve": "hybrid_unet3plus_2d3d_slice_inject_encoder",
            "proposal_hybrid_3d_2d_slice_inject": "hybrid_unet3plus_2d3d_slice_inject_encoder",
            "mini_unet": "mini_unet3d_encoder",
            "mini_unet3d": "mini_unet3d_encoder",
            "mini_unet_3d": "mini_unet3d_encoder",
            "miniunet3d": "mini_unet3d_encoder",
            "adaptive_kernel_moe": "adaptive_kernel_moe_encoder",
            "adap_kernel_moe": "adaptive_kernel_moe_encoder",
            "multi_kernel_moe": "adaptive_kernel_moe_encoder",
            "multi_kenel_moe": "adaptive_kernel_moe_encoder",
            "unet_moe": "adaptive_kernel_moe_encoder",
            "unet3d_moe": "adaptive_kernel_moe_encoder",
            "adaptive_kernel_moe_add": "adaptive_kernel_moe_add_encoder",
            "adap_kernel_moe_add": "adaptive_kernel_moe_add_encoder",
            "multi_kernel_moe_add": "adaptive_kernel_moe_add_encoder",
            "multi_kenel_moe_add": "adaptive_kernel_moe_add_encoder",
            "unet_moe_add": "adaptive_kernel_moe_add_encoder",
            "unet3d_moe_add": "adaptive_kernel_moe_add_encoder",
            "adaptive_kernel_moe_other": "adaptive_kernel_moe_other_encoder",
            "adap_kernel_moe_other": "adaptive_kernel_moe_other_encoder",
            "multi_kernel_moe_other": "adaptive_kernel_moe_other_encoder",
            "multi_kenel_moe_other": "adaptive_kernel_moe_other_encoder",
            "unet_moe_other": "adaptive_kernel_moe_other_encoder",
            "unet3d_moe_other": "adaptive_kernel_moe_other_encoder",
            "adaptive_kernel_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "adap_kernel_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "multi_kernel_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "multi_kenel_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "unet_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "unet3d_moe_dynamic": "adaptive_kernel_moe_dynamic_encoder",
            "adaptive_kernel_softmoe": "adaptive_kernel_softmoe_encoder",
            "adap_kernel_softmoe": "adaptive_kernel_softmoe_encoder",
            "multi_kernel_softmoe": "adaptive_kernel_softmoe_encoder",
            "multi_kenel_softmoe": "adaptive_kernel_softmoe_encoder",
            "unet_softmoe": "adaptive_kernel_softmoe_encoder",
            "unet3d_softmoe": "adaptive_kernel_softmoe_encoder",
            "unet_3_plus_slice": "unet3plus_slice_encoder",
            "unet3plus_slice": "unet3plus_slice_encoder",
            "unet_3plus_slice": "unet3plus_slice_encoder",
            "unet+++_slice": "unet3plus_slice_encoder",
            "unet_3_plus_proposal_slice": "unet3plus_slice_encoder",
            "unet3plus_proposal_slice": "unet3plus_slice_encoder",
        }
        return fixed_backbones.get(name, model.__class__.__name__)

    fixed_backbones = {
        "resunet": "residual_encoder",
        "residual_unet": "residual_encoder",
        "vnet": "vnet_encoder",
        "unetr": "vit_encoder",
        "transunet": "transunet_encoder",
        "trans_unet": "transunet_encoder",
        "att_unet": "attention_unet_encoder",
        "attention_unet": "attention_unet_encoder",
        "r2unet": "recurrent_residual_unet_encoder",
        "unet_3_plus": "unet3plus_encoder",
        "unet3plus": "unet3plus_encoder",
        "unet_3plus": "unet3plus_encoder",
        "unet+++": "unet3plus_encoder",
        "unet_plus_plus_plus": "unet3plus_encoder",
        "unet_3_plus_cgm": "unet3plus_encoder",
        "unet3plus_cgm": "unet3plus_encoder",
        "unet3plus_hybrid_cgm": "unet3plus_encoder",
        "unet_3_plus_hybrid_cgm": "unet3plus_encoder",
        "swin_unet": "swin_transformer",
        "swinunet": "swin_transformer",
        "swin-unet": "swin_transformer",
        "nnunet": "nnunet2d_encoder",
        "nn_unet": "nnunet2d_encoder",
        "nnunet2d": "nnunet2d_encoder",
        "nn_unet2d": "nnunet2d_encoder",
    }
    return fixed_backbones.get(name, model.__class__.__name__)


def build_model(cfg: Mapping[str, Any], dataset_in_channels: int) -> ModelBuildResult:
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    experiment_name = str(get_nested(cfg, "experiment.name", get_nested(cfg, "project.name", ""))).lower()
    experiment_stage = str(get_nested(cfg, "experiment.stage", "")).lower()
    if "proposal_hybrid_3d_2d" in experiment_name:
        if experiment_stage == "train_2d":
            model_type = "2D"
        elif experiment_stage in {"train_3d", "hybrid"}:
            model_type = "3D"
    if model_type not in {"2D", "3D"}:
        raise ValueError(f"model.type must be 2D or 3D, got {model_type}")

    _prepare_imports(model_type)

    raw_name = str(get_nested(cfg, "model.name", "unet"))
    proposal_improve = "improve" in experiment_name or "slice_inject" in experiment_name or "improve" in raw_name.lower() or "slice_inject" in raw_name.lower()
    if "proposal_hybrid_3d_2d" in experiment_name:
        if experiment_stage == "train_2d":
            raw_name = "full_unet3plus_2d"
        elif experiment_stage == "train_3d":
            raw_name = "full_unet3d_3_plus"
        elif experiment_stage == "hybrid":
            raw_name = "hybrid_3d_2d_improve" if proposal_improve else "hybrid_3d_2d"
    name = raw_name.lower()
    backbone = str(get_nested(cfg, "model.backbone", "") or "")
    num_classes = int(get_nested(cfg, "model.num_classes", get_nested(cfg, "dataset.num_classes", 2)))
    requested_channels = get_nested(cfg, "model.in_channels", "auto")
    in_channels = int(dataset_in_channels if str(requested_channels).lower() == "auto" else requested_channels)
    image_size = get_nested(cfg, "training.image_size", [256, 256])
    preserve_depth = bool(get_nested(cfg, "training.preserve_depth", get_nested(cfg, "dataset.preserve_depth", False)))
    if model_type == "3D" and not preserve_depth and len(image_size) < 3:
        raise ValueError("training.image_size must contain [height, width, depth] for 3D models.")
    if model_type == "3D" and preserve_depth and len(image_size) < 2:
        raise ValueError("training.image_size must contain [height, width] when training.preserve_depth=true.")
    model_args = dict(get_nested(cfg, "model.args", {}) or {})
    typed_args = get_nested(cfg, f"model.args_{model_type.lower()}", None)
    if isinstance(typed_args, Mapping):
        model_args.update(dict(typed_args))

    if model_type == "2D":
        model = _instantiate_2d(name, backbone, in_channels, num_classes, image_size, model_args)
    else:
        model = _instantiate_3d(name, in_channels, num_classes, image_size, model_args)

    resolved_backbone = _resolved_backbone_name(model_type, name, model, backbone)
    return ModelBuildResult(
        model=model,
        name=str(getattr(model, "model_name", name)),
        backbone=resolved_backbone,
        in_channels=in_channels,
        num_classes=num_classes,
    )
