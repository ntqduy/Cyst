import math
from collections.abc import Mapping

from torch.nn import Module, ModuleList, Sequential
from torch.nn import BatchNorm3d, Conv3d, ConvTranspose3d, Dropout3d, MaxPool3d
from torch.nn import GELU
import torch
import torch.nn.functional as F


def _match_spatial_3d(source, reference):
    if source.shape[-3:] == reference.shape[-3:]:
        return source
    return torch.nn.functional.interpolate(source, size=reference.shape[-3:], mode="trilinear", align_corners=False)

def _kernel_tuple(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    return tuple(int(item) for item in kernel_size)


def _dilation_tuple(dilation):
    if isinstance(dilation, int):
        return (dilation, dilation, dilation)
    return tuple(int(item) for item in dilation)


def _same_padding(kernel_size, dilation=1):
    kernel = _kernel_tuple(kernel_size)
    dilation = _dilation_tuple(dilation)
    return tuple((int(k) - 1) * int(d) // 2 for k, d in zip(kernel, dilation))


def _softplus_inverse(value):
    value = float(max(value, 1e-8))
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def get_temperature(epoch, warmup_epochs=10, temp_start=5.0, temp_end=1.0):
    if int(epoch) >= int(warmup_epochs):
        return float(temp_end)
    alpha = float(epoch) / float(max(1, int(warmup_epochs)))
    return float(temp_start) * (1.0 - alpha) + float(temp_end) * alpha


def load_balance_loss(router_probs_per_layer):
    loss = None
    for probs in router_probs_per_layer:
        mean_prob = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
        target = torch.full_like(mean_prob, 1.0 / float(mean_prob.numel()))
        value = torch.sum((mean_prob - target) ** 2)
        loss = value if loss is None else loss + value
    return loss


def entropy_loss(router_probs_per_layer):
    loss = None
    for probs in router_probs_per_layer:
        p = probs.reshape(-1, probs.shape[-1])
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
        value = -entropy
        loss = value if loss is None else loss + value
    return loss


class MultiDilatedConv3DExpert(Module):
    def __init__(self, inp_feat, out_feat, kernel_size=(3, 3, 3), dilations=(1, 6, 12)):
        super(MultiDilatedConv3DExpert, self).__init__()
        self.kernel_size = _kernel_tuple(kernel_size)
        self.dilations = tuple(_dilation_tuple(item) for item in dilations)
        self.branches = ModuleList(
            [
                Sequential(
                    Conv3d(
                        int(inp_feat),
                        int(out_feat),
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                        padding=_same_padding(self.kernel_size, dilation=dilation),
                        bias=True,
                    ),
                    BatchNorm3d(int(out_feat)),
                    GELU(),
                )
                for dilation in self.dilations
            ]
        )
        self.fuse = Sequential(
            Conv3d(int(out_feat) * len(self.dilations), int(out_feat), kernel_size=1, bias=True),
            BatchNorm3d(int(out_feat)),
            GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


def _normalise_expert_spec(item):
    if isinstance(item, str):
        key = item.lower()
        if key in {"dilated_3x3x3_d1_6_12", "3x3x3_dilated_1_6_12", "aspp_3x3x3"}:
            return {"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 6, 12)}
        raise ValueError(f"Unsupported expert spec: {item}")

    if isinstance(item, Mapping):
        spec_type = str(item.get("type", item.get("kind", "conv"))).lower()
        kernel = _kernel_tuple(item.get("kernel", item.get("kernel_size", (3, 3, 3))))
        if spec_type in {"multi_dilated", "dilated", "aspp"}:
            dilations = item.get("dilations", item.get("dilation", (1, 6, 12)))
            if isinstance(dilations, int):
                dilations = (dilations,)
            return {"type": "multi_dilated", "kernel": kernel, "dilations": tuple(int(value) for value in dilations)}
        return {"type": "conv", "kernel": kernel, "dilation": _dilation_tuple(item.get("dilation", 1))}

    return {"type": "conv", "kernel": _kernel_tuple(item), "dilation": (1, 1, 1)}


def _expert_label(spec):
    kernel = "x".join(str(item) for item in spec["kernel"])
    if spec["type"] == "multi_dilated":
        dilations = "-".join(str(_dilation_tuple(item)[0]) for item in spec["dilations"])
        return f"{kernel}_dilated_{dilations}"
    dilation = _dilation_tuple(spec.get("dilation", 1))
    if dilation == (1, 1, 1):
        return kernel
    return f"{kernel}_d{dilation[0]}"


def _serialise_expert_specs(specs):
    result = []
    for spec in specs:
        if spec["type"] == "multi_dilated":
            result.append(
                {
                    "type": "multi_dilated",
                    "kernel": list(spec["kernel"]),
                    "dilations": [int(_dilation_tuple(item)[0]) for item in spec["dilations"]],
                }
            )
        else:
            result.append({"type": "conv", "kernel": list(spec["kernel"]), "dilation": list(_dilation_tuple(spec.get("dilation", 1)))})
    return result


class SoftMoE(Module):
    """Soft mixture of multi-kernel 3D convolution experts."""

    DEFAULT_EXPERT_SPECS = (
        {"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 6, 12)},
        {"type": "conv", "kernel": (1, 3, 3), "dilation": (1, 1, 1)},  # in-slice fine boundary
        {"type": "conv", "kernel": (1, 5, 5), "dilation": (1, 1, 1)},  # in-slice wider context
        {"type": "conv", "kernel": (3, 3, 3), "dilation": (1, 1, 1)},  # local 3D structure
        {"type": "conv", "kernel": (5, 5, 5), "dilation": (1, 1, 1)},  # medium 3D context
        {"type": "conv", "kernel": (7, 7, 7), "dilation": (1, 1, 1)},  # large 3D context
    )
    DEFAULT_EXPERT_KERNELS = tuple(item["kernel"] for item in DEFAULT_EXPERT_SPECS)

    def __init__(
        self,
        inp_feat,
        out_feat,
        expert_kernels=None,
        top_k=2,
        router_noise_std=0.0,
        noisy_router=None,
        router_noise_epsilon=1e-2,
        pre_moe_conv=True,
        pre_moe_conv_kernel=(3, 3, 3),
        shared_expert=False,
        shared_expert_kernel=(3, 3, 3),
        shared_expert_weight=1.0,
        routed_fusion="concat_1x1",
        dispatch="soft",
        residual=None,
        router_temperature_warmup_epochs=10,
        router_temperature_start=5.0,
        router_temperature_end=1.0,
    ):
        super(SoftMoE, self).__init__()
        self.inp_feat = int(inp_feat)
        self.out_feat = int(out_feat)
        raw_experts = expert_kernels if expert_kernels is not None else self.DEFAULT_EXPERT_SPECS
        self.expert_specs = tuple(_normalise_expert_spec(item) for item in raw_experts)
        self.expert_kernels = tuple(spec["kernel"] for spec in self.expert_specs)
        self.expert_labels = tuple(_expert_label(spec) for spec in self.expert_specs)
        self.num_experts = len(self.expert_kernels)
        self.requested_top_k = max(1, min(int(top_k), self.num_experts))
        self.top_k = self.num_experts
        self.router_noise_std = float(router_noise_std)
        self.noisy_router = bool(self.router_noise_std > 0.0) if noisy_router is None else bool(noisy_router)
        self.router_noise_epsilon = float(router_noise_epsilon)
        self.use_pre_moe_conv = bool(pre_moe_conv)
        self.pre_moe_conv_kernel = _kernel_tuple(pre_moe_conv_kernel)
        self.use_shared_expert = bool(shared_expert)
        self.shared_expert_kernel = _kernel_tuple(shared_expert_kernel)
        self.shared_expert_weight = float(shared_expert_weight)
        self.routed_fusion = str(routed_fusion or "concat_1x1").lower()
        self.requested_dispatch = str(dispatch or "soft").lower()
        self.dispatch = "soft"
        self.residual = residual
        self.router_temperature_warmup_epochs = int(router_temperature_warmup_epochs)
        self.router_temperature_start = float(router_temperature_start)
        self.router_temperature_end = float(router_temperature_end)
        self.router_epoch = 0
        self.router_temperature = get_temperature(
            self.router_epoch,
            warmup_epochs=self.router_temperature_warmup_epochs,
            temp_start=self.router_temperature_start,
            temp_end=self.router_temperature_end,
        )
        if self.routed_fusion not in {"concat_1x1", "sum"}:
            raise ValueError("SoftMoE routed_fusion must be concat_1x1 or sum.")

        self.experts = ModuleList([self._build_expert(spec) for spec in self.expert_specs])
        self.pre_moe_conv = (
            Sequential(
                Conv3d(
                    self.inp_feat,
                    self.inp_feat,
                    kernel_size=self.pre_moe_conv_kernel,
                    padding=_same_padding(self.pre_moe_conv_kernel),
                    bias=True,
                ),
                BatchNorm3d(self.inp_feat),
                GELU(),
            )
            if self.use_pre_moe_conv
            else None
        )
        self.gate = Conv3d(self.inp_feat, self.num_experts, kernel_size=1, bias=True)
        self.noise_gate = Conv3d(self.inp_feat, self.num_experts, kernel_size=1, bias=True) if self.noisy_router else None
        self._init_noise_gate()
        self.shared_expert = (
            Sequential(
                Conv3d(
                    self.inp_feat,
                    self.out_feat,
                    kernel_size=self.shared_expert_kernel,
                    padding=_same_padding(self.shared_expert_kernel),
                    bias=True,
                ),
                BatchNorm3d(self.out_feat),
                GELU(),
            )
            if self.use_shared_expert
            else None
        )
        self.expert_fuse = (
            Conv3d(self.out_feat * self.num_experts, self.out_feat, kernel_size=1, bias=True)
            if self.routed_fusion == "concat_1x1"
            else None
        )

        if self.residual is not None:
            self.residual_upsampler = Conv3d(self.inp_feat, self.out_feat, kernel_size=1, bias=False)

        self.last_gates = None
        self.last_router_probs = None
        self.last_noise_std = None
        self.last_balance_loss = None
        self.last_load_balance_loss = None
        self.last_entropy_loss = None
        self.register_buffer("gate_importance_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("gate_load_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("gate_count", torch.zeros(1), persistent=False)

    def _init_noise_gate(self):
        if self.noise_gate is None:
            return
        torch.nn.init.zeros_(self.noise_gate.weight)
        initial_std = max(float(self.router_noise_std), float(self.router_noise_epsilon))
        raw_std = _softplus_inverse(initial_std - float(self.router_noise_epsilon))
        torch.nn.init.constant_(self.noise_gate.bias, raw_std)

    def _build_expert(self, spec):
        if spec["type"] == "multi_dilated":
            return MultiDilatedConv3DExpert(
                self.inp_feat,
                self.out_feat,
                kernel_size=spec["kernel"],
                dilations=spec["dilations"],
            )
        kernel_size = spec["kernel"]
        dilation = _dilation_tuple(spec.get("dilation", 1))
        return Sequential(
            Conv3d(
                self.inp_feat,
                self.out_feat,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=_same_padding(kernel_size, dilation=dilation),
                bias=True,
            ),
            BatchNorm3d(self.out_feat),
            GELU(),
        )

    def _noisy_logits(self, x, clean_logits):
        if not (self.training and self.noisy_router and self.noise_gate is not None):
            self.last_noise_std = None
            return clean_logits
        noise_std = F.softplus(self.noise_gate(x)) + self.router_noise_epsilon
        self.last_noise_std = noise_std.detach()
        return clean_logits + torch.randn_like(clean_logits) * noise_std

    def set_router_epoch(self, epoch):
        self.router_epoch = int(epoch)
        self.router_temperature = get_temperature(
            self.router_epoch,
            warmup_epochs=self.router_temperature_warmup_epochs,
            temp_start=self.router_temperature_start,
            temp_end=self.router_temperature_end,
        )

    def _routing(self, x):
        clean_logits = self.gate(x)
        logits = self._noisy_logits(x, clean_logits)
        temperature = max(float(self.router_temperature), 1e-6)
        gates = torch.softmax(logits / temperature, dim=1)
        probs = gates.movedim(1, -1).contiguous()
        return gates, probs

    def _soft_gates(self, x):
        gates, probs = self._routing(x)
        self.last_router_probs = probs
        self.last_load_balance_loss = load_balance_loss([probs])
        self.last_entropy_loss = entropy_loss([probs])
        self.last_balance_loss = self.last_load_balance_loss
        return gates

    def _sparse_gates(self, x):
        return self._soft_gates(x)

    def _balance_loss(self, probs):
        return load_balance_loss([probs])

    def _update_router_stats(self, gates):
        with torch.no_grad():
            gate_values = gates.detach()
            importance = gate_values.mean(dim=(2, 3, 4))
            winners = F.one_hot(gate_values.argmax(dim=1), num_classes=self.num_experts)
            load = winners.movedim(-1, 1).float().mean(dim=(2, 3, 4))
            self.gate_importance_sum.add_(importance.sum(dim=0).to(self.gate_importance_sum.device))
            self.gate_load_sum.add_(load.sum(dim=0).to(self.gate_load_sum.device))
            self.gate_count.add_(gate_values.new_tensor([float(gate_values.shape[0])]).to(self.gate_count.device))

    def reset_router_stats(self):
        self.gate_importance_sum.zero_()
        self.gate_load_sum.zero_()
        self.gate_count.zero_()

    def router_stats(self):
        count = self.gate_count.clamp_min(1.0)
        return {
            "importance": (self.gate_importance_sum / count).detach().cpu(),
            "load": (self.gate_load_sum / count).detach().cpu(),
            "samples": int(self.gate_count.detach().cpu().item()),
            "top_k": int(self.top_k),
            "requested_top_k": int(self.requested_top_k),
            "routing": "soft_voxel",
            "noisy_router": bool(self.noisy_router),
            "router_noise_std": float(self.router_noise_std),
            "router_epoch": int(self.router_epoch),
            "router_temperature": float(self.router_temperature),
            "load_balance_loss": None if self.last_load_balance_loss is None else float(self.last_load_balance_loss.detach().cpu()),
            "entropy_loss": None if self.last_entropy_loss is None else float(self.last_entropy_loss.detach().cpu()),
            "pre_moe_conv": bool(self.use_pre_moe_conv),
            "pre_moe_conv_kernel": tuple(int(item) for item in self.pre_moe_conv_kernel),
            "shared_expert": bool(self.use_shared_expert),
            "shared_expert_kernel": tuple(int(item) for item in self.shared_expert_kernel),
            "routed_fusion": self.routed_fusion,
            "dispatch": self.dispatch,
            "requested_dispatch": self.requested_dispatch,
            "kernels": [tuple(int(item) for item in kernel) for kernel in self.expert_kernels],
            "expert_labels": list(self.expert_labels),
        }

    def _forward_soft(self, x, gates):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_outputs = expert_outputs * gates.unsqueeze(2)
        if self.routed_fusion == "concat_1x1":
            return self.expert_fuse(torch.flatten(weighted_outputs, start_dim=1, end_dim=2))
        return weighted_outputs.sum(dim=1)

    def forward(self, x):
        moe_input = self.pre_moe_conv(x) if self.pre_moe_conv is not None else x
        gates, probs = self._routing(moe_input)
        self.last_gates = gates.detach()
        self.last_router_probs = probs
        self.last_load_balance_loss = self._balance_loss(probs)
        self.last_entropy_loss = entropy_loss([probs])
        self.last_balance_loss = self.last_load_balance_loss
        self._update_router_stats(gates)

        output = self._forward_soft(moe_input, gates)

        if self.shared_expert is not None:
            output = output + self.shared_expert_weight * self.shared_expert(moe_input)
        if self.residual is not None:
            output = output + self.residual_upsampler(moe_input)
        return output


# Compatibility alias for older imports from this soft-MoE module.
SparseMoE = SoftMoE


class UNetMoE(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(
        self,
        in_dim=1,
        out_dim=2,
        feat_channels=(64, 256, 256, 512, 1024),
        residual='conv',
        moe_top_k=2,
        moe_expert_kernels=None,
        moe_router_noise_std=0.0,
        moe_noisy_router=None,
        moe_router_noise_epsilon=1e-2,
        moe_pre_conv=True,
        moe_pre_conv_kernel=(3, 3, 3),
        moe_shared_expert=False,
        moe_shared_expert_kernel=(3, 3, 3),
        moe_shared_expert_weight=1.0,
        moe_routed_fusion="concat_1x1",
        moe_dispatch="soft",
        moe_router_temperature_warmup_epochs=10,
        moe_router_temperature_start=5.0,
        moe_router_temperature_end=1.0,
    ):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNetMoE, self).__init__()
        self.model_name = "adaptive_kernel_softmoe"
        self.backbone_name = "adaptive_kernel_softmoe_encoder"
        self.architecture_config = {
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "feat_channels": list(feat_channels),
            "residual": residual,
            "moe_routing": "soft_voxel",
            "moe_active_experts": "all",
            "moe_top_k": int(moe_top_k),
            "moe_requested_top_k": int(moe_top_k),
            "moe_expert_specs": _serialise_expert_specs(
                tuple(_normalise_expert_spec(item) for item in (moe_expert_kernels or SoftMoE.DEFAULT_EXPERT_SPECS))
            ),
            "moe_router_noise_std": float(moe_router_noise_std),
            "moe_noisy_router": bool(float(moe_router_noise_std) > 0.0) if moe_noisy_router is None else bool(moe_noisy_router),
            "moe_router_noise_epsilon": float(moe_router_noise_epsilon),
            "moe_pre_conv": bool(moe_pre_conv),
            "moe_pre_conv_kernel": list(_kernel_tuple(moe_pre_conv_kernel)),
            "moe_shared_expert": bool(moe_shared_expert),
            "moe_shared_expert_kernel": list(_kernel_tuple(moe_shared_expert_kernel)),
            "moe_shared_expert_weight": float(moe_shared_expert_weight),
            "moe_routed_fusion": str(moe_routed_fusion),
            "moe_dispatch": "soft",
            "moe_requested_dispatch": str(moe_dispatch),
            "moe_router_temperature_warmup_epochs": int(moe_router_temperature_warmup_epochs),
            "moe_router_temperature_start": float(moe_router_temperature_start),
            "moe_router_temperature_end": float(moe_router_temperature_end),
            "encoder_skip_conv": "3x3x3",
            "decoder_fusion": "all_encoder_plus_previous_decoder",
        }

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        moe_kwargs = {
            "expert_kernels": moe_expert_kernels,
            "top_k": moe_top_k,
            "router_noise_std": moe_router_noise_std,
            "noisy_router": moe_noisy_router,
            "router_noise_epsilon": moe_router_noise_epsilon,
            "pre_moe_conv": moe_pre_conv,
            "pre_moe_conv_kernel": moe_pre_conv_kernel,
            "shared_expert": moe_shared_expert,
            "shared_expert_kernel": moe_shared_expert_kernel,
            "shared_expert_weight": moe_shared_expert_weight,
            "routed_fusion": moe_routed_fusion,
            "dispatch": moe_dispatch,
            "residual": residual,
            "router_temperature_warmup_epochs": moe_router_temperature_warmup_epochs,
            "router_temperature_start": moe_router_temperature_start,
            "router_temperature_end": moe_router_temperature_end,
        }
        self.conv_blk1 = SoftMoE(in_dim, feat_channels[0], **moe_kwargs)
        self.conv_skip1 = Sequential(
            Conv3d(in_dim, feat_channels[0], kernel_size=3, padding=1, bias=True),
            BatchNorm3d(feat_channels[0]),
            GELU(),
        )
        self.conv_blk2 = SoftMoE(feat_channels[0], feat_channels[1], **moe_kwargs)
        self.conv_skip2 = Sequential(
            Conv3d(feat_channels[0], feat_channels[1], kernel_size=3, padding=1, bias=True),
            BatchNorm3d(feat_channels[1]),
            GELU(),
        )
        self.conv_blk3 = SoftMoE(feat_channels[1], feat_channels[2], **moe_kwargs)
        self.conv_skip3 = Sequential(
            Conv3d(feat_channels[1], feat_channels[2], kernel_size=3, padding=1, bias=True),
            BatchNorm3d(feat_channels[2]),
            GELU(),
        )
        self.conv_blk4 = SoftMoE(feat_channels[2], feat_channels[3], **moe_kwargs)
        self.conv_skip4 = Sequential(
            Conv3d(feat_channels[2], feat_channels[3], kernel_size=3, padding=1, bias=True),
            BatchNorm3d(feat_channels[3]),
            GELU(),
        )
        self.conv_blk5 = SoftMoE(feat_channels[3], feat_channels[4], **moe_kwargs)
        self.conv_skip5 = Sequential(
            Conv3d(feat_channels[3], feat_channels[4], kernel_size=3, padding=1, bias=True),
            BatchNorm3d(feat_channels[4]),
            GELU(),
        )
        # Decoder full-scale fusion: all encoder features plus all previous decoder features.
        self.decoder_fusion_channels = int(feat_channels[0])
        self.decoder_encoder_projections = torch.nn.ModuleDict()
        self.decoder_history_projections = torch.nn.ModuleDict()
        self.decoder_fusion_blocks = torch.nn.ModuleDict()
        for target_index in range(4):
            key = str(target_index)
            history_indices = self._decoder_history_indices(target_index)
            self.decoder_encoder_projections[key] = ModuleList(
                [
                    Conv3D_Project_Block(source_channels, self.decoder_fusion_channels)
                    for source_channels in feat_channels
                ]
            )
            self.decoder_history_projections[key] = ModuleList(
                [
                    Conv3D_Project_Block(feat_channels[source_index], self.decoder_fusion_channels)
                    for source_index in history_indices
                ]
            )
            self.decoder_fusion_blocks[key] = Conv3D_Block(
                self.decoder_fusion_channels * (len(feat_channels) + len(history_indices)),
                feat_channels[target_index],
                residual=residual,
            )

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = Dropout3d(p=0.5)

    def router_probs_per_layer(self):
        probs = []
        for module in self.modules():
            if isinstance(module, SoftMoE) and module.last_router_probs is not None:
                probs.append(module.last_router_probs)
        return probs

    def moe_load_balance_loss(self):
        probs = self.router_probs_per_layer()
        if not probs:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return load_balance_loss(probs)

    def moe_entropy_loss(self):
        probs = self.router_probs_per_layer()
        if not probs:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return entropy_loss(probs)

    def moe_balance_loss(self):
        return self.moe_load_balance_loss()

    def set_moe_epoch(self, epoch):
        for module in self.modules():
            if isinstance(module, SoftMoE):
                module.set_router_epoch(epoch)

    def reset_moe_router_stats(self):
        for module in self.modules():
            if isinstance(module, SoftMoE):
                module.reset_router_stats()

    def moe_router_stats(self):
        stats = []
        for name, module in self.named_modules():
            if isinstance(module, SoftMoE):
                item = module.router_stats()
                item["layer"] = name
                stats.append(item)
        return stats

    @staticmethod
    def _decoder_history_indices(target_index):
        return list(range(3, int(target_index), -1))

    @staticmethod
    def _resize_to_reference(source, reference):
        return _match_spatial_3d(source, reference)

    def _decode_full_scale(self, encoder_features):
        decoded = {}
        for target_index in (3, 2, 1, 0):
            key = str(target_index)
            reference = encoder_features[target_index]
            projected = []
            for source_index, source in enumerate(encoder_features):
                resized = self._resize_to_reference(source, reference)
                projected.append(self.decoder_encoder_projections[key][source_index](resized))
            for projection, source_index in zip(self.decoder_history_projections[key], self._decoder_history_indices(target_index)):
                resized = self._resize_to_reference(decoded[source_index], reference)
                projected.append(projection(resized))
            decoded[target_index] = self.decoder_fusion_blocks[key](torch.cat(projected, dim=1))
        return decoded

    def forward(self, x, return_features: bool = False):
        # Encoder part

        x1 = self.conv_blk1(x)
        x1_skip = self.conv_skip1(x)

        x_low1 = self.pool1(x1_skip + x1)
        x2 = self.conv_blk2(x_low1)
        x2_skip = self.conv_skip2(x_low1)

        x_low2 = self.pool2(x2_skip + x2)
        x3 = self.conv_blk3(x_low2)
        x3_skip = self.conv_skip3(x_low2)

        x_low3 = self.pool3(x3_skip + x3)
        x4 = self.conv_blk4(x_low3)
        x4_skip = self.conv_skip4(x_low3)

        x_low4 = self.pool4(x4_skip + x4)
        base = self.conv_blk5(x_low4)
        base = self.conv_skip5(x_low4) + base

        decoded = self._decode_full_scale([x1, x2, x3, x4, base])
        d_high4 = decoded[3]
        d_high3 = decoded[2]
        d_high3 = self.dropout(d_high3)
        d_high2 = decoded[1]
        d_high2 = self.dropout(d_high2)
        d_high1 = decoded[0]

        seg = self.one_conv(d_high1)

        if return_features:
            features = {
                "encoder": [x1, x2, x3, x4, base],
                "bottleneck": base,
                "decoder": {
                    "up4": d_high4,
                    "up3": d_high3,
                    "up2": d_high2,
                    "up1": d_high1,
                    "final": d_high1,
                },
            }
            return seg, features

        return seg



class Conv3D_Project_Block(Module):
    def __init__(self, inp_feat, out_feat):
        super(Conv3D_Project_Block, self).__init__()
        self.block = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=1, bias=True),
            BatchNorm3d(out_feat),
            GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            GELU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            GELU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            GELU())

    def forward(self, x):
        return self.deconv(x)
