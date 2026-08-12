import math
from collections.abc import Mapping

from torch.nn import Module, ModuleList, Sequential
from torch.nn import AdaptiveAvgPool3d, BatchNorm3d, Conv3d, ConvTranspose3d, Dropout3d, Linear, MaxPool3d
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
        key = item.strip().lower().replace("*", "x")
        if key in {"distill", "dilated", "multi_dilated", "dilated_3x3x3_d1_6_12", "3x3x3_dilated_1_6_12", "aspp_3x3x3"}:
            return {"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 6, 12)}
        kernel_parts = key.split("x")
        if len(kernel_parts) == 3 and all(part.isdigit() for part in kernel_parts):
            return {"type": "conv", "kernel": tuple(int(part) for part in kernel_parts), "dilation": (1, 1, 1)}
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


class SparseMoE(Module):
    """Sparse mixture of multi-kernel 3D convolution experts."""

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
        expert=None,
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
        dispatch="sparse",
        residual=None,
    ):
        super(SparseMoE, self).__init__()
        self.inp_feat = int(inp_feat)
        self.out_feat = int(out_feat)
        raw_experts = expert_kernels if expert_kernels is not None else expert if expert is not None else self.DEFAULT_EXPERT_SPECS
        self.expert_specs = tuple(_normalise_expert_spec(item) for item in raw_experts)
        self.expert_kernels = tuple(spec["kernel"] for spec in self.expert_specs)
        self.expert_labels = tuple(_expert_label(spec) for spec in self.expert_specs)
        self.num_experts = len(self.expert_kernels)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.router_noise_std = float(router_noise_std)
        self.noisy_router = bool(self.router_noise_std > 0.0) if noisy_router is None else bool(noisy_router)
        self.router_noise_epsilon = float(router_noise_epsilon)
        self.use_pre_moe_conv = bool(pre_moe_conv)
        self.pre_moe_conv_kernel = _kernel_tuple(pre_moe_conv_kernel)
        self.use_shared_expert = bool(shared_expert)
        self.shared_expert_kernel = _kernel_tuple(shared_expert_kernel)
        self.shared_expert_weight = float(shared_expert_weight)
        self.routed_fusion = str(routed_fusion or "concat_1x1").lower()
        self.dispatch = str(dispatch or "sparse").lower()
        self.residual = residual
        if self.routed_fusion not in {"concat_1x1", "sum"}:
            raise ValueError("SparseMoE routed_fusion must be concat_1x1 or sum.")
        if self.dispatch not in {"sparse", "sparse_stream", "dense"}:
            raise ValueError("SparseMoE dispatch must be sparse, sparse_stream, or dense.")

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
        self.gate_pool = AdaptiveAvgPool3d(1)
        self.gate = Linear(self.inp_feat, self.num_experts)
        self.noise_gate = Linear(self.inp_feat, self.num_experts) if self.noisy_router else None
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
            Conv3d(self.out_feat * self.top_k, self.out_feat, kernel_size=1, bias=True)
            if self.routed_fusion == "concat_1x1"
            else None
        )

        if self.residual is not None:
            self.residual_upsampler = Conv3d(self.inp_feat, self.out_feat, kernel_size=1, bias=False)

        self.last_gates = None
        self.last_noise_std = None
        self.last_balance_loss = None
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

    def _noisy_logits(self, pooled, clean_logits):
        if not (self.training and self.noisy_router and self.noise_gate is not None):
            self.last_noise_std = None
            return clean_logits
        noise_std = F.softplus(self.noise_gate(pooled)) + self.router_noise_epsilon
        self.last_noise_std = noise_std.detach()
        return clean_logits + torch.randn_like(clean_logits) * noise_std

    def _routing(self, x):
        pooled = self.gate_pool(x).flatten(1)
        clean_logits = self.gate(pooled)
        logits = self._noisy_logits(pooled, clean_logits)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=1)
        top_gates = torch.softmax(top_values, dim=1)
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_indices, top_gates)
        return gates, top_indices, top_gates

    def _sparse_gates(self, x):
        gates, _top_indices, _top_gates = self._routing(x)
        return gates

    def _balance_loss(self, gates):
        target_importance = gates.new_full((self.num_experts,), 1.0 / float(self.num_experts))
        target_load = gates.new_full((self.num_experts,), float(self.top_k) / float(self.num_experts))
        importance = gates.mean(dim=0)
        load = (gates > 0).float().mean(dim=0)
        importance_loss = (importance - target_importance).square().mean()
        load_loss = (load - target_load).square().mean()
        return float(self.num_experts) * (importance_loss + load_loss)

    def _update_router_stats(self, gates):
        with torch.no_grad():
            gate_values = gates.detach()
            self.gate_importance_sum.add_(gate_values.sum(dim=0).to(self.gate_importance_sum.device))
            self.gate_load_sum.add_((gate_values > 0).float().sum(dim=0).to(self.gate_load_sum.device))
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
            "noisy_router": bool(self.noisy_router),
            "router_noise_std": float(self.router_noise_std),
            "pre_moe_conv": bool(self.use_pre_moe_conv),
            "pre_moe_conv_kernel": tuple(int(item) for item in self.pre_moe_conv_kernel),
            "shared_expert": bool(self.use_shared_expert),
            "shared_expert_kernel": tuple(int(item) for item in self.shared_expert_kernel),
            "routed_fusion": self.routed_fusion,
            "kernels": [tuple(int(item) for item in kernel) for kernel in self.expert_kernels],
            "expert_labels": list(self.expert_labels),
        }

    def _forward_topk_concat(self, x, top_indices, top_gates):
        rank_outputs = []
        output_shape = (x.shape[0], self.out_feat, *x.shape[-3:])
        for rank in range(self.top_k):
            rank_output = x.new_zeros(output_shape)
            rank_indices = top_indices[:, rank]
            rank_gates = top_gates[:, rank]
            for expert_index, expert in enumerate(self.experts):
                selected = rank_indices == expert_index
                if not bool(selected.any()):
                    continue
                expert_output = expert(x[selected])
                weight = rank_gates[selected].view(-1, 1, 1, 1, 1)
                rank_output[selected] = expert_output * weight
            rank_outputs.append(rank_output)
        return self.expert_fuse(torch.cat(rank_outputs, dim=1))

    def _forward_sparse(self, x, gates):
        output = x.new_zeros((x.shape[0], self.out_feat, *x.shape[-3:]))
        for expert_index, expert in enumerate(self.experts):
            selected = gates[:, expert_index] > 0
            if not bool(selected.any()):
                continue
            expert_output = expert(x[selected])
            weight = gates[selected, expert_index].view(-1, 1, 1, 1, 1)
            output[selected] = output[selected] + expert_output * weight
        return output

    def _forward_sparse_stream(self, x, gates):
        if not x.is_cuda:
            return self._forward_sparse(x, gates)

        active = [index for index in range(self.num_experts) if bool((gates[:, index] > 0).any())]
        streams = [torch.cuda.Stream(device=x.device) for _ in active]
        expert_results = []
        for stream, expert_index in zip(streams, active):
            selected = gates[:, expert_index] > 0
            with torch.cuda.stream(stream):
                expert_output = self.experts[expert_index](x[selected])
                weight = gates[selected, expert_index].view(-1, 1, 1, 1, 1)
                expert_results.append((selected, expert_output * weight, stream))

        current_stream = torch.cuda.current_stream(device=x.device)
        for _, _, stream in expert_results:
            current_stream.wait_stream(stream)

        output = x.new_zeros((x.shape[0], self.out_feat, *x.shape[-3:]))
        for selected, weighted_output, _ in expert_results:
            output[selected] = output[selected] + weighted_output
        return output

    def _forward_dense(self, x, gates):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weights = gates.view(gates.shape[0], gates.shape[1], 1, 1, 1, 1)
        return (expert_outputs * weights).sum(dim=1)

    def forward(self, x):
        moe_input = self.pre_moe_conv(x) if self.pre_moe_conv is not None else x
        gates, top_indices, top_gates = self._routing(moe_input)
        self.last_gates = gates.detach()
        self.last_balance_loss = self._balance_loss(gates)
        self._update_router_stats(gates)

        if self.routed_fusion == "concat_1x1":
            output = self._forward_topk_concat(moe_input, top_indices, top_gates)
        elif self.dispatch == "dense":
            output = self._forward_dense(moe_input, gates)
        elif self.dispatch == "sparse_stream":
            output = self._forward_sparse_stream(moe_input, gates)
        else:
            output = self._forward_sparse(moe_input, gates)

        if self.shared_expert is not None:
            output = output + self.shared_expert_weight * self.shared_expert(moe_input)
        if self.residual is not None:
            output = output + self.residual_upsampler(moe_input)
        return output


class UNetMoE(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity
    DEFAULT_LAYER_EXPERTS = (
        ({"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 2, 3)}, "3x3x3", "5x5x5", "7x7x7"),
        ({"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 3, 5)}, "3x3x3", "5x5x5", "7x7x7"),
        ({"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 3, 6)}, "1x5x5", "3x3x3", "5x5x5"),
        ({"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 2, 4)}, "1x3x3", "1x5x5", "1x7x7"),
        ({"type": "multi_dilated", "kernel": (3, 3, 3), "dilations": (1, 2, 4)}, "1x5x5", "3x3x3", "5x5x5"),
    )

    def __init__(
        self,
        in_dim=1,
        out_dim=2,
        feat_channels=(64, 256, 256, 512, 1024),
        residual='conv',
        moe_top_k=2,
        moe_expert_kernels=None,
        moe_layer_experts=None,
        moe_router_noise_std=0.0,
        moe_noisy_router=None,
        moe_router_noise_epsilon=1e-2,
        moe_pre_conv=True,
        moe_pre_conv_kernel=(3, 3, 3),
        moe_shared_expert=False,
        moe_shared_expert_kernel=(3, 3, 3),
        moe_shared_expert_weight=1.0,
        moe_routed_fusion="concat_1x1",
        moe_dispatch="sparse",
    ):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNetMoE, self).__init__()
        raw_layer_experts = self.DEFAULT_LAYER_EXPERTS if moe_layer_experts is None else moe_layer_experts
        if all(isinstance(item, (str, Mapping)) or not isinstance(item, (list, tuple)) for item in raw_layer_experts):
            layer_experts = tuple(tuple(raw_layer_experts) for _ in range(5))
        else:
            layer_experts = tuple(tuple(item) for item in raw_layer_experts)
        if len(layer_experts) != 5:
            raise ValueError("UNetMoE dynamic setup expects exactly 5 layer expert lists.")
        effective_layer_experts = (
            tuple(tuple(moe_expert_kernels) for _ in range(5))
            if moe_expert_kernels is not None
            else layer_experts
        )
        self.encoder_layer_experts = layer_experts
        self.model_name = "adaptive_kernel_moe_dynamic"
        self.backbone_name = "adaptive_kernel_moe_dynamic_encoder"
        self.architecture_config = {
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "feat_channels": list(feat_channels),
            "residual": residual,
            "moe_top_k": int(moe_top_k),
            "moe_layer_expert_specs": [
                _serialise_expert_specs(tuple(_normalise_expert_spec(item) for item in experts))
                for experts in effective_layer_experts
            ],
            "moe_expert_kernels_override": (
                _serialise_expert_specs(tuple(_normalise_expert_spec(item) for item in moe_expert_kernels))
                if moe_expert_kernels is not None
                else None
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
            "moe_dispatch": str(moe_dispatch),
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
        }
        self.conv_blk1 = SparseMoE(in_dim, feat_channels[0], expert=layer_experts[0], **moe_kwargs)
        self.conv_blk2 = SparseMoE(feat_channels[0], feat_channels[1], expert=layer_experts[1], **moe_kwargs)
        self.conv_blk3 = SparseMoE(feat_channels[1], feat_channels[2], expert=layer_experts[2], **moe_kwargs)
        self.conv_blk4 = SparseMoE(feat_channels[2], feat_channels[3], expert=layer_experts[3], **moe_kwargs)
        self.conv_blk5 = SparseMoE(feat_channels[3], feat_channels[4], expert=layer_experts[4], **moe_kwargs)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = Dropout3d(p=0.5)

    def moe_balance_loss(self):
        losses = []
        for module in self.modules():
            if isinstance(module, SparseMoE) and module.last_balance_loss is not None:
                losses.append(module.last_balance_loss)
        if not losses:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return torch.stack(losses).mean()

    def reset_moe_router_stats(self):
        for module in self.modules():
            if isinstance(module, SparseMoE):
                module.reset_router_stats()

    def moe_router_stats(self):
        stats = []
        for name, module in self.named_modules():
            if isinstance(module, SparseMoE):
                item = module.router_stats()
                item["layer"] = name
                stats.append(item)
        return stats

    def forward(self, x, return_features: bool = False):
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4_up = _match_spatial_3d(self.deconv_blk4(base), x4)
        d4 = torch.cat([d4_up, x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3_up = _match_spatial_3d(self.deconv_blk3(d_high4), x3)
        d3 = torch.cat([d3_up, x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = self.dropout(d_high3)

        d2_up = _match_spatial_3d(self.deconv_blk2(d_high3), x2)
        d2 = torch.cat([d2_up, x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = self.dropout(d_high2)

        d1_up = _match_spatial_3d(self.deconv_blk1(d_high2), x1)
        d1 = torch.cat([d1_up, x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

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
