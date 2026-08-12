from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
for path in (str(BASELINE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Multi_kenel_MoE.adap_kernel_moe import SparseMoE, UNetMoE, get_temperature


def test_noisy_router() -> None:
    torch.manual_seed(7)
    router = SparseMoE(
        inp_feat=4,
        out_feat=4,
        top_k=2,
        router_noise_std=0.2,
        noisy_router=True,
        pre_moe_conv=True,
        pre_moe_conv_kernel=(3, 3, 3),
        shared_expert=True,
        shared_expert_kernel=(3, 3, 3),
        routed_fusion="concat_1x1",
        dispatch="dense",
    )
    if not router.noisy_router or router.noise_gate is None:
        raise AssertionError("SparseMoE did not create a learned noisy router.")
    if router.pre_moe_conv is None or router.pre_moe_conv_kernel != (3, 3, 3):
        raise AssertionError("SparseMoE did not create the pre-MoE 3x3x3 conv.")
    if router.shared_expert is None or router.shared_expert_kernel != (3, 3, 3):
        raise AssertionError("SparseMoE did not create the shared 3x3x3 expert.")
    if (1, 1, 1) in router.expert_kernels:
        raise AssertionError("The old 1x1x1 expert should be replaced.")
    if router.expert_specs[0]["type"] != "multi_dilated" or router.expert_specs[0]["dilations"] != (1, 6, 12):
        raise AssertionError("First expert should be 3x3x3 multi-dilated with rates 1, 6, 12.")
    if router.expert_fuse is None or tuple(router.expert_fuse.kernel_size) != (1, 1, 1):
        raise AssertionError("SparseMoE did not create the routed expert Conv1x1x1 fuse.")
    if router.expert_fuse.in_channels != router.top_k * router.out_feat:
        raise AssertionError("Routed expert fuse should receive top_k * out_feat channels.")

    x = torch.randn(5, 4, 4, 4, 4)
    router.train()
    torch.manual_seed(1)
    gates_a = router._sparse_gates(x)
    torch.manual_seed(2)
    gates_b = router._sparse_gates(x)
    if torch.allclose(gates_a, gates_b):
        raise AssertionError("Training gates should change when noisy router samples different noise.")

    nonzero_per_sample = (gates_a > 0).sum(dim=1)
    if not torch.equal(nonzero_per_sample, torch.full_like(nonzero_per_sample, 2)):
        raise AssertionError(f"Expected exactly top_k=2 active experts per sample, got {nonzero_per_sample.tolist()}.")
    if not torch.allclose(gates_a.sum(dim=1), torch.ones(gates_a.shape[0]), atol=1e-6):
        raise AssertionError("Sparse gate probabilities should sum to 1 per sample.")

    router.zero_grad(set_to_none=True)
    gates = router._sparse_gates(x)
    weights = torch.arange(router.num_experts, dtype=gates.dtype).view(1, -1)
    loss = (gates * weights).sum()
    loss.backward()
    if router.noise_gate.weight.grad is None or not torch.isfinite(router.noise_gate.weight.grad).all():
        raise AssertionError("Noisy router did not produce finite gradients for noise_gate.")

    router.zero_grad(set_to_none=True)
    output = router(x)
    if tuple(output.shape) != (5, 4, 4, 4, 4):
        raise AssertionError(f"Unexpected SparseMoE output shape with shared expert: {tuple(output.shape)}.")
    output.mean().backward()
    pre_weight = router.pre_moe_conv[0].weight
    if pre_weight.grad is None or not torch.isfinite(pre_weight.grad).all():
        raise AssertionError("Pre-MoE conv did not receive finite gradients.")
    shared_weight = router.shared_expert[0].weight
    if shared_weight.grad is None or not torch.isfinite(shared_weight.grad).all():
        raise AssertionError("Shared expert did not receive finite gradients.")
    if router.expert_fuse.weight.grad is None or not torch.isfinite(router.expert_fuse.weight.grad).all():
        raise AssertionError("Routed expert Conv1x1x1 fuse did not receive finite gradients.")

    router.eval()
    gates_c = router._sparse_gates(x)
    gates_d = router._sparse_gates(x)
    if not torch.allclose(gates_c, gates_d):
        raise AssertionError("Eval gates should be deterministic clean routing.")


def test_router_temperature_and_losses() -> None:
    if abs(get_temperature(0, warmup_epochs=10, temp_start=5.0, temp_end=1.0) - 5.0) > 1e-6:
        raise AssertionError("Router temperature should start at temp_start.")
    if abs(get_temperature(10, warmup_epochs=10, temp_start=5.0, temp_end=1.0) - 1.0) > 1e-6:
        raise AssertionError("Router temperature should end at temp_end after warmup.")

    model = UNetMoE(
        in_dim=1,
        out_dim=2,
        feat_channels=(2, 4, 8, 16, 32),
        moe_top_k=2,
        moe_router_temperature_warmup_epochs=10,
        moe_router_temperature_start=5.0,
        moe_router_temperature_end=1.0,
    )
    model.set_moe_epoch(0)
    temperatures = [module.router_temperature for module in model.modules() if isinstance(module, SparseMoE)]
    if not temperatures or any(abs(value - 5.0) > 1e-6 for value in temperatures):
        raise AssertionError(f"Expected all router temperatures to be 5.0 at epoch 0, got {temperatures}.")
    model.set_moe_epoch(10)
    temperatures = [module.router_temperature for module in model.modules() if isinstance(module, SparseMoE)]
    if any(abs(value - 1.0) > 1e-6 for value in temperatures):
        raise AssertionError(f"Expected all router temperatures to be 1.0 after warmup, got {temperatures}.")

    x = torch.randn(1, 1, 16, 32, 32)
    y = model(x)
    lb_loss = model.moe_load_balance_loss()
    ent_loss = model.moe_entropy_loss()
    if tuple(y.shape) != (1, 2, 16, 32, 32):
        raise AssertionError(f"Unexpected UNetMoE output shape: {tuple(y.shape)}.")
    if not torch.isfinite(lb_loss) or float(lb_loss.detach()) < 0:
        raise AssertionError(f"Load-balance loss should be finite and non-negative, got {float(lb_loss.detach())}.")
    if not torch.isfinite(ent_loss) or float(ent_loss.detach()) >= 0:
        raise AssertionError(f"Entropy loss should be finite and negative, got {float(ent_loss.detach())}.")
    if len(model.router_probs_per_layer()) != 5:
        raise AssertionError("Expected router probabilities from all 5 MoE layers.")


def test_unet_moe_decoder_fusion() -> None:
    torch.manual_seed(11)
    model = UNetMoE(
        in_dim=1,
        out_dim=2,
        feat_channels=(2, 4, 8, 16, 32),
        moe_top_k=2,
        moe_router_noise_std=0.1,
        moe_noisy_router=True,
        moe_pre_conv=True,
        moe_shared_expert=True,
        moe_routed_fusion="concat_1x1",
    ).eval()
    x = torch.randn(2, 1, 16, 32, 32)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (2, 2, 16, 32, 32):
        raise AssertionError(f"Unexpected UNetMoE output shape: {tuple(y.shape)}.")
    if model.architecture_config.get("decoder_fusion") != "all_encoder_plus_previous_decoder":
        raise AssertionError("UNetMoE decoder fusion metadata was not set.")


def main() -> None:
    test_noisy_router()
    test_router_temperature_and_losses()
    test_unet_moe_decoder_fusion()
    print("moe noisy router + temperature + balance/entropy loss + shared expert + decoder fusion regression: ok")


if __name__ == "__main__":
    main()
