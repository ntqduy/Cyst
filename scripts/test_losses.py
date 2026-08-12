from __future__ import annotations

import torch

from losses import build_loss


def _assert_valid_loss(loss: torch.Tensor, name: str) -> None:
    if loss.ndim != 0:
        raise AssertionError(f"{name} did not return a scalar loss: shape={tuple(loss.shape)}")
    if not torch.isfinite(loss):
        raise AssertionError(f"{name} returned non-finite loss: {loss.item()}")


def _run_case(name: str, cfg: dict, logits: torch.Tensor, target: torch.Tensor, encoder_features=None) -> None:
    criterion = build_loss({"training": {"loss": cfg}}, num_classes=2)
    loss = criterion(logits, target, encoder_features=encoder_features)
    _assert_valid_loss(loss, name)
    loss.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError(f"{name} backward did not produce finite logits gradients.")
    print(f"{name}: ok loss={float(loss.detach()):.6f}")


def main() -> None:
    torch.manual_seed(7)

    logits_2d = torch.randn(2, 1, 128, 128, requires_grad=True)
    target_2d = (torch.rand(2, 1, 128, 128) > 0.98).float()
    _run_case("dice_bce_2d", {"name": "dice_bce"}, logits_2d, target_2d)

    logits_2d_ft = torch.randn(2, 1, 128, 128, requires_grad=True)
    target_2d_ft = (torch.rand(2, 1, 128, 128) > 0.98).float()
    _run_case("dice_focal_tversky_2d", {"name": "dice_focal_tversky"}, logits_2d_ft, target_2d_ft)

    batch, depth, height, width = 1, 32, 128, 128
    logits_3d = torch.randn(batch, 1, depth, height, width, requires_grad=True)
    target_3d = (torch.rand(batch, 1, depth, height, width) > 0.995).float()
    encoder_features = [
        torch.randn(batch, 16, depth, height, width, requires_grad=True),
        torch.randn(batch, 32, depth // 2, height // 2, width // 2, requires_grad=True),
        torch.randn(batch, 64, depth // 4, height // 4, width // 4, requires_grad=True),
        torch.randn(batch, 128, depth // 8, height // 8, width // 8, requires_grad=True),
    ]
    _run_case(
        "proposed_3d",
        {
            "name": "proposed",
            "alpha": 0.3,
            "beta": 0.7,
            "gamma": 1.33,
            "lambda_boundary": 0.2,
            "lambda_attention": 0.1,
            "attention_weights": "auto",
        },
        logits_3d,
        target_3d,
        encoder_features=encoder_features,
    )


if __name__ == "__main__":
    main()
