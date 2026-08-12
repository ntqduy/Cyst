from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class SliceSelector:
    """Select 2D slices from a 3D volume [B, C, D, H, W]."""

    def __init__(
        self,
        mode: str = "uniform",
        num_slices: int = 5,
        seed: int | None = 42,
        num_groups: int = 5,
        samples_per_group: int = 1,
        similarity_metric: str = "mad",
        **_: Any,
    ) -> None:
        self.mode = str(mode or "uniform").lower().replace("-", "_")
        self.num_slices = max(1, int(num_slices))
        self.seed = None if seed is None else int(seed)
        self.num_groups = max(1, int(num_groups))
        self.samples_per_group = max(1, int(samples_per_group))
        self.similarity_metric = str(similarity_metric or "mad").lower()
        self._generator = torch.Generator(device="cpu")
        if self.seed is not None:
            self._generator.manual_seed(self.seed)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "SliceSelector":
        cfg = dict(cfg or {})
        proposal_cfg = dict(cfg.pop("proposal", {}) or {})
        for key, value in proposal_cfg.items():
            cfg.setdefault(key, value)
        return cls(**cfg)

    def __call__(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if volume.ndim != 5:
            raise ValueError(f"SliceSelector expects [B,C,D,H,W], got {tuple(volume.shape)}")
        batch_size, channels, depth, height, width = volume.shape
        if depth <= 0:
            raise ValueError("Cannot select slices from a volume with D=0.")

        indices = self._select_indices(volume)
        gather_index = indices[:, None, :, None, None].expand(batch_size, channels, indices.shape[1], height, width)
        selected = torch.gather(volume, dim=2, index=gather_index).permute(0, 2, 1, 3, 4).contiguous()
        return selected, indices

    def _select_indices(self, volume: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        if mode == "uniform":
            return self._uniform_indices(volume)
        if mode == "random":
            return self._random_indices(volume)
        if mode in {"middle", "center", "centre", "mid", "single", "single_center", "center_slice", "middle_slice"}:
            return self._middle_indices(volume)
        if mode == "proposal":
            return self._proposal_indices(volume)
        raise ValueError(f"Unsupported slice selection mode: {self.mode}")

    def _middle_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        index = torch.tensor([depth // 2], device=volume.device, dtype=torch.long)
        return index.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _uniform_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        if self.num_slices == 1:
            return self._middle_indices(volume)
        positions = torch.linspace(0, depth - 1, steps=self.num_slices, device=volume.device)
        indices = positions.round().long().clamp_(0, depth - 1)
        return indices.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _random_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        rows = []
        for _ in range(batch_size):
            if self.num_slices <= depth:
                row = torch.randperm(depth, generator=self._generator, device="cpu")[: self.num_slices]
            else:
                row = torch.randint(depth, (self.num_slices,), generator=self._generator, device="cpu")
            rows.append(torch.sort(row).values)
        return torch.stack(rows, dim=0).to(device=volume.device, dtype=torch.long)

    def _proposal_indices(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, _, _ = volume.shape
        rows = []
        with torch.no_grad():
            for batch_index in range(batch_size):
                selected = self._proposal_indices_for_one(volume[batch_index], depth)
                rows.append(selected.to(device=volume.device, dtype=torch.long))
        return torch.stack(rows, dim=0)

    def _proposal_indices_for_one(self, sample: torch.Tensor, depth: int) -> torch.Tensor:
        device = sample.device
        boundaries = torch.linspace(0, depth, steps=min(self.num_groups, depth) + 1, device=device).round().long()
        chosen: list[int] = []

        for group_index in range(len(boundaries) - 1):
            start = int(boundaries[group_index].item())
            end = int(boundaries[group_index + 1].item())
            if end <= start:
                continue
            candidates = torch.arange(start, end, device=device)
            group = sample[:, start:end].permute(1, 0, 2, 3).contiguous()
            scores = self._proposal_scores(group)
            take = min(self.samples_per_group, len(candidates))
            local = torch.topk(scores, k=take, largest=True).indices
            chosen.extend(int(item) for item in candidates[local].detach().cpu().tolist())

        target = self.num_slices
        if len(chosen) < target:
            fallback = torch.linspace(0, depth - 1, steps=target, device=device).round().long().detach().cpu().tolist()
            for item in fallback:
                index = int(item)
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= target:
                    break

        if not chosen:
            chosen = [depth // 2]
        while len(chosen) < target:
            chosen.append(chosen[-1])

        return torch.tensor(sorted(chosen[:target]), device=device, dtype=torch.long).clamp_(0, depth - 1)

    def _proposal_scores(self, group: torch.Tensor) -> torch.Tensor:
        if group.shape[0] == 1:
            return torch.ones(1, device=group.device)

        if self.similarity_metric in {"cos", "cosine", "cosine_similarity"}:
            flattened = group.float().flatten(1)
            normalized = F.normalize(flattened, dim=1, eps=1e-6)
            similarity = normalized @ normalized.t()
            # Lower average similarity means less redundant; negate for topk.
            return -similarity.mean(dim=1)

        if self.similarity_metric not in {"mad", "mean_absolute_deviation"}:
            raise ValueError(f"Unsupported proposal similarity_metric: {self.similarity_metric}")

        center = group.float().mean(dim=0, keepdim=True)
        return (group.float() - center).abs().flatten(1).mean(dim=1)
