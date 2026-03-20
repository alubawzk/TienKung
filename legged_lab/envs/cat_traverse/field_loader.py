# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file is part of the CAT IsaacLab task scaffold for Click-and-Traverse.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_field_dir(field_path: str | Path) -> Path:
    raw_path = Path(field_path)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        REPO_ROOT / raw_path,
        REPO_ROOT / "TienKung" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw_path).resolve()


class CatTraverseFieldLoader:
    """Loads CAT field tensors and provides trilinear sampling helpers."""

    def __init__(self, field_path: str | Path, dx: float, origin: tuple[float, float, float], device: str):
        self.field_dir = resolve_field_dir(field_path)
        self.dx = float(dx)
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)

        self.sdf = self._load_tensor("sdf.npy", device, torch.float32)
        self.gf = self._load_tensor("gf.npy", device, torch.float32)
        self.bf = self._load_tensor("bf.npy", device, torch.float32)
        self.obs = self._load_tensor("obs.npy", device, torch.float32)

        self.grid_shape = tuple(int(v) for v in self.sdf.shape)
        self.max_index = torch.tensor([dim - 2 for dim in self.grid_shape], dtype=torch.float32, device=device)
        self._corner_offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.long,
            device=device,
        )

    def _load_tensor(self, name: str, device: str, dtype: torch.dtype) -> torch.Tensor:
        file_path = self.field_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing CAT field asset: {file_path}")
        array = np.load(file_path)
        return torch.as_tensor(array, dtype=dtype, device=device)

    def world_to_grid(self, points_w: torch.Tensor) -> torch.Tensor:
        return (points_w - self.origin) / self.dx

    def sample(self, field: torch.Tensor, points_w: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation over a scalar or vector 3D grid."""

        flat_points = points_w.reshape(-1, 3)
        idx = self.world_to_grid(flat_points)
        idx = torch.clamp_min(idx, 0.0)
        idx = torch.minimum(idx, self.max_index)

        base = torch.floor(idx).long()
        frac = idx - base.float()

        corners = base[:, None, :] + self._corner_offsets[None, :, :]
        corner_values = field[corners[..., 0], corners[..., 1], corners[..., 2]]

        wx = torch.stack([1.0 - frac[:, 0], frac[:, 0]], dim=1)
        wy = torch.stack([1.0 - frac[:, 1], frac[:, 1]], dim=1)
        wz = torch.stack([1.0 - frac[:, 2], frac[:, 2]], dim=1)
        weights = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)

        if corner_values.ndim == 2:
            result = torch.sum(corner_values * weights, dim=1)
        else:
            result = torch.sum(corner_values * weights.unsqueeze(-1), dim=1)
        return result.reshape(points_w.shape[:-1] + result.shape[1:])

    def sample_all(self, points_w: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "sdf": self.sample(self.sdf, points_w),
            "gf": self.sample(self.gf, points_w),
            "bf": self.sample(self.bf, points_w),
            "obs": self.sample(self.obs, points_w),
        }
