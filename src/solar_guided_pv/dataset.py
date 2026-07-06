"""Datasets for solar-guided PV forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .solar_position import CalibrationParams, gaussian_sun_masks, solar_angles, sun_pixel


@dataclass(frozen=True)
class SiteConfig:
    latitude: float = 37.427
    longitude: float = -122.174
    mask_sigmas: tuple[float, ...] = (12.0, 32.0, 64.0)


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else next(iter(data.keys()))
        return data[key]
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return np.asarray(Image.open(path).convert("RGB"))
    raise ValueError(f"Unsupported data file: {path}")


def _image_to_tensor(image: np.ndarray, image_size: tuple[int, int] | None) -> Tensor:
    if image.ndim != 3:
        raise ValueError("Expected an HWC image array.")
    pil = Image.fromarray(image.astype(np.uint8))
    if image_size is not None:
        pil = pil.resize(image_size, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class ManifestPVDataset(Dataset):
    """Dataset backed by a manifest CSV.

    Required columns:
    - ``power_path``: .npy/.npz file with shape [L] or [L, C]
    - ``image_paths``: semicolon-separated sky image paths
    - ``target_path``: .npy/.npz file with shape [H]
    - ``timestamps``: semicolon-separated ISO timestamps matching images

    Paths may be absolute or relative to ``root``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path = ".",
        site: SiteConfig = SiteConfig(),
        calibration: CalibrationParams | None = None,
        image_size: tuple[int, int] | None = (128, 128),
    ) -> None:
        self.root = Path(root)
        self.rows = pd.read_csv(manifest_path)
        self.site = site
        self.calibration = calibration
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows.iloc[index]
        power = torch.as_tensor(_load_array(self._resolve(row["power_path"])), dtype=torch.float32)
        if power.dim() == 1:
            power = power.unsqueeze(-1)

        image_paths = [self._resolve(p.strip()) for p in str(row["image_paths"]).split(";") if p.strip()]
        timestamps = [datetime.fromisoformat(t.strip()) for t in str(row["timestamps"]).split(";") if t.strip()]
        if len(image_paths) != len(timestamps):
            raise ValueError("image_paths and timestamps must have the same length.")

        image_tensors = []
        mask_tensors = []
        for image_path, timestamp in zip(image_paths, timestamps):
            image = _load_array(image_path)
            image_tensor = _image_to_tensor(image, self.image_size)
            image_tensors.append(image_tensor)

            width, height = self.image_size or (image.shape[1], image.shape[0])
            if self.calibration is None:
                masks = np.zeros((len(self.site.mask_sigmas), height, width), dtype=np.float32)
            else:
                angles = solar_angles(timestamp, self.site.latitude, self.site.longitude)
                x, y = sun_pixel(angles, self.calibration)
                if self.image_size is not None:
                    scale_x = width / image.shape[1]
                    scale_y = height / image.shape[0]
                    x, y = x * scale_x, y * scale_y
                masks = gaussian_sun_masks((width, height), (x, y), self.site.mask_sigmas)
            mask_tensors.append(torch.from_numpy(masks))

        target = torch.as_tensor(_load_array(self._resolve(row["target_path"])), dtype=torch.float32)
        return {
            "power": power,
            "images": torch.stack(image_tensors, dim=0),
            "masks": torch.stack(mask_tensors, dim=0),
            "target": target.flatten(),
        }


class SyntheticPVDataset(Dataset):
    """Small synthetic dataset for smoke tests and pipeline debugging."""

    def __init__(
        self,
        samples: int = 32,
        power_length: int = 96,
        image_steps: int = 4,
        horizon: int = 6,
        image_size: tuple[int, int] = (64, 64),
        mask_sigmas: Sequence[float] = (5.0, 12.0, 24.0),
        seed: int = 7,
    ) -> None:
        self.samples = samples
        self.power_length = power_length
        self.image_steps = image_steps
        self.horizon = horizon
        self.image_size = image_size
        self.mask_sigmas = tuple(mask_sigmas)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        width, height = self.image_size
        phase = 2 * np.pi * ((index % 24) / 24.0)
        t = np.linspace(0.0, 1.0, self.power_length + self.horizon, dtype=np.float32)
        clear_curve = np.maximum(0.0, np.sin(np.pi * t + phase / 8.0))

        # Synthetic moving cloud approaches the sun and attenuates future power.
        x_s = width * (0.45 + 0.1 * np.sin(phase))
        y_s = height * (0.45 + 0.1 * np.cos(phase))
        cloud_offset = np.linspace(-18.0, 8.0, self.image_steps, dtype=np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        images = []
        masks = []
        attenuation = 0.0
        for step, offset in enumerate(cloud_offset):
            cloud = np.exp(-((xx - (x_s + offset)) ** 2 + (yy - y_s) ** 2) / (2 * 10.0**2))
            sky = np.zeros((height, width, 3), dtype=np.float32)
            sky[..., 2] = 0.9
            sky[..., 1] = 0.55 + 0.35 * cloud
            sky[..., 0] = 0.25 + 0.55 * cloud
            images.append(torch.from_numpy(sky).permute(2, 0, 1))
            masks_np = gaussian_sun_masks((width, height), (x_s, y_s), self.mask_sigmas)
            masks.append(torch.from_numpy(masks_np))
            attenuation = max(attenuation, float((cloud * masks_np[1]).mean()))

        noise = self.rng.normal(0.0, 0.02, size=self.power_length + self.horizon).astype(np.float32)
        series = clear_curve * (1.0 - 3.0 * attenuation) + noise
        series = np.clip(series, 0.0, 1.0)
        return {
            "power": torch.from_numpy(series[: self.power_length]).unsqueeze(-1),
            "images": torch.stack(images, dim=0),
            "masks": torch.stack(masks, dim=0),
            "target": torch.from_numpy(series[self.power_length :]),
        }
