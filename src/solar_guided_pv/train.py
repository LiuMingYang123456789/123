"""Training entry point for the complete PV forecasting experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import ManifestPVDataset, SiteConfig, SyntheticPVDataset
from .model import ModelConfig, SolarGuidedPVForecaster
from .solar_position import CalibrationParams


@dataclass
class TrainConfig:
    dataset: str = "synthetic"
    manifest_path: str | None = None
    data_root: str = "."
    output_dir: str = "runs/solar_guided_pv"
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.2
    num_workers: int = 0
    seed: int = 42
    latitude: float = 37.427
    longitude: float = -122.174
    image_size: tuple[int, int] = (64, 64)
    mask_sigmas: tuple[float, ...] = (5.0, 12.0, 24.0)
    calibration: dict[str, float] | None = None
    model: ModelConfig = field(default_factory=ModelConfig)


def _load_config(path: str | None) -> TrainConfig:
    config = TrainConfig()
    if path is None:
        return config
    data = json.loads(Path(path).read_text())
    model_data = data.pop("model", {})
    for key, value in data.items():
        setattr(config, key, value)
    config.model = ModelConfig(**{**asdict(config.model), **model_data})
    return config


def _make_dataset(config: TrainConfig):
    if config.dataset == "synthetic":
        return SyntheticPVDataset(
            samples=40,
            image_size=tuple(config.image_size),
            mask_sigmas=tuple(config.mask_sigmas),
            horizon=config.model.horizon,
        )
    if config.dataset == "manifest":
        if config.manifest_path is None:
            raise ValueError("manifest_path is required when dataset='manifest'")
        calibration = CalibrationParams(**config.calibration) if config.calibration else None
        return ManifestPVDataset(
            manifest_path=config.manifest_path,
            root=config.data_root,
            site=SiteConfig(
                latitude=config.latitude,
                longitude=config.longitude,
                mask_sigmas=tuple(config.mask_sigmas),
            ),
            calibration=calibration,
            image_size=tuple(config.image_size),
        )
    raise ValueError(f"Unknown dataset type: {config.dataset}")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            pred = model(batch["power"], batch["images"], batch["masks"])
            loss = mae(pred, batch["target"])
            rmse = torch.sqrt(mse(pred, batch["target"]) + 1e-8)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        batch_size = batch["target"].shape[0]
        count += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_mae += float(mae(pred.detach(), batch["target"])) * batch_size
        total_rmse += float(rmse.detach()) * batch_size
    return {
        "loss": total_loss / max(count, 1),
        "mae": total_mae / max(count, 1),
        "rmse": total_rmse / max(count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reverso + SC-SG-SFSU PV forecaster.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = _load_config(args.config)
    torch.manual_seed(config.seed)
    device = torch.device(args.device)
    dataset = _make_dataset(config)

    val_len = max(1, int(len(dataset) * config.val_fraction))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SolarGuidedPVForecaster(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))

    best_mae = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, None, device)
        print(
            f"epoch={epoch:03d} "
            f"train_mae={train_metrics['mae']:.5f} train_rmse={train_metrics['rmse']:.5f} "
            f"val_mae={val_metrics['mae']:.5f} val_rmse={val_metrics['rmse']:.5f}"
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(config.model),
                    "train_config": asdict(config),
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
