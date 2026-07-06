import torch

from solar_guided_pv.dataset import SyntheticPVDataset
from solar_guided_pv.model import ModelConfig, SolarGuidedPVForecaster
from solar_guided_pv.solar_position import CalibrationSample, fit_self_calibration, sun_pixel


def test_model_forward_smoke():
    dataset = SyntheticPVDataset(samples=2, power_length=32, image_steps=2, horizon=3, image_size=(32, 32))
    batch = [dataset[0], dataset[1]]
    power = torch.stack([item["power"] for item in batch], dim=0)
    images = torch.stack([item["images"] for item in batch], dim=0)
    masks = torch.stack([item["masks"] for item in batch], dim=0)
    model = SolarGuidedPVForecaster(
        ModelConfig(
            horizon=3,
            reverso_dim=16,
            reverso_layers=2,
            reverso_heads=4,
            conv_kernel=7,
            cloud_dim=16,
            fusion_dim=32,
        )
    )
    output = model(power, images, masks)
    assert output.shape == (2, 3)


def test_self_calibration_smoke():
    samples = []
    x_c, y_c, a, b, yaw = 256.0, 255.0, 180.0, 2.0, 0.05
    for i in range(10):
        theta = 0.2 + 0.08 * i
        gamma = 0.3 + 0.2 * i
        radius = a * theta + b
        samples.append(
            CalibrationSample(
                zenith=theta,
                azimuth=gamma,
                x_obs=x_c + radius * torch.sin(torch.tensor(gamma + yaw)).item(),
                y_obs=y_c + radius * torch.cos(torch.tensor(gamma + yaw)).item(),
            )
        )
    params = fit_self_calibration(samples, yaw_search_degrees=10.0, yaw_steps=301)
    x, y = sun_pixel(type("Angles", (), {"zenith": samples[0].zenith, "azimuth": samples[0].azimuth})(), params)
    assert abs(x - samples[0].x_obs) < 1.0
    assert abs(y - samples[0].y_obs) < 1.0
