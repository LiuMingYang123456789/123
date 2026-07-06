"""Solar position calibration and mask generation utilities.

The module implements the self-calibrated solar position guidance
described in the experiment design:

1. compute astronomical zenith/azimuth angles from timestamp and site;
2. extract observed sun centers from clear-sky images;
3. jointly calibrate sky center, camera yaw bias, and radial projection;
4. generate multi-scale Gaussian masks around the calibrated sun pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, atan2, cos, exp, floor, pi, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SolarAngles:
    """Solar zenith and azimuth in radians."""

    zenith: float
    azimuth: float


@dataclass(frozen=True)
class CalibrationSample:
    """Observed sun center and corresponding astronomical angles."""

    zenith: float
    azimuth: float
    x_obs: float
    y_obs: float


@dataclass(frozen=True)
class CalibrationParams:
    """Self-calibrated image-to-sun projection parameters."""

    x_center: float
    y_center: float
    yaw_bias: float
    radial_a: float
    radial_b: float
    rmse: float

    def radius(self, zenith: float) -> float:
        return self.radial_a * zenith + self.radial_b


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def solar_angles(timestamp: datetime, latitude_deg: float, longitude_deg: float) -> SolarAngles:
    """Compute approximate solar zenith and azimuth.

    This follows the same astronomical quantities used in the cited paper
    while adding longitude-aware local solar time correction. The returned
    azimuth is measured clockwise from north and expressed in radians.
    """

    ts = _as_utc(timestamp)
    day = ts.timetuple().tm_yday
    hour_utc = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    gamma = 2.0 * pi / 365.0 * (day - 1 + (hour_utc - 12.0) / 24.0)

    # Equation-of-time and declination approximations from common solar
    # geometry practice. They are accurate enough for pixel-mask guidance.
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma)
        - 0.032077 * sin(gamma)
        - 0.014615 * cos(2.0 * gamma)
        - 0.040849 * sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * cos(gamma)
        + 0.070257 * sin(gamma)
        - 0.006758 * cos(2.0 * gamma)
        + 0.000907 * sin(2.0 * gamma)
        - 0.002697 * cos(3.0 * gamma)
        + 0.00148 * sin(3.0 * gamma)
    )

    true_solar_minutes = (hour_utc * 60.0 + eq_time + 4.0 * longitude_deg) % 1440.0
    hour_angle = radians(true_solar_minutes / 4.0 - 180.0)
    lat = radians(latitude_deg)

    cos_zenith = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hour_angle)
    cos_zenith = float(np.clip(cos_zenith, -1.0, 1.0))
    zenith = float(np.arccos(cos_zenith))

    # Robust azimuth measured clockwise from north.
    azimuth = atan2(
        sin(hour_angle),
        cos(hour_angle) * sin(lat) - np.tan(decl) * cos(lat),
    ) + pi
    azimuth = azimuth % (2.0 * pi)
    return SolarAngles(zenith=zenith, azimuth=azimuth)


def otsu_threshold(gray: np.ndarray) -> int:
    """Return Otsu threshold for an 8-bit gray image."""

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_between = -1.0
    threshold = 0

    for value in range(256):
        weight_background += hist[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * hist[value]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > max_between:
            max_between = between
            threshold = value
    return threshold


def detect_sun_center(image_path: str | Path, min_bright_pixels: int = 16) -> tuple[float, float]:
    """Detect sun center from a clear-sky image using Otsu segmentation.

    The implementation intentionally avoids OpenCV. It thresholds the image
    brightness, keeps the brightest connected component approximately by
    selecting pixels above the threshold in the upper intensity tail, and
    returns their centroid. For calibration, heavily clouded images should be
    filtered out before calling this function.
    """

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    gray = np.max(arr, axis=2)
    threshold = otsu_threshold(gray)
    mask = gray >= threshold
    if int(mask.sum()) < min_bright_pixels:
        # Fall back to the top 0.1% brightest pixels.
        cutoff = np.percentile(gray, 99.9)
        mask = gray >= cutoff
    ys, xs = np.nonzero(mask)
    if len(xs) < min_bright_pixels:
        raise ValueError(f"Could not detect a reliable sun center in {image_path}")
    return float(xs.mean()), float(ys.mean())


def fit_self_calibration(
    samples: Sequence[CalibrationSample],
    yaw_search_degrees: float = 20.0,
    yaw_steps: int = 801,
) -> CalibrationParams:
    """Fit sky center, yaw bias, and linear radial projection.

    For a fixed yaw bias, the projection equations are linear in
    ``x_center, y_center, radial_a, radial_b``:

    x = x_c + (a * theta + b) * sin(gamma + delta)
    y = y_c + (a * theta + b) * cos(gamma + delta)

    We therefore grid-search the one nonlinear variable, yaw bias, and solve
    each linear subproblem with least squares. This is deterministic and does
    not require SciPy.
    """

    if len(samples) < 5:
        raise ValueError("At least five calibration samples are recommended.")

    yaw_values = np.linspace(
        radians(-yaw_search_degrees),
        radians(yaw_search_degrees),
        yaw_steps,
        dtype=np.float64,
    )
    theta = np.asarray([s.zenith for s in samples], dtype=np.float64)
    gamma = np.asarray([s.azimuth for s in samples], dtype=np.float64)
    observed = np.empty((2 * len(samples),), dtype=np.float64)
    observed[0::2] = [s.x_obs for s in samples]
    observed[1::2] = [s.y_obs for s in samples]

    best: tuple[float, np.ndarray, float] | None = None
    for yaw in yaw_values:
        direction = gamma + yaw
        sin_dir = np.sin(direction)
        cos_dir = np.cos(direction)
        design = np.zeros((2 * len(samples), 4), dtype=np.float64)
        design[0::2, 0] = 1.0
        design[1::2, 1] = 1.0
        design[0::2, 2] = theta * sin_dir
        design[0::2, 3] = sin_dir
        design[1::2, 2] = theta * cos_dir
        design[1::2, 3] = cos_dir

        params, *_ = np.linalg.lstsq(design, observed, rcond=None)
        residual = observed - design @ params
        rmse = float(np.sqrt(np.mean(residual**2)))
        if best is None or rmse < best[2]:
            best = (float(yaw), params, rmse)

    assert best is not None
    yaw, params, rmse = best
    return CalibrationParams(
        x_center=float(params[0]),
        y_center=float(params[1]),
        yaw_bias=yaw,
        radial_a=float(params[2]),
        radial_b=float(params[3]),
        rmse=rmse,
    )


def sun_pixel(angles: SolarAngles, params: CalibrationParams) -> tuple[float, float]:
    """Project solar angles to calibrated image pixel coordinates."""

    direction = angles.azimuth + params.yaw_bias
    radius = params.radius(angles.zenith)
    x = params.x_center + radius * sin(direction)
    y = params.y_center + radius * cos(direction)
    return float(x), float(y)


def gaussian_sun_masks(
    image_size: tuple[int, int],
    sun_xy: tuple[float, float],
    sigmas: Sequence[float],
) -> np.ndarray:
    """Create multi-scale Gaussian masks centered at the sun pixel.

    Returns an array with shape ``(len(sigmas), H, W)`` in ``float32``.
    """

    width, height = image_size
    x_s, y_s = sun_xy
    yy, xx = np.mgrid[0:height, 0:width]
    masks = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("Mask sigma must be positive.")
        mask = np.exp(-((xx - x_s) ** 2 + (yy - y_s) ** 2) / (2.0 * sigma**2))
        masks.append(mask.astype(np.float32))
    return np.stack(masks, axis=0)


def build_calibration_samples(
    image_paths: Iterable[str | Path],
    timestamps: Iterable[datetime],
    latitude_deg: float,
    longitude_deg: float,
) -> list[CalibrationSample]:
    """Create calibration samples from clear-sky images and timestamps."""

    samples: list[CalibrationSample] = []
    for image_path, timestamp in zip(image_paths, timestamps):
        angles = solar_angles(timestamp, latitude_deg, longitude_deg)
        x_obs, y_obs = detect_sun_center(image_path)
        samples.append(
            CalibrationSample(
                zenith=angles.zenith,
                azimuth=angles.azimuth,
                x_obs=x_obs,
                y_obs=y_obs,
            )
        )
    return samples
