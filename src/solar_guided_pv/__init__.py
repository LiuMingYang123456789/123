"""Solar-guided photovoltaic power forecasting package.

The package keeps model imports explicit so calibration utilities can run in
lightweight environments before PyTorch is installed.
"""

from .solar_position import CalibrationParams, solar_angles, sun_pixel

__all__ = ["CalibrationParams", "solar_angles", "sun_pixel"]
