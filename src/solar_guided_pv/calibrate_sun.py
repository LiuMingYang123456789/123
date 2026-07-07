"""CLI for fitting self-calibrated solar position parameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .solar_position import build_calibration_samples, fit_self_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit self-calibrated solar position parameters.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path and timestamp columns.")
    parser.add_argument("--root", default=".", help="Root directory for relative image paths.")
    parser.add_argument("--latitude", type=float, default=37.427)
    parser.add_argument("--longitude", type=float, default=-122.174)
    parser.add_argument("--output", default="solar_calibration.json")
    parser.add_argument("--yaw-search-degrees", type=float, default=20.0)
    args = parser.parse_args()

    root = Path(args.root)
    rows = pd.read_csv(args.manifest)
    image_paths = [
        path if Path(path).is_absolute() else root / path for path in rows["image_path"].astype(str).tolist()
    ]
    timestamps = [datetime.fromisoformat(ts) for ts in rows["timestamp"].astype(str).tolist()]
    samples = build_calibration_samples(image_paths, timestamps, args.latitude, args.longitude)
    params = fit_self_calibration(samples, yaw_search_degrees=args.yaw_search_degrees)
    Path(args.output).write_text(json.dumps(asdict(params), indent=2))
    print(json.dumps(asdict(params), indent=2))


if __name__ == "__main__":
    main()
