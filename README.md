# Solar-guided PV Forecasting Experiment

This repository contains an experimental implementation of a multimodal
photovoltaic power forecasting model:

```text
historical PV sequence -> Reverso-style encoder -> Fp
sky image sequence + self-calibrated sun masks -> SG-SFSU + ConvLSTM -> Fc
Fp + Fc -> dynamic gated fusion -> future PV power
```

The default site coordinates are:

- latitude: `37.427`
- longitude: `-122.174`

## Main components

- `solar_guided_pv.solar_position`
  - astronomical solar zenith/azimuth calculation
  - clear-sky sun-center extraction with Otsu thresholding
  - self-calibrated projection fitting:
    `x_c, y_c, yaw_bias, r(theta)=a*theta+b`
  - multi-scale Gaussian solar-neighborhood mask generation
- `solar_guided_pv.model`
  - Reverso-inspired history power encoder
  - solar-guided SCSU spatial branch
  - cloud frequency selection unit
  - solar-neighborhood enhancement branch
  - ConvLSTM cloud sequence encoder
  - dynamic gated multimodal fusion head
- `solar_guided_pv.dataset`
  - synthetic dataset for debugging
  - manifest-based dataset for real experiments

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Fit self-calibrated solar position parameters

Prepare a clear-sky calibration CSV:

```csv
image_path,timestamp
clear/0001.jpg,2019-06-01T18:00:00+00:00
clear/0002.jpg,2019-06-01T18:05:00+00:00
```

Then run:

```bash
python3 -m solar_guided_pv.calibrate_sun \
  --manifest clear_sky_manifest.csv \
  --root /path/to/data \
  --latitude 37.427 \
  --longitude -122.174 \
  --output solar_calibration.json
```

The output JSON can be copied into the training config under `calibration`.

## Train with synthetic data

```bash
python3 -m solar_guided_pv.train \
  --config experiments/solar_guided_pv_config.json
```

## Train with real data

Create a training manifest:

```csv
power_path,image_paths,timestamps,target_path
samples/power_0001.npy,images/a.jpg;images/b.jpg;images/c.jpg,2019-06-01T18:00:00+00:00;2019-06-01T18:01:00+00:00;2019-06-01T18:02:00+00:00,samples/target_0001.npy
```

Expected arrays:

- `power_path`: shape `[L]` or `[L, C]`
- `target_path`: shape `[H]`
- `image_paths`: a semicolon-separated sequence of sky images
- `timestamps`: ISO timestamps matching the image sequence

Set the config:

```json
{
  "dataset": "manifest",
  "manifest_path": "train_manifest.csv",
  "data_root": "/path/to/data",
  "calibration": {
    "x_center": 256.0,
    "y_center": 256.0,
    "yaw_bias": 0.0,
    "radial_a": 180.0,
    "radial_b": 0.0,
    "rmse": 0.0
  }
}
```

Then run the same training command.

## Quick tests

```bash
python3 -m pytest tests
```