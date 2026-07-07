# Handoff: Solar-guided Reverso + SG-SFSU PV Forecasting

This document is written for the next AI/researcher who takes over the
project. It explains the research idea, how the code maps to the idea, what is
implemented, and what should be controlled or improved next.

## 1. Research objective

The target task is short/ultra-short-term photovoltaic power forecasting with
two modalities:

1. historical PV power sequence;
2. sky image sequence from an all-sky/fisheye camera.

The proposed model is:

```text
historical PV power -> Reverso-style temporal encoder -> Fp
sky images + self-calibrated sun masks -> SG-SFSU -> ConvLSTM -> Fc
Fp + Fc -> dynamic gated fusion -> future PV power
```

The key idea is that the sky image branch should not treat all image regions
equally. PV output is especially sensitive to clouds around the sun disk and
near-sun cloud edges. Therefore the cloud branch uses a self-calibrated solar
position guidance module to generate multi-scale sun-neighborhood masks.

## 2. Paper inspirations and what is actually reused

### Reverso paper

Paper: `Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting`.

Original Reverso uses:

- long convolution sequence mixing;
- DeltaNet linear RNN layers;
- MLP channel mixing;
- attention-based decoder head;
- mostly univariate forecasting.

Code mapping:

- `ReversoEncoder` in `src/solar_guided_pv/model.py`
- `GatedLongConv`
- `DeltaNetLayer`
- `ReversoBlock`

Important note:

- This implementation is not a pretrained Reverso foundation model.
- It is a Reverso-inspired trainable encoder for the PV history branch.
- If pretrained Reverso weights/code become available, replacing
  `ReversoEncoder` with the official encoder is a high-value next step.

### SFS-DETR / SCSU paper

Paper: `SFS-DETR: Spatial-Frequency Selection for UAV Object Detection`.

Original method includes:

- SCSU: spatial channel selection with square, horizontal strip, vertical strip
  depthwise convolutions.
- FCSU: frequency component selection through FFT, grouped modulation, and
  position-aware frequency weights.
- SAF/MFE: multi-scale alignment and enhancement for UAV object detection.

Code mapping:

- `SolarGuidedSCSU`: SCSU adapted with sun-neighborhood statistics.
- `CloudFrequencySelectionUnit`: FCSU adapted for cloud image features.
- `SolarGuidedSFSU`: fuses stem feature, spatial branch, frequency branch, and
  explicit solar-neighborhood enhancement.

Important note:

- The sun-guided branch is not from the original SFS-DETR paper.
- It is a task-specific physical prior added for sky-image PV forecasting.

### Solar-position paper

The uploaded solar-position-related paper locates the sun by:

1. computing solar zenith/azimuth from time and geographic location;
2. selecting clear-sky images;
3. detecting the observed sun center with Otsu thresholding;
4. fitting the relationship between zenith angle and image radius;
5. converting polar solar coordinates to image coordinates.

This project improves that idea by fitting:

```text
x = x_center + r(theta_z) * sin(azimuth + yaw_bias)
y = y_center + r(theta_z) * cos(azimuth + yaw_bias)
r(theta_z) = radial_a * theta_z + radial_b
```

This jointly calibrates:

- real sky image center;
- camera yaw/orientation bias;
- radial fisheye projection approximation.

Code mapping:

- `src/solar_guided_pv/solar_position.py`
- `src/solar_guided_pv/calibrate_sun.py`

## 3. Code map

```text
src/solar_guided_pv/
  __init__.py
  solar_position.py   # astronomical angles, Otsu sun detection, calibration, masks
  calibrate_sun.py    # CLI for fitting solar calibration JSON
  dataset.py          # synthetic dataset + manifest real-data dataset
  model.py            # Reverso + SG-SFSU + ConvLSTM + fusion model
  train.py            # training CLI

experiments/
  solar_guided_pv_config.json   # small synthetic debug config

examples/
  clear_sky_manifest_example.csv
  train_manifest_example.csv

tests/
  test_smoke.py
```

## 4. Data expected by the code

### Clear-sky calibration data

Use this to fit solar calibration parameters.

CSV columns:

```csv
image_path,timestamp
clear/0001.jpg,2019-06-01T18:00:00+00:00
clear/0002.jpg,2019-06-01T18:05:00+00:00
```

Requirements:

- images where the sun is visible and not heavily cloud-obscured;
- timestamps must be accurate and timezone-aware;
- images should come from the same camera and resolution as training images;
- cover morning, noon, afternoon, and ideally multiple seasons.

Command:

```bash
python3 -m solar_guided_pv.calibrate_sun \
  --manifest clear_sky_manifest.csv \
  --root /path/to/data \
  --latitude 37.427 \
  --longitude -122.174 \
  --output solar_calibration.json
```

Output fields:

```json
{
  "x_center": 256.0,
  "y_center": 256.0,
  "yaw_bias": 0.0,
  "radial_a": 180.0,
  "radial_b": 0.0,
  "rmse": 0.0
}
```

### Forecasting training data

CSV columns:

```csv
power_path,image_paths,timestamps,target_path
samples/power_0001.npy,images/a.jpg;images/b.jpg,2019-06-01T18:00:00+00:00;2019-06-01T18:01:00+00:00,samples/target_0001.npy
```

Expected arrays:

- `power_path`: `.npy` or `.npz`, shape `[L]` or `[L, C]`
- `target_path`: `.npy` or `.npz`, shape `[H]`
- `image_paths`: semicolon-separated image sequence
- `timestamps`: semicolon-separated ISO timestamps matching image sequence

## 5. Forward pass data flow

### Historical PV branch

Input:

```text
power: [B, L, C_power]
```

Flow:

```text
min-max normalize -> Linear projection -> alternating long conv / DeltaNet blocks
-> last hidden state -> Fp
```

Output:

```text
Fp: [B, reverso_dim]
```

### Sky image branch

Input:

```text
images: [B, T, 3, H, W]
masks:  [B, T, S, H, W]
```

For each frame:

```text
[image, masks] -> stem feature X
X + masks -> SolarGuidedSCSU spatial branch
X + masks -> CloudFrequencySelectionUnit frequency branch
X * masks -> explicit solar-neighborhood enhancement
concat -> 1x1 fusion -> frame feature
```

Then:

```text
frame features over T -> ConvLSTM -> global pool -> Fc
```

Output:

```text
Fc: [B, cloud_dim]
```

### Fusion

```text
Fp -> projection
Fc -> projection
alpha = sigmoid(MLP([Fp, Fc]))
F = alpha * Fc + (1 - alpha) * Fp
F -> MLP -> future PV horizon
```

Output:

```text
prediction: [B, horizon]
```

## 6. Why the explicit sun-neighborhood branch exists

`F_sun = Conv(X * M_sun)` is included because full-image cloud features can
dilute the physically decisive region. In PV forecasting:

- clouds far away from the sun may not affect current power;
- small clouds or cloud edges near the sun may trigger large ramps;
- a global cloud encoder may learn overall cloudiness but miss local solar
  occlusion details.

The explicit branch provides a physically meaningful ROI feature. It should be
kept unless ablation proves it is harmful.

## 7. What is implemented vs. what remains research work

Implemented:

- complete trainable PyTorch model scaffold;
- calibration utilities;
- synthetic debugging dataset;
- manifest dataset for real data;
- training loop;
- smoke tests.

Not yet implemented:

- real SKIPP'D preprocessing scripts;
- official pretrained Reverso weight loading;
- real evaluation metrics split by weather/cloud type;
- optical-flow/upwind sun mask;
- SAF-style multi-scale pyramid fusion;
- uncertainty estimation;
- ablation scripts.

## 8. Recommended next experiments

Run in this order:

1. Install dependencies and run smoke tests.
2. Run synthetic training to make sure the loop works.
3. Build clear-sky calibration manifest and fit `solar_calibration.json`.
4. Build real train/validation/test manifests.
5. Train the full model on real data.
6. Run ablations:
   - without masks;
   - without explicit `F_sun`;
   - without frequency branch;
   - without Reverso branch;
   - concat fusion instead of gated fusion.
7. Add weather/cloud-type stratified evaluation:
   - clear sky;
   - partly cloudy;
   - overcast;
   - ramp events.

## 9. Known assumptions and risks

- Timestamps must be correct. Timezone mistakes will shift sun masks.
- The calibration JSON is camera-specific. Do not reuse across cameras unless
  the optics, crop, and orientation are identical.
- `radial_a * theta + radial_b` is a simple projection approximation. A future
  version can replace it with fisheye models or polynomial projection.
- The current DeltaNet implementation is explicit over sequence length. It is
  readable but not highly optimized.
- Full model tests require PyTorch and Pillow. The base cloud environment used
  during implementation did not include them.

## 10. Minimal command sequence for the next AI

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest tests

python3 -m solar_guided_pv.train \
  --config experiments/solar_guided_pv_config.json
```

After real data is prepared:

```bash
python3 -m solar_guided_pv.calibrate_sun \
  --manifest clear_sky_manifest.csv \
  --root /path/to/data \
  --latitude 37.427 \
  --longitude -122.174 \
  --output solar_calibration.json

python3 -m solar_guided_pv.train \
  --config experiments/real_data_config.json
```

