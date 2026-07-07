# Method, Logic and Code Thinking: Solar-guided Reverso + SG-SFSU PV Forecasting

This document is the single most important handoff file. It explains, for the
next AI/researcher, the **complete method**, the **reasoning behind every
design choice**, and the **code organization** so they can continue the work
without reading the chat history.

Companion files:

- `docs/HANDOFF.md`: operational handoff (commands, data formats, next steps).
- `docs/INNOVATION.md`: paper-writing summary of innovations and advantages.
- `docs/METHOD_AND_CODE.md` (this file): the full method + logic + code map.

---

## 1. Task definition

**Goal:** short/ultra-short-term photovoltaic (PV) power forecasting using two
modalities:

1. historical PV power time series;
2. sky image sequence from an all-sky / fisheye camera.

**Inputs at time t:**

- `P(t-L:t)`: historical PV power, length L, optionally multivariate.
- `I(t-M:t)`: sky image sequence, M frames.
- timestamps for each image.
- site latitude / longitude (default 37.427, -122.174, Stanford).

**Output:**

- `P_hat(t+1:t+H)`: future PV power over horizon H.

**Key physical intuition:** PV output is most sensitive to clouds *near the
sun disk* and to *cloud edges* crossing the sun. Clouds far from the sun barely
affect current power. Therefore the cloud branch must not treat all image
regions equally.

---

## 2. Overall architecture and why

```
historical PV power P(t-L:t)
      |
Reverso-style temporal encoder  -> Fp
      |
      +-----------------------------------+
                                          |
sky images I(t-M:t) + timestamps + lat/lon
      |
SC-SPGM (self-calibrated solar position)  -> multi-scale solar masks
      |
SG-SFSU per frame:
   - SG-SCSU  (solar-guided spatial channel selection)
   - CFSU     (FFT cloud frequency selection)
   - SG-REM   (solar-guided wavelet enhancement)
   - F_sun    (explicit solar-neighborhood enhancement)
      |
ConvLSTM / Temporal Transformer over frames -> Fc
      |
Fp + Fc -> dynamic gated fusion -> MLP -> P_hat(t+1:t+H)
```

**Why two branches?**
- History branch captures daily cycle, trend, and lag response.
- Cloud branch captures short-term occlusion and ramps.
- Neither alone is enough: history misses occlusion; a single cloud image
  misses the daily cycle.

**Why solar guidance everywhere in the cloud branch?**
- The physically decisive region is around the sun.
- Without guidance, full-image convolutions dilute that signal.
- We inject solar masks into spatial, FFT, and wavelet selection so all three
  feature paths focus on the sun neighborhood.

---

## 3. Module-by-module method and logic

### 3.1 SC-SPGM: Self-Calibrated Solar Position Guidance Module

**Motivation.** We need the sun's pixel location in each sky image to build
masks. A cited paper computes solar zenith/azimuth from time + location, then
fits `r = f(theta_z)` from clear-sky images. But that assumes the image center
equals the sky-circle center and the camera faces exactly north. Real cameras
have yaw bias, lens distortion, and crop offsets.

**Method.**
1. Compute solar angles from timestamp + lat/lon:
   - `theta_z`: zenith angle
   - `gamma_s`: azimuth angle (clockwise from north)
2. From clear-sky images, detect the observed sun center `(x_obs, y_obs)` via
   Otsu thresholding + largest bright blob centroid.
3. Joint self-calibration fit:
   ```
   x = x_c + r(theta_z) * sin(gamma_s + delta_gamma)
   y = y_c + r(theta_z) * cos(gamma_s + delta_gamma)
   r(theta_z) = a * theta_z + b
   ```
   - `x_c, y_c`: real sky-circle center
   - `delta_gamma`: camera yaw bias
   - `a, b`: radial fisheye approximation
4. For any new timestamp, compute angles, project to `(x_s, y_s)`.
5. Build multi-scale Gaussian masks:
   ```
   M_sigma(x,y) = exp(-((x-x_s)^2 + (y-y_s)^2) / (2*sigma^2))
   ```
   with `sigma in {small, medium, large}`.

**Why joint instead of single fit?** Fitting only `r(theta_z)` leaves center
and yaw errors uncorrected; the mask drifts away from the real sun. The joint
fit uses a 1D grid search over `delta_gamma` (the only nonlinear variable) and
closed-form least squares for the rest — deterministic, no SciPy needed.

**Code.** `src/solar_guided_pv/solar_position.py`:
- `solar_angles()` — astronomy.
- `detect_sun_center()` — Otsu sun detection (lazy PIL import).
- `fit_self_calibration()` — joint calibration.
- `sun_pixel()` — project angles to pixel.
- `gaussian_sun_masks()` — multi-scale masks.
- `build_calibration_samples()` — clear-sky sample builder.
- `calibrate_sun.py` — CLI wrapper.

### 3.2 Reverso-style historical power encoder

**Motivation.** The Reverso paper shows small hybrid models (long convolution
+ DeltaNet linear RNN) match large Transformers for time-series forecasting at
a fraction of the size. PV history is a 1-D series with strong daily cycle and
long-range dependency — a good fit.

**Method.**
```
power -> min-max normalize -> linear projection to dim d
       -> alternating blocks of:
            GatedLongConv (causal depthwise long conv + short conv gate)
            DeltaNetLayer (delta-rule linear RNN)
          each followed by MLP channel mixing
       -> take last hidden state -> Fp
```
- Causal left-padding in long conv prevents future leakage.
- DeltaNet implements the delta-rule state update explicitly over time
  (readable, not heavily optimized).
- Min-max normalization matches Reverso's [0,1] normalization.

**Code.** `src/solar_guided_pv/model.py`: `GatedLongConv`, `DeltaNetLayer`,
`ReversoBlock`, `ReversoEncoder`.

**Note:** this is a Reverso-*inspired* trainable encoder, not a pretrained
Reverso foundation model. If official Reverso weights become available,
replacing `ReversoEncoder` is a high-value upgrade.

### 3.3 SG-SCSU: Solar-guided Spatial Channel Selection Unit

**Motivation.** SFS-DETR's SCSU uses three parallel depthwise convolutions
(square `k*k`, horizontal strip `1*(3k+2)`, vertical strip `(3k+2)*1`) and
selects among them with channel weights from global average pooling. For UAV
images that is fine. For sky images, the important region is the sun
neighborhood, so selection should also consider that region.

**Method.**
```
sun_mask = max over scales of masks
global_stat = GAP(X)
sun_stat    = GAP(X * sun_mask)
W = Softmax(Conv1x1([global_stat, sun_stat]))   # shape (B, 3, C, 1, 1)
F_spatial = sum_i W_i * DWConv_i(X)
```
- Two SCSU instances with `k=3` and `k=5` capture multi-scale local features.

**Code.** `SolarGuidedSCSU` in `model.py`.

### 3.4 CFSU: Cloud Frequency Selection Unit

**Motivation.** SFS-DETR's FCSU does FFT, groups frequency components, and
selects positions. For sky images, different frequency bands map to physical
cloud structures: low = overall cloud/brightness, mid = cloud structure, high =
cloud edges / occlusion boundaries. PV ramps correlate with high-frequency
cloud-edge motion near the sun.

**Method.**
```
X_freq = FFT2(X)
X_sun  = X * sun_mask
X_sun_freq = FFT2(X_sun)
concat [real, imag] of both
W_F = Softmax(Conv1x1(concat))           # groups weights
modulate grouped frequency features
restore via IFFT2
```
Solar guidance makes the frequency selection sensitive to sun-neighborhood
spectral changes.

**Code.** `CloudFrequencySelectionUnit` in `model.py`.

### 3.5 SG-REM: Solar-guided Wavelet Enhancement Module (designed, not yet coded)

**Motivation.** MARSS's REM uses 2D discrete wavelet transform to split
features into LL, LH, HL, HH subbands and applies directional convolutions.
This is good for anisotropic data. Cloud images have directional cloud bands
and high-frequency edges. But REM has no task prior and was built for radar
axes. We adapt it for clouds and add solar guidance. It also complements CFSU:
CFSU is global FFT frequency; SG-REM is local wavelet band + direction.

**Method (planned).**
```
F_LL, F_LH, F_HL, F_HH = DWT2(X)
alpha = Softmax(MLP([GAP(X), GAP(X * M_sun)]))
F_rem = sum_i alpha_i * F_i
```
- LL: 1x1 conv, global low-frequency cloud amount.
- HH: 3x3 conv + optional edge operator, cloud-edge high frequency.
- LH/HL: axial convs for cloud-band directions, optionally diagonal convs.
- Solar-guided alpha decides which band matters given sun-neighborhood state.

**Code status:** not yet implemented in `model.py`. Next AI should add a
`SolarGuidedREM` class and integrate it into `SolarGuidedSFSU`.

### 3.6 Explicit solar-neighborhood enhancement

**Motivation.** Even with solar-guided selection, the global encoders still see
the whole image. An explicit ROI feature guarantees the sun-neighborhood signal
is preserved.

**Method.**
```
sun_weighted = X.unsqueeze(1) * masks.unsqueeze(2)   # per-scale
F_sun = Conv(flatten(sun_weighted))
```

**Code.** Inside `SolarGuidedSFSU` in `model.py`.

**Why keep it even with SG-SCSU/CFSU/REM present?** Ablation should decide, but
the default keeps it as a physical-prior safety path.

### 3.7 SG-SFSU: the per-frame fusion unit

**Method.**
```
stem: Conv1x1 + BN + GELU on [image, masks]
spatial  = SG-SCSU(stem, masks)
freq     = CFSU(stem, masks)
sun      = solar-neighborhood enhancement
wavelet  = SG-REM(stem, masks)        # planned
frame_feature = Conv1x1(concat(stem, spatial, freq, sun, wavelet))
```

**Code.** `SolarGuidedSFSU` in `model.py`. Currently fuses 4 paths; SG-REM is
the 5th to add.

### 3.8 Cloud sequence encoder

**Motivation.** A single frame cannot model cloud motion. We need temporal
aggregation over M frames.

**Method.** `ConvLSTMCell` runs over the per-frame features, then global
average pool + LayerNorm -> `Fc`. A Temporal Transformer is an alternative.

**Code.** `ConvLSTMCell`, `CloudSequenceEncoder` in `model.py`.

### 3.9 Dynamic gated fusion + prediction head

**Motivation.** Different weather regimes weight history vs cloud differently:
clear sky -> trust history; cloudy -> trust cloud; fast occlusion -> trust
sun-neighborhood features.

**Method.**
```
Fp_proj = Linear(Fp)
Fc_proj = Linear(Fc)
alpha   = sigmoid(MLP([Fp_proj, Fc_proj]))
F       = alpha * Fc_proj + (1-alpha) * Fp_proj
P_hat   = MLP(LayerNorm(F))
```

**Code.** `DynamicFusionHead`, `SolarGuidedPVForecaster` in `model.py`.

---

## 4. Reasoning chain (why this design, end to end)

1. PV ramps are caused mostly by clouds near the sun -> we need solar position.
2. Astronomical-only sun position is inaccurate due to camera bias ->
   self-calibrate using clear-sky images.
3. A single mask scale is too rigid -> multi-scale Gaussian masks.
4. Cloud images have directional bands and edges -> multi-branch depthwise
   convs (SCSU) + frequency (CFSU) + wavelet (SG-REM).
5. Selection weights should focus on the sun -> inject `GAP(X * M_sun)` into
   every selection path.
6. Global encoders still dilute the sun signal -> explicit `F_sun` ROI branch.
7. Single-frame cloud features miss motion -> ConvLSTM over frames.
8. History and cloud importance varies by weather -> dynamic gated fusion.
9. Long PV history is expensive with Transformers -> Reverso hybrid encoder.
10. The whole thing must run on small sky images (e.g., SKIPP'D 64x64) ->
    depthwise convs, grouped frequency, single-level wavelet, compact dims.

---

## 5. Code map and how to extend

```
src/solar_guided_pv/
  __init__.py          # light import; avoids forcing torch for calibration
  solar_position.py    # SC-SPGM math + sun detection + mask generation
  calibrate_sun.py     # CLI: fit calibration JSON from clear-sky manifest
  dataset.py           # SyntheticPVDataset + ManifestPVDataset
  model.py             # all neural modules + full forecaster
  train.py             # training CLI

experiments/
  solar_guided_pv_config.json   # default (synthetic, 64x64) config

examples/
  clear_sky_manifest_example.csv
  train_manifest_example.csv

tests/
  test_smoke.py        # model forward + self-calibration smoke tests

docs/
  HANDOFF.md           # operational handoff
  INNOVATION.md        # paper summary
  METHOD_AND_CODE.md   # this file
```

**How to add SG-REM (next AI's first code task):**
1. In `model.py`, add a `SolarGuidedREM(nn.Module)` class implementing DWT2
   (use `torch.fft`-based Haar or a fixed convolutional wavelet bank to avoid
   extra deps), 4-band directional convs, and solar-guided softmax weights.
2. In `SolarGuidedSFSU.__init__`, add `self.rem = SolarGuidedREM(...)`.
3. In `SolarGuidedSFSU.forward`, compute `wavelet = self.rem(x, masks)` and
   include it in the final `concat`.
4. Update the `fuse` Conv1x1 input channels from `4*out_channels` to
   `5*out_channels`.
5. Add a smoke test in `tests/test_smoke.py`.

**How to add SAF-style multi-scale fusion later:**
- Add a pyramid feature alignment path before ConvLSTM if multiple resolutions
  of frame features are used.

**How to plug in real Reverso weights:**
- Replace `ReversoEncoder` with the official encoder, keep the same output
  interface `Fp: [B, reverso_dim]`.

---

## 6. Data the code expects

**Clear-sky calibration CSV** (for SC-SPGM):
```csv
image_path,timestamp
clear/0001.jpg,2019-06-01T18:00:00+00:00
```
Requirements: sun visible, accurate timezone-aware timestamps, same camera and
resolution as training, cover morning/noon/afternoon and ideally multiple
seasons.

**Training manifest CSV:**
```csv
power_path,image_paths,timestamps,target_path
samples/power_0001.npy,images/a.jpg;images/b.jpg,2019-06-01T18:00:00+00:00;2019-06-01T18:01:00+00:00,samples/target_0001.npy
```
- `power_path`: `.npy/.npz`, shape `[L]` or `[L, C]`.
- `target_path`: `.npy/.npz`, shape `[H]`.
- `image_paths`: semicolon-separated image sequence.
- `timestamps`: semicolon-separated ISO timestamps matching images.

**For SKIPP'D specifically:**
- image size 64x64
- `mask_sigmas` around [5, 12, 24]
- latitude 37.427, longitude -122.174
- calibration done on 64x64 clear-sky frames (or on originals then resized,
  the dataset code rescales sun coordinates automatically).

---

## 7. Forward-pass tensor shapes

```
power:  [B, L, C_power]
images: [B, T, 3, H, W]
masks:  [B, T, S, H, W]      # S = len(mask_sigmas)

ReversoEncoder(power)  -> Fp: [B, reverso_dim]
CloudSequenceEncoder(images, masks) -> Fc: [B, cloud_dim]
DynamicFusionHead(Fp, Fc) -> P_hat: [B, horizon]
```

---

## 8. Run order for the next AI

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

---

## 9. What is implemented vs. what remains

**Implemented:**
- SC-SPGM (calibration + masks)
- Reverso encoder
- SG-SCSU
- CFSU
- explicit F_sun
- ConvLSTM cloud sequence encoder
- dynamic gated fusion
- training loop + CLIs + synthetic + manifest datasets + smoke tests

**Not implemented:**
- SG-REM (wavelet enhancement) — designed here, needs coding
- real SKIPP'D preprocessing scripts
- official pretrained Reverso weight loading
- weather/cloud-type stratified evaluation
- optical-flow / upwind sun mask
- SAF-style multi-scale pyramid fusion
- uncertainty estimation
- ablation scripts

---

## 10. Ablation plan (do these after baseline trains)

- without solar masks (zero masks)
- without explicit `F_sun`
- without CFSU (frequency branch)
- without SG-REM (once added)
- without Reverso branch (cloud only)
- without cloud branch (history only)
- gated fusion vs concat fusion
- single-scale mask vs multi-scale masks
- self-calibrated sun position vs astronomical-only

---

## 11. Known assumptions and risks

- Timestamps must be correct; timezone errors shift all masks.
- Calibration JSON is camera-specific; do not reuse across cameras.
- `r(theta) = a*theta + b` is a simple projection; can be upgraded to fisheye
  models or polynomial.
- DeltaNet is explicit over time (readable, not optimized).
- Full model tests need PyTorch + Pillow; the base cloud environment used
  during initial implementation did not include them.
- SG-REM is not yet in the code; the next AI should add it before claiming the
  full method is implemented.

---

## 12. One-paragraph method for a paper

> This paper proposes a multimodal photovoltaic power forecasting framework
> that combines a Reverso-inspired historical power encoder with a
> self-calibrated solar-guided cloud image encoder. A self-calibrated solar
> position guidance module jointly fits the sky-circle center, camera yaw
> bias, and radial projection from clear-sky images, and generates
> multi-scale solar-neighborhood masks. These masks are injected into a
> solar-guided spatial channel selection unit, a cloud frequency selection
> unit, and a solar-guided wavelet enhancement module, so that spatial,
> FFT-frequency, and wavelet-band feature selection all focus on the
> sun-neighborhood occlusion and cloud-edge variation. Per-frame features are
> aggregated by a ConvLSTM, and a dynamic gated fusion combines the history
> and cloud features for future PV power prediction. The framework targets
> short-term ramp and cloudy-condition forecasting using sky image sequences.
