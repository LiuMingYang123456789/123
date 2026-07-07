# Innovation Summary: Solar-guided Reverso + SG-SFSU PV Forecasting

This document is a ready-to-cite summary of the innovations, problems solved,
and advantages of the current forecasting framework. It is intended for paper
writing and for the next AI/researcher who continues the project.

## 1. What the framework is

In one sentence:

> A Reverso-based historical PV power branch combined with a self-calibrated
> solar-guided spatial-frequency-wavelet cloud image branch, fused by a dynamic
> gated mechanism for sky-image-assisted photovoltaic power forecasting.

Overall pipeline:

```text
historical PV power sequence P(t-L:t)
   |
Reverso-style temporal encoder
   |
historical power feature Fp

sky image sequence I(t-M:t) + timestamps + lat/lon
   |
SC-SPGM self-calibrated solar position guidance module
   |
multi-scale solar masks
   |
SG-SFSU (SG-SCSU + CFSU + SG-REM + solar-neighborhood enhancement)
   |
ConvLSTM / Temporal Transformer
   |
cloud image temporal feature Fc

Fp + Fc -> dynamic gated fusion -> future H-step PV power prediction
```

## 2. Innovations

### Innovation 1: Reverso for historical PV power modeling

Introduces Reverso's efficient temporal modeling idea into the PV history
branch using alternating long convolution + DeltaNet linear RNN + MLP blocks.

- small parameter count
- suitable for long-range dependency
- lighter than pure Transformer
- better long-period modeling than plain LSTM

### Innovation 2: Self-Calibrated Solar Position Guidance Module (SC-SPGM)

Improves the cited paper's "astronomical formula + clear-sky image fitting"
approach by jointly calibrating:

```
x = x_c + r(theta_z) * sin(gamma_s + delta_gamma)
y = y_c + r(theta_z) * cos(gamma_s + delta_gamma)
r(theta_z) = a * theta_z + b
```

A single fit recovers:

- true sky-circle center (x_c, y_c)
- camera yaw bias delta_gamma
- radial projection r(theta_z)

and produces multi-scale solar-neighborhood masks:

```
M_small, M_medium, M_large
```

Compared with the original paper's single fit r = f(theta_z), this is more
robust to camera installation bias and lens distortion.

### Innovation 3: Solar-guided spatial channel selection unit (SG-SCSU)

Extends SFS-DETR's SCSU with solar-neighborhood statistics:

```
W = Softmax(Conv([GAP(X), GAP(X * M_sun)]))
```

The selection weights for square, horizontal-strip, and vertical-strip
depthwise convolution branches are driven by both global cloud context and
solar-neighborhood cloud context, focusing on clouds near the sun.

### Innovation 4: Cloud frequency selection unit (CFSU)

Adapts SFS-DETR's FCSU for cloud images:

```
X -> FFT -> grouped frequency selection -> IFFT
```

with solar-neighborhood-guided frequency weights, so that:

- low frequency: overall cloud amount and brightness
- medium frequency: cloud structure
- high frequency: cloud edges and solar occlusion boundaries

are each adaptively enhanced.

### Innovation 5: Explicit solar-neighborhood enhancement branch

Constructs an explicit physically meaningful ROI feature:

```
F_sun = Conv(X * M_sun)
```

This prevents the decisive solar-occlusion signal from being diluted by
full-image convolution over irrelevant cloud regions.

### Innovation 6: Solar-guided wavelet enhancement module (SG-REM)

Adapts MARSS's REM into a solar-guided cloud version with 4-band wavelet
decoupling and solar-guided weights:

```
F_LL, F_LH, F_HL, F_HH = DWT(X)
F_rem = sum_i alpha_i * F_i
alpha  = Softmax(MLP([GAP(X), GAP(X * M_sun)]))
```

This complements CFSU, forming an FFT global-frequency + DWT local-directional
enhancement pair.

### Innovation 7: Dynamic multimodal gated fusion

Instead of simple concatenation, the history and cloud features are fused by:

```
alpha = sigmoid(MLP([Fp, Fc]))
F     = alpha * Fc + (1 - alpha) * Fp
```

so the model can adaptively decide whether to rely on historical trend or
cloud-image information under different weather conditions.

## 3. Problems solved

### Problem 1: History-only methods miss short-term cloud occlusion

Pure PV history models struggle to forecast ramps caused by cloud occlusion.
This framework adds an explicit cloud image branch to model occlusion.

### Problem 2: Treating all sky-image regions equally dilutes key signals

Whole-image averaging dilutes the critical solar-neighborhood occlusion signal.
SC-SPGM generates solar masks and injects them uniformly into SCSU, CFSU, and
REM.

### Problem 3: Camera bias and lens distortion break solar localization

Pure astronomical formulas or simple linear fits cause mask misalignment. The
joint calibration of center, yaw bias, and radial projection improves
localization accuracy.

### Problem 4: SCSU was designed for UAV images, not cloud images

Original SCSU has no solar prior and only performs spatial selection. This work
adds solar guidance, frequency selection, and wavelet enhancement to fit cloud
PV forecasting.

### Problem 5: A single frequency-domain enhancement is insufficient

FFT-based selection alone cannot preserve directional structure. CFSU (FFT) +
SG-REM (DWT) form a complementary pair.

### Problem 6: Fixed contribution between history and cloud features

Different weather regimes weight history and cloud differently. The dynamic
gated fusion allocates weights adaptively.

### Problem 7: Long PV history modeling is expensive

Transformers are heavy. Reverso-style long convolution + DeltaNet is lighter
and efficient for long sequences.

## 4. Advantages over other models

### vs. history-only models (ARIMA / LSTM / Transformer)

| model | cloud image | short-term occlusion response |
|---|---|---|
| ARIMA / LSTM | no | weak |
| Transformer | no | medium |
| ours | yes | strong |

### vs. plain CNN-on-cloud-image models

| model | solar prior | key-region focus |
|---|---|---|
| plain CNN | no | whole-image average |
| ours | yes | solar-neighborhood enhanced |

### vs. original SCSU / SFS-DETR

| model | solar guide | frequency | wavelet | temporal |
|---|---|---|---|---|
| SCSU | no | no | no | no |
| SFS-DETR | no | FCSU | no | no |
| ours | yes | CFSU | SG-REM | ConvLSTM |

### vs. original Reverso

| model | multimodal | cloud image | solar guide |
|---|---|---|---|
| Reverso | univariate | no | no |
| ours | two-branch | yes | yes |

### vs. MARSS REM

| module | task | solar guide | directional conv |
|---|---|---|---|
| REM | radar segmentation | no | radar axial |
| SG-REM | cloud forecasting | yes | cloud-band direction |

### vs. cloud-type-guided paper

| model | solar localization | key region | network input |
|---|---|---|---|
| cloud-type-guided | astronomical + linear fit | fixed window | scalar cloud cover |
| ours | self-calibrated fit | multi-scale masks | end-to-end cloud images |

### Computational efficiency

- lightweight Reverso branch
- depthwise conv + wavelet in cloud branch, controllable parameters
- trainable at 64x64 on a commodity GPU
- fits public datasets such as SKIPP'D for fast validation

## 5. One-paragraph paper-ready summary

> This paper addresses the problems of insufficient short-term cloud-occlusion
> modeling, dilution of key solar-neighborhood information, and camera
> calibration bias in sky-image-assisted photovoltaic power forecasting. It
> proposes a forecasting framework that fuses a Reverso-based historical PV
> power branch with a self-calibrated solar-guided spatial-frequency-wavelet
> cloud image branch. A self-calibrated solar position guidance module
> generates multi-scale solar masks, which are injected uniformly into SCSU,
> CFSU, and SG-REM so that spatial, FFT-frequency, and wavelet-band selection
> all focus on solar-neighborhood occlusion and cloud-edge variation. A dynamic
> gated fusion then combines the two modalities for multimodal forecasting,
> significantly outperforming history-only and plain cloud-image models under
> short-term ramp and cloudy conditions.
