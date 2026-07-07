# Local handoff entry

This repository already contains the complete handoff material for the
solar-guided Reverso + SG-SFSU photovoltaic forecasting idea.

Start here — full method, logic, and code thinking:

```text
docs/METHOD_AND_CODE.md
```

Operational handoff (commands, data formats, next steps):

```text
docs/HANDOFF.md
```

Paper-writing summary of innovations, solved problems, and advantages:

```text
docs/INNOVATION.md
```

Example manifests:

```text
examples/clear_sky_manifest_example.csv
examples/train_manifest_example.csv
```

Main code:

```text
src/solar_guided_pv/solar_position.py
src/solar_guided_pv/model.py
src/solar_guided_pv/dataset.py
src/solar_guided_pv/calibrate_sun.py
src/solar_guided_pv/train.py
```

Recommended next-agent prompt:

```text
Please first read docs/HANDOFF.md, then continue the research on the
Reverso + self-calibrated solar-guided SG-SFSU photovoltaic forecasting
pipeline. Focus next on preparing real data manifests, fitting solar
calibration parameters, training on real sky-image/PV data, and running
the ablations listed in the handoff document.
```

