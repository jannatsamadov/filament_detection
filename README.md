# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk images
(2048x2048), for the [IEEE BigData Cup competition](https://www.kaggle.com/competitions/filament-segmentation-2026).
Scored with Panoptic Quality.

## Approach: spine-first, no boxes

A filament is a long, thin, curved ribbon — a median **8.2 px wide and 125 px
long**, filling only **24%** of its own bounding box. Measured over 2665 ground
truth boxes, only **33%** could reach IoU 0.7 with any default anchor of a
Mask R-CNN FPN, so the region proposal network never received enough positive
examples and every configuration we tried collapsed to predicting nothing.

This solution drops boxes entirely. A U-Net predicts three maps at the native
2048 resolution, and instances are assembled geometrically:

```
image ──▶ U-Net ──┬──▶ semantic   (filament / background)
                  ├──▶ spine      (Gaussian ridge along the centreline)
                  └──▶ radius     (distance to the mask boundary)

spine ──▶ threshold ──▶ connected components ──▶ smoothing spline
      ──▶ sweep a disk of the predicted radius ──▶ ∩ semantic ──▶ instance
```

Each instance is a swept curve, so it is a single connected region by
construction — the fragmentation that Panoptic Quality punishes hardest cannot
occur, and there is no NMS. Separation comes from the spines themselves: only
**0.9%** of spines come within 10 px of another filament's spine, against 2.8%
of the masks.

Sweeping the *annotated* spine with its true half-width reconstructs the
annotated mask at **median IoU 0.758**, which bounds what the representation can
achieve. Feeding the assembly its own ground-truth maps scores **PQ 0.734**.

## Preprocessing

The images arrive geometrically registered (disk centre 1024,1024 ± 0.5 px,
radius 904 ± 0.8 px) but *not* photometrically: the depth of limb darkening
varies from 0.72 to 0.85 between frames, which is atmospheric rather than solar.
Dividing out each frame's own quiet-sun radial profile (75th percentile per
radial bin, so filaments do not drag it down) **doubles filament contrast**
(0.67σ → 1.33σ) and more than halves the confusion with quiet chromosphere
(27.9% → 11.4%).

Note the limb must be located by the steepest radial brightness drop, not by
thresholding: a scattered-light halo surrounds the disk out to about 1.15 R☉ and
a naive fit reads the radius 13% too large.

## Layout

| Path | Purpose |
|---|---|
| `src/geometry.py` | solar limb detection (fit residual ~2 px) |
| `src/photometry.py` | limb-darkening flattening |
| `src/preprocess.py` | one-off flattening of all frames |
| `src/precompute_targets.py` | one-off rasterisation of the three targets |
| `src/spine_data.py` | patch dataset, split grouped by photograph |
| `src/spine_model.py` | U-Net with ResNet encoder, three heads |
| `src/spine_train.py` | training, full-frame PQ validation |
| `src/spine_infer.py` | instance assembly (spine → spline → sweep) |
| `src/spine_calibrate.py` | threshold sweep on cached predictions |
| `src/spine_predict.py` | test set → RLE → submission CSV |
| `src/verify_submission.py` | eight checks before an upload is spent |
| `src/metrics.py` | Panoptic Quality |
| `src/thermal.py` | GPU duty-cycling for a laptop with a dead fan |
| `src/check_*.py` | the measurements the design decisions rest on |

## Reproducing

```bash
python -m venv .venv && ./.venv/Scripts/python -m pip install -r requirements.txt
# place the Kaggle data in MAGFiLO_1.0_Kaggle_2026/
python src/preprocess.py train test
python src/precompute_targets.py
python src/spine_train.py --epochs 20
python src/spine_calibrate.py
python src/spine_predict.py --out submission.csv
python src/verify_submission.py submission.csv --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
```

## Acknowledgment

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program,
managed by the National Solar Observatory, operated by AURA under a cooperative
agreement with the National Science Foundation and with contribution from NOAA.
Ground truth is MAGFiLO (doi:10.1038/s41597-024-03876-y).
