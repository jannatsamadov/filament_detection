# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk photographs (2048 × 2048).

- Competition: [IEEE BigData Cup, Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026), scored by panoptic quality (PQ).
- Team: J. Samadov, N. S. Dzhalilov, A. Habilov.

**Start with [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb).** It walks through the whole pipeline and rebuilds the submitted file from the saved model outputs. The technical report is in [`report/overleaf/`](report/overleaf/).

## Result

| | fold-0 validation PQ (nested) | public test PQ |
|---|---|---|
| single ResNet-50 U-Net, D4 | 0.434 | 0.36 |
| ten fold U-Nets + selection rule with heliographic features | 0.452 | 0.38 |
| + Qwen3-VL segmenter map + YOLO11 features (**submitted**) | **0.458** | **0.39** |

One expert scored against the other experts reaches PQ 0.35 on the same photographs. At the pixel level the submitted map agrees with the experts (F1 0.71) better than one expert agrees with the rest (0.65).

## Pipeline

```
GONG Hα 2048² ─┬─ 10 U-Nets (ResNet-34/50 × 5 folds, full frame, soft target, D4) ──┐
               ├─ Qwen3-VL-2B vision encoder + mask decoder (pre-trained on GONG) ──┴─ mean map (Qwen 2/12)
               │                                                                      │
               │                                     candidates: t 0.6, join 3 px, area ≥ 100 px
               │                                                                      │
               └─ YOLO11l-seg × 5 folds ── agreement features ──▶ selection rule (gradient boosting, 25 features:
                                                                   appearance, heliographic latitude and longitude,
                                                                   axis orientation, YOLO agreement), cut 0.18
                                                                                      │
                                                                               masks → COCO RLE
```

1. **U-Nets** ([`kernels/aug_unet/run.py`](kernels/aug_unet/run.py)).
   - Input: the whole 2048 frame, so filaments are never cut by tiles.
   - Target: the fraction of experts who drew each pixel.
   - Loss: BCE (positive weight 10) plus soft Dice on the disk.
   - Augmentation: rotation by any angle about the disk centre, mirrors, and photometric changes within the ranges measured between GONG frames.
   - Inference: the 8 symmetries of the square (D4) are averaged.
2. **Qwen3-VL segmenter** ([`qwen_seg.py`](kernels/aug_unet/qwen_seg.py)).
   - The 24-block vision encoder of Qwen3-VL-2B with a convolutional mask decoder.
   - Pre-trained on 2000 unlabelled GONG frames with the U-Nets' maps as targets, then fine-tuned. It is a second family whose errors differ from the U-Nets'.
3. **YOLO11l-seg** ([`yolo_seg.py`](kernels/aug_unet/yolo_seg.py)), one per fold. It is used as an independent opinion on each candidate, not for the masks.
4. **Candidates and selection rule** ([`candidate_pool.py`](kernels/aug_unet/candidate_pool.py), [`kfold.py`](kernels/aug_unet/kfold.py)).
   - Each candidate is described by its appearance, its place in a threshold/join grid, its heliographic latitude and longitude from the central meridian (with the solar B0 angle at the observation time), the orientation of its long axis, and four YOLO agreement features.
   - Gradient-boosted trees predict its IoU with the experts.
   - The rule is trained on the out-of-fold candidates of all 707 photographs; the cut is chosen nested on fold 0's ensemble.

Every number above is out-of-fold or nested, with bootstrap intervals over photographs (see the report).

## Repository

| Path | What it is |
|---|---|
| `notebooks/pipeline.ipynb` | the whole pipeline, executed, rebuilding the submitted CSV |
| `kernels/aug_unet/` | the code. Each Kaggle kernel is generated from it by `kernels/build.py` |
| `kernels/<name>/kernel-metadata.json` | the Kaggle kernels that trained the models and wrote their probability maps |
| `kernels/aug_unet/pixel_*.py`, `pixnet.py` | the physical per-pixel analysis: a table of position, time, station, contrast against the quiet Sun, neighbours and differential rotation, with random forests and a per-pixel network (report, Section 4) |
| `report/` | the technical report (ACM template) and its figure scripts |
| `src/verify_submission.py` | checks a CSV the way the scorer decodes it, before an upload |
| `src/` (rest) | the earlier spine/half-width approach and the measurements the design rests on |

## Reproducing

```bash
pip install -r requirements.txt
kaggle competitions download -c filament-segmentation-2026   # unzip into MAGFiLO_1.0_Kaggle_2026/
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb
```

- The notebook downloads the trained models' outputs from the public Kaggle kernels with the Kaggle CLI, so no GPU is needed. Refitting the selection rule takes about an hour on 4 CPU cores.
- To retrain a model, run `python kernels/build.py`, then `kaggle kernels push -p kernels/<kernel>`. Each U-Net kernel trains two models side by side on Kaggle's two T4 GPUs.

## Acknowledgment

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, operated by AURA under a cooperative agreement with the National Science Foundation and with contribution from NOAA. Ground truth is MAGFiLO (doi:10.1038/s41597-024-03876-y).
