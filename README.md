# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk photographs (2048 × 2048).

- Competition: [IEEE BigData Cup, Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026), scored by panoptic quality (PQ).
- Team: J. Samadov, N. S. Dzhalilov, A. Habilov.

**Start with [`notebooks/filnet_pipeline.ipynb`](notebooks/filnet_pipeline.ipynb).** It walks through the submitted pipeline and builds the submitted file from the saved model outputs. The technical report is [`report/report.pdf`](report/report.pdf) (source in [`report/overleaf/`](report/overleaf/)).

## Result

The submitted model is **FilNet**: one family of U-Nets trained from random initialisation (no ImageNet, no other pre-training), on a physically normalised input.

| | weights | initialisation | validation PQ | public test PQ |
|---|---|---|---|---|
| **FilNet, three fold models + selection rule (submitted)** | 17.6 M each | from scratch | **0.451** (out of fold, 423 photographs) | **0.39** |
| U-Net ResNet-50 + the same rule, the same photographs | 32.5 M | ImageNet | 0.452 | 0.38 |
| ensemble of 16 ImageNet-initialised models ([`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb)) | – | ImageNet | 0.458 (fold 0, nested) | 0.39 |

- One expert scored against the other experts reaches PQ 0.35 on the same photographs.
- If the selection rule is replaced by what the *other* experts of the same photograph did with each candidate, PQ does not rise (0.443–0.449 against 0.446); only the scored expert's own choices raise it. What remains is one annotator's personal choice (report, Section 3.3).

## Pipeline

```
GONG Hα 2048²
   │  fixed arithmetic, no weights
   ▼
7 channels: brightness · contrast against the quiet Sun at the same μ · darkness against the
            surroundings at ~30 px and ~130 px · sin(latitude) · longitude from the central meridian · μ
   │
   ▼
FilNet × folds: residual U-Net, 6 resolutions (1 … 1/32), widths 24–384, 17.6 M weights, from scratch
   │  whole frame, averaged over the 8 symmetries of the square (D4); fold models averaged
   ▼
mean map ──▶ candidates: threshold 0.6, pieces within 3 px joined, area ≥ 100 px
   │
   ▼
selection rule: gradient-boosted trees on 21 features (appearance, heliographic latitude and longitude,
                axis orientation) predict each candidate's IoU with the experts; cut 0.24
   │
   ▼
masks → COCO RLE
```

1. **Input channels** ([`filnet.py`](kernels/aug_unet/filnet.py), `front`). A filament is darker than the quiet Sun around it, not dark in absolute terms. The quiet-Sun brightness as a function of μ is measured on each frame (twice, the second time without filaments, spots and plage) and divided out; the local darkness is that contrast minus its mean over the surroundings.
2. **FilNet** ([`filnet.py`](kernels/aug_unet/filnet.py)).
   - Trained on 512 px windows, every other one centred on a filament, 12 per step; 3–4 hours per fold on one T4.
   - Target: the fraction of experts who drew each pixel. Loss: BCE (positive weight 10) plus Tversky, at three resolutions.
   - Augmentation: rotation by any angle about the disk centre, mirrors, and photometric changes within the ranges measured between GONG frames; the channels are computed after it.
   - The model is the moving average of the weights. Inference: the whole frame, D4.
3. **Candidates and selection rule** ([`candidate_pool.py`](kernels/aug_unet/candidate_pool.py), [`kfold.py`](kernels/aug_unet/kfold.py)). The rule is trained on out-of-fold candidates: every photograph's candidates come from the fold model that never saw it.

Every number above is out of fold, with bootstrap intervals over photographs (see the report).

## Repository

| Path | What it is |
|---|---|
| `notebooks/filnet_pipeline.ipynb` | the submitted pipeline, executed, building the submitted CSV |
| `notebooks/pipeline.ipynb` | the 16-model ImageNet ensemble (same public score), executed, rebuilding its CSV |
| `kernels/aug_unet/filnet.py` | FilNet: input channels, network, training loop |
| `kernels/aug_unet/filnet_*.py`, `panel_oracle.py` | the measurements quoted in the report (the rule out of fold, the control without physical channels, outline settings, a keep head on the network's own features, other experts as judges) |
| `kernels/aug_unet/` (rest) | the ensemble's code. Each Kaggle kernel is generated from this folder by `kernels/build.py` |
| `kernels/<name>/kernel-metadata.json` | the Kaggle kernels that trained the models and wrote their probability maps (`filament-filnet`, `-filnet-b`, `-filnet-c` for FilNet) |
| `kernels/aug_unet/pixel_*.py`, `pixnet.py` | the physical per-pixel analysis: a table of position, time, station, contrast against the quiet Sun, neighbours and differential rotation, with random forests and a per-pixel network (report, Section 4) |
| `report/` | the technical report (ACM template) and its figure scripts |
| `src/verify_submission.py` | checks a CSV the way the scorer decodes it, before an upload |
| `src/` (rest) | the earlier spine/half-width approach and the measurements the design rests on |

## Reproducing

```bash
pip install -r requirements.txt
kaggle competitions download -c filament-segmentation-2026   # unzip into MAGFiLO_1.0_Kaggle_2026/
jupyter nbconvert --to notebook --execute notebooks/filnet_pipeline.ipynb
```

- The notebook downloads the fold models' probability maps from the public Kaggle kernels with the Kaggle CLI, so no GPU is needed. It takes about half an hour on 4 CPU cores.
- To retrain a fold, run `python kernels/build.py`, then `kaggle kernels push -p kernels/filnet` (fold 0), `kernels/filnet_b` (folds 1–2) or `kernels/filnet_c` (folds 3–4). Each kernel trains two models side by side on Kaggle's two T4 GPUs.

## Acknowledgment

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, operated by AURA under a cooperative agreement with the National Science Foundation and with contribution from NOAA. Ground truth is MAGFiLO (doi:10.1038/s41597-024-03876-y).
