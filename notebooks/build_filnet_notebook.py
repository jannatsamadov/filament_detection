"""Writes notebooks/filnet_pipeline.ipynb (no outputs) and the Kaggle kernel that executes it.

    python notebooks/build_filnet_notebook.py     # then: kaggle kernels push -p kernels/filnet_notebook_run

The submitted file is this notebook's output: the kernel copies the code the notebook imports into
/kaggle/working/repo with the repository's layout, runs the notebook there with nbconvert, and leaves the
executed copy (outputs and figures) and the CSV in its output. The executed copy is the one committed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KERNEL = REPO / "kernels" / "filnet_notebook_run"
sys.path.insert(0, str(HERE))
from build_notebook import md, code  # noqa: E402


CELLS = [
    md(r"""
# FilNet: solar filament segmentation with a network trained from nothing

Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup, Kaggle). Team: J. Samadov, N. S. Dzhalilov, A. Habilov.

This notebook goes through the submitted pipeline and **builds the submitted file** from the fold models' saved probability maps:

1. **Data.** 707 GONG H$\alpha$ photographs (2048 × 2048), each outlined by one to three experts.
2. **Input.** A filament is darker than the quiet Sun around it, not dark in absolute terms. Each frame is turned, with fixed arithmetic, into seven channels: brightness; contrast against the quiet Sun at the same $\mu$ (limb darkening measured on the frame itself and divided out); darkness against the surroundings at two sizes (≈30 px and ≈130 px); heliographic latitude, longitude from the central meridian, and $\mu$.
3. **Model.** FilNet, a residual U-Net with six resolutions and 17.6 M weights, **trained from random initialisation** (no ImageNet, no other pre-training) on 512 px windows, half of them centred on a filament. One model per fold, trained on Kaggle T4 GPUs by the kernels `filament-filnet`, `filament-filnet-b`, `filament-filnet-c` (`kernels/aug_unet/filnet.py`).
4. **Inference.** Each fold model predicts the whole 2048 × 2048 frame, averaged over the 8 symmetries of the square (D4). The fold models' maps are averaged.
5. **Candidates.** Connected regions of the mean map above a threshold, nearby pieces joined, small ones dropped. Each candidate has 21 features: appearance, its place in a threshold/join grid, heliographic latitude and longitude, orientation.
6. **Selection rule.** Gradient-boosted trees predict each candidate's IoU with the experts from the 21 features. They are trained on the out-of-fold candidates (every photograph's candidates come from the fold model that never saw it), and candidates below a cut are discarded.

The technical report (`report/`) describes the method, the evaluation and what else we tried, including an ensemble of sixteen ImageNet-initialised models that scores the same on the public test set (`notebooks/pipeline.ipynb`).

**Running it.** Nothing here needs a GPU: the networks' probability maps were written by the training kernels.
- **On Kaggle:** attach the competition data and the outputs of the FilNet kernels listed below.
- **Locally:** put the competition data in `MAGFiLO_1.0_Kaggle_2026/` (`kaggle competitions download -c filament-segmentation-2026`, then unzip); the next cells download the kernels' outputs with the Kaggle CLI.
"""),
    code(r"""
import hashlib, inspect, json, os, subprocess, sys, time
import multiprocessing as mp
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "kernels" / "aug_unet" / "filnet.py").exists())
CODE = REPO / "kernels" / "aug_unet"
sys.path.insert(0, str(CODE))

import run                   # data split, geometry, augmentation, PQ (imports torch before cv2)
import post_local as P       # reading the saved probability maps
import candidate_pool as C   # candidate filaments and their features
import kfold as K            # out-of-fold rows, the selection rule, the test CSV
import filnet as FN          # the network, its fixed input channels and its training loop
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn, torch

SETTING = __SETTING__        # (threshold, join radius in px) of the candidates
MIN_AREA = __MIN_AREA__      # px
C.ONE_SETTING, C.POOL_MIN_AREA = SETTING, MIN_AREA
VIEW = "d4"
SUBMITTED_MD5 = "__SUBMITTED_MD5__"   # the file submitted to Kaggle; this notebook's output
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, "
      f"opencv {cv2.__version__}, scikit-learn {sklearn.__version__}, on Kaggle: {ON_KAGGLE}")
"""),
    code(r"""
# the training kernels' outputs: each fold model's probability maps (validation and test, plain and D4)
KERNELS = ["filament-filnet", "filament-filnet-b", "filament-filnet-c"]
if ON_KAGGLE:
    SRC, D, OUT = Path("/kaggle/input"), Path("/tmp/fil_post"), Path("/kaggle/working")
else:
    SRC, D, OUT = REPO / "inputs", REPO / "inputs" / "work", REPO / "notebooks" / "out"
    for k in KERNELS:
        if not (SRC / k).exists():
            subprocess.run(["kaggle", "kernels", "output", f"jannatsamadov/{k}", "-p", str(SRC / k)], check=False)
OUT.mkdir(parents=True, exist_ok=True)
models = P.extract(SRC, D)
FIL = {0: "filnet_phys", 1: "filnet_phys_f1", 2: "filnet_phys_f2", 3: "filnet_phys_f3", 4: "filnet_phys_f4"}
FOLDS = [k for k in FIL if FIL[k] in models]
print("saved maps:", ", ".join(models))
print("FilNet fold models used:", [FIL[k] for k in FOLDS])
"""),
    md(r"""
## 1. Data, and how much the experts disagree

Each training photograph was outlined by one to three experts. The scorer treats every expert's set as a separate ground truth and pools them. Two experts agree with each other at PQ 0.35 only, and they disagree mostly on **whether** a faint structure is a filament, not on where it lies.
"""),
    code(r"""
ann_path, tr_dir, te_dir = run.find_data()
coco = json.load(open(ann_path))
sets_of, by_set = {}, {}
for im in coco["images"]:                       # one entry per photograph and expert
    sets_of.setdefault(im["file_name"], []).append(im["id"])
for a in coco["annotations"]:
    by_set.setdefault(a["image_id"], []).append(a)
n_exp = np.bincount([len(v) for v in sets_of.values()])
print(f"{len(sets_of)} training photographs, {len(coco['annotations'])} outlines;",
      "photographs by number of experts:", {k: int(n) for k, n in enumerate(n_exp) if n})

three = sorted(f for f, s in sets_of.items() if len(s) == 3)
EXAMPLE = max(three, key=lambda f: sum(len(by_set.get(s, [])) for s in sets_of[f]))
img = cv2.imread(str(tr_dir / EXAMPLE), cv2.IMREAD_GRAYSCALE)
over = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
union = np.zeros(img.shape, np.uint8)
for col, sid in zip([(0, 190, 255), (255, 150, 0), (60, 220, 60)], sets_of[EXAMPLE]):
    for a in by_set.get(sid, []):
        m = run.ann_mask(a).astype(np.uint8)
        union |= m
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, col, 3)
n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(union, np.ones((41, 41), np.uint8)))
cx, cy = cen[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))].astype(int)
y0, x0 = int(np.clip(cy - 320, 0, 2048 - 640)), int(np.clip(cx - 320, 0, 2048 - 640))
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(over); ax[0].add_patch(plt.Rectangle((x0, y0), 640, 640, fill=False, ec="w", lw=1.5))
ax[0].set_title(f"{EXAMPLE}: the three experts' outlines (blue, orange, green)")
ax[1].imshow(over[y0:y0 + 640, x0:x0 + 640]); ax[1].set_title("close-up")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 2. The input: what a pixel is compared with

`filnet.front` turns a frame into the network's channels with fixed arithmetic (no weights):
- the **quiet-Sun profile** is the mean brightness in 48 rings of $\mu$, measured twice on the frame itself, the second time without the pixels that differ from the first profile by more than 12% (filaments, spots, plage);
- **contrast** = brightness / profile − 1, so a filament is about −0.2 anywhere on the disk;
- **local darkness** = contrast minus its mean over ≈30 px and over ≈130 px, means taken over disk pixels only;
- **position** = sin(latitude), longitude from the central meridian and $\mu$, from the disk geometry and the solar $B_0$ angle of the date.
"""),
    code(r"""
print(inspect.getsource(FN.front))
STEM = "20130530133654Bh"                         # the test photograph with the most filaments
img = cv2.imread(str(te_dir / f"{STEM}.jpeg"), cv2.IMREAD_GRAYSCALE)
geo = run.geo_channels(torch.tensor([[1.0, 0.0, 0.0, 1.0, run.b0_deg(STEM)]]), torch.device("cpu"))
ch = FN.front(run.to_input(img)[None, None], geo, "phys")[0].numpy()
names = ["brightness", "contrast against the quiet Sun at the same mu", "darker than its ~30 px surroundings",
         "darker than its ~130 px surroundings", "sin(latitude)", "longitude from the central meridian", "mu"]
fig, ax = plt.subplots(1, 4, figsize=(20, 5.3))
for a, k in zip(ax, range(4)):
    a.imshow(ch[k][::2, ::2], cmap="gray", vmin=-2, vmax=2); a.set_title(names[k]); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3. The network and its training

- **Architecture:** a residual U-Net with six resolutions (1, 1/2, … 1/32) and widths 24, 48, 96, 192, 320, 384; full resolution is kept because a filament is about 8 px wide. No pre-trained weights and no attention.
- **Training windows:** 512 × 512 px cut from augmented full frames, every other one centred near a filament pixel, 12 per step.
- **Augmentation:** rotation by any angle about the disk centre, mirrors, contrast, gamma, brightness, blur or sharpening, noise and JPEG re-compression, within the ranges measured between GONG frames. The channels above are computed after the augmentation.
- **Target and loss:** the share of the photograph's experts who drew the pixel; BCE (positive weight 10) plus Tversky ($\alpha$ 0.6 on false positives, $\beta$ 0.4 on misses), at three resolutions (deep supervision).
- **Optimiser:** AdamW (lr $10^{-3}$, weight decay $10^{-2}$), two warm-up epochs then cosine decay; the model is the exponential moving average of the weights.
- **Cost:** one T4 GPU per fold, 3–4 hours (150 000–200 000 windows); 0.3 s per photograph for one pass.
"""),
    code(r"""
net = FN.FilNet(FN.IN_CH["phys"])
print(f"FilNet: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, {FN.IN_CH['phys']} input channels, "
      f"widths {FN.WIDTHS}, blocks per size {FN.DEPTH}")
del net
print("\n" + inspect.getsource(FN.loss_one))
rows = []
for p in sorted(SRC.rglob("filnet_*.json")):
    r = json.load(open(p))
    rows.append({"model": p.stem, "epochs": r.get("epochs"), "windows seen": r.get("steps", 0) * r.get("batch", 0),
                 "PQ, one pass": round(r["plain"]["best"]["PQ"], 4), "PQ, D4": round(r["d4"]["best"]["PQ"], 4),
                 "s per photograph": r.get("sec_per_photo")})
print("Each fold model on its own validation photographs (fixed rules, best of a small grid), as its training kernel wrote it:")
pd.DataFrame(rows)
"""),
    md(r"""
`filnet_raw` is the control arm of fold 0: the same network and training with brightness and position only (4 channels), without the contrast and local-darkness channels.

## 4. Inference: the fold models' maps, averaged
"""),
    code(r"""
test_files = sorted(p.name for p in te_dir.glob("*.jpeg"))
test_stems = [Path(n).stem for n in test_files]
MEAN = "ens:" + "+".join(FIL[k] for k in FOLDS) + f"/{VIEW}"       # post_local.load_map averages the listed models
panels = [("H-alpha", img)] + [(FIL[k], P.load_map(D, f"{FIL[k]}/{VIEW}", "test", STEM)) for k in FOLDS[:2]] \
    + [(f"mean of {len(FOLDS)} fold models", P.load_map(D, MEAN, "test", STEM))]
fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.3))
for a, (title, im) in zip(ax, panels):
    a.imshow(im[::2, ::2], cmap="gray" if title == "H-alpha" else "magma", vmin=0, vmax=255)
    a.set_title(title); a.axis("off")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 5. Candidates and the selection rule, out of fold

For every training photograph the candidates come from the map of the fold model that never saw it. A candidate adds ½ to the denominator of PQ and its IoU to the numerator only if it is matched, so keeping it pays only if its probability of a match times its expected IoU exceeds PQ/2: the rule predicts that product (the candidate's mean matched IoU over the photograph's experts).
"""),
    code(r"""
t0 = time.time()
truth = K.fold_truth(False)                     # every fold's validation photographs and expert sets
pool = mp.Pool(os.cpu_count() or 4)
cache = D / f"filnet_rows_{VIEW}"
cache.mkdir(parents=True, exist_ok=True)
jobs = [(str(D), f"{FIL[n]}/{VIEW}", stem, truth[n]["cache"], f, sets, str(cache), True)
        for n in FOLDS for f, stem, sets in truth[n]["frames"]]
frames = K.one_setting(list(pool.imap(K.pool_job, jobs)))
folds = np.concatenate([[n] * len(truth[n]["frames"]) for n in FOLDS])
COLS = K.GEO_COLS                               # the 21 features
print(f"{len(frames)} photographs out of fold, {sum(len(f['X']) for f in frames)} candidates at t {SETTING[0]}, "
      f"join {SETTING[1]} px, area >= {MIN_AREA} px; {time.time() - t0:.0f} s")

keep_all = K.counts_at_cuts(frames, [np.ones(len(f["X"])) for f in frames])[:, 0]
scores = [None] * len(frames)
for k in FOLDS:                                 # leave one fold out
    m = K.fit(frames, np.flatnonzero(folds != k), COLS, 0)
    for n in np.flatnonzero(folds == k):
        scores[n] = K.score(m, [frames[n]], COLS)[0]
counts = K.counts_at_cuts(frames, scores)
pq = K.pq_of(counts.sum(axis=0))
CUT = float(C.POOL_CUTS[int(np.argmax(pq))])
nested = K.rule_oof(frames, folds, COLS)        # each fold scored with the cut chosen on the other folds
table = [{"": "every candidate kept", **K.summary(keep_all.sum(axis=0))},
         {"": f"selection rule, cut {CUT} (chosen on these photographs)", **K.summary(counts[:, int(np.argmax(pq))].sum(axis=0))},
         {"": "selection rule, nested cut", **K.summary(nested.sum(axis=0))}]
for k in FOLDS:
    table.append({"": f"  fold {k}, nested cut", **K.summary(nested[folds == k].sum(axis=0))})
pd.DataFrame(table)
"""),
    md(r"""
## 6. The submitted file

The rule is fitted on every out-of-fold candidate and applied to the candidates of the mean test map.
"""),
    code(r"""
final = K.fit(frames, np.arange(len(frames)), COLS, 0)
name = f"submission_filnet{len(FOLDS)}_rule21.csv"
info = K.write_csv(D, MEAN, test_files, OUT / name, final, CUT, COLS)
ver = K.verify(OUT / name)
md5 = hashlib.md5((OUT / name).read_bytes()).hexdigest()
print(info, "\nMD5", md5, "\nsame as the submitted file:", md5 == SUBMITTED_MD5)
"""),
    code(r"""
from pycocotools import mask as mu
sub = pd.read_csv(OUT / name)
rows_ = sub[sub.filament_id.str.startswith(STEM + "_")]
over = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for rle in rows_.segmentation_rle:
    m = mu.decode({"size": [2048, 2048], "counts": rle.encode()})
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(over, cs, -1, (255, 150, 0), 3)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(over); ax[0].set_title(f"{STEM}: {len(rows_)} submitted filaments")
ax[1].imshow(over[500:1140, 900:1540]); ax[1].set_title("close-up")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.show()

# a validation photograph: the experts' outlines (green) and FilNet's kept candidates (orange)
n = int(np.flatnonzero(folds == FOLDS[0])[np.argmax([len(frames[i]["X"]) for i in np.flatnonzero(folds == FOLDS[0])])])
f = frames[n]
vimg = cv2.imread(str(tr_dir / f"{f['stem']}.jpeg"), cv2.IMREAD_GRAYSCALE)
over = cv2.cvtColor(vimg, cv2.COLOR_GRAY2RGB)
for sid in sets_of[f"{f['stem']}.jpeg"]:
    for a in by_set.get(sid, []):
        cs, _ = cv2.findContours(run.ann_mask(a).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, (60, 220, 60), 2)
for rle, s in zip(f["rle"], scores[n]):
    if s >= CUT:
        cs, _ = cv2.findContours(mu.decode(rle), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, (255, 150, 0), 2)
plt.figure(figsize=(9, 9)); plt.imshow(over); plt.axis("off")
plt.title(f"validation photograph {f['stem']} (not seen by its fold model): experts green, FilNet orange"); plt.show()
"""),
    md(r"""
## 7. What the numbers mean

- **From nothing, FilNet equals an ImageNet-initialised U-Net twice its size.** With the same rule on the same photographs a ResNet-50 U-Net (32.5 M weights, ImageNet) reaches the same PQ within 0.001, and on the public test set FilNet scores 0.39, the same as our ensemble of sixteen ImageNet-initialised models (`notebooks/pipeline.ipynb`).
- **The physical channels matter.** The control arm without the contrast and local-darkness channels (`filnet_raw`, table in section 3) is lower on the same photographs, with and without the rule.
- **The ceiling is the experts' disagreement.** One expert scored against another reaches PQ 0.35. If the rule is replaced by what the *other* experts of the same photograph actually did with each candidate, PQ does not rise (0.443–0.449 against 0.446 for the rule); only the scored expert's own choices raise it (0.55–0.61). What remains is therefore one annotator's personal choice, which no image-based model can read.

The report lists the approaches that did not reach the submission.
"""),
]


def notebook(setting=(0.6, 3), min_area=100, md5=""):
    cells = json.loads(json.dumps(CELLS).replace("__SETTING__", repr(tuple(setting))).replace("__MIN_AREA__", str(min_area))
                       .replace("__SUBMITTED_MD5__", md5))
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python")))


def kernel(nb, sources):
    """kernels/filnet_notebook_run: the repository files the notebook reads, then nbconvert --execute."""
    files = {f"kernels/aug_unet/{n}": (REPO / "kernels" / "aug_unet" / n).read_text(encoding="utf-8")
             for n in ("run.py", "post_local.py", "candidate_pool.py", "kfold.py", "qwen_seg.py", "filnet.py")}
    files["src/verify_submission.py"] = (REPO / "src" / "verify_submission.py").read_text(encoding="utf-8")
    files["notebooks/filnet_pipeline.ipynb"] = json.dumps(nb, indent=1)
    src = ('"""Kaggle kernel filament-filnet-notebook, generated by notebooks/build_filnet_notebook.py: runs '
           'notebooks/filnet_pipeline.ipynb."""\n'
           "import subprocess, sys\nfrom pathlib import Path\n\n"
           f"FILES = {files!r}\n"
           'R = Path("/kaggle/working/repo")\n'
           "for rel, text in FILES.items():\n"
           "    (R / rel).parent.mkdir(parents=True, exist_ok=True)\n"
           '    (R / rel).write_text(text, encoding="utf-8")\n'
           'Path("/kaggle/working/pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],\n'
           '                                                  capture_output=True, text=True).stdout)\n'
           'r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",\n'
           '                    "--ExecutePreprocessor.timeout=-1",\n'
           '                    "--output-dir", "/kaggle/working", "--output", "filnet_pipeline_executed",\n'
           '                    str(R / "notebooks" / "filnet_pipeline.ipynb")], cwd=R / "notebooks")\n'
           'print("nbconvert exit", r.returncode, flush=True)\n'
           "import shutil\nshutil.rmtree(R, ignore_errors=True)\n"
           "sys.exit(r.returncode)\n")
    KERNEL.mkdir(parents=True, exist_ok=True)
    (KERNEL / "run.py").write_text(src, encoding="utf-8")
    meta = dict(id="jannatsamadov/filament-filnet-notebook", title="filament filnet notebook", code_file="run.py",
                language="python", kernel_type="script", is_private="true", enable_gpu="false", enable_tpu="false",
                enable_internet="true", dataset_sources=[], competition_sources=["filament-segmentation-2026"],
                kernel_sources=[f"jannatsamadov/{k}" for k in sources], model_sources=[])
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.6)
    ap.add_argument("--join", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--md5", default="")
    ap.add_argument("--sources", default="filament-filnet,filament-filnet-b,filament-filnet-c")
    a = ap.parse_args()
    nb = notebook((a.t, a.join), a.min_area, a.md5)
    (HERE / "filnet_pipeline.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    kernel(nb, a.sources.split(","))
    print("wrote notebooks/filnet_pipeline.ipynb and kernels/filnet_notebook_run/")
