"""Which images stay hard after flattening, and why?

Flattening halved the average confusion between filament and quiet-sun pixels,
but the image-to-image spread did not shrink. This asks what the surviving hard
cases have in common: the halo outside the disk, the depth of limb darkening,
the granular contrast of the chromosphere itself, or simply how the filaments in
that image happen to be drawn.
"""
from __future__ import annotations

import sys, json, collections, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026"
IMGS = BASE / "train" / "train_images"
FIGS = ROOT / "figures"
sys.path.insert(0, str(HERE))
from geometry import fit_disk, radial_coordinates   # noqa: E402
from photometry import flatten                      # noqa: E402


def masks_for(anns):
    m = Image.new("L", (2048, 2048), 0)
    dr = ImageDraw.Draw(m)
    for a in anns:
        p = a["segmentation"][0]
        dr.polygon(list(zip(p[0::2], p[1::2])), fill=1)
    return np.array(m, bool)


def main(n=60):
    d = json.load(open(BASE / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    keys = sorted(by_img)
    random.seed(0); random.shuffle(keys)

    rec = []
    for k in keys:
        if len(rec) >= n:
            break
        f = IMGS / fname[k]
        if not f.exists():
            continue
        img = np.array(Image.open(f).convert("L")).astype(np.float32)
        fit = fit_disk(img, n_rays=180)
        rr, _ = radial_coordinates(img.shape, fit)
        fl = flatten(img, rr)
        M = masks_for(by_img[k])
        disk = rr <= 0.98
        fil, qui = M & disk, (~M) & disk
        if fil.sum() < 500:
            continue

        confusion = float(np.mean(fl[qui] < np.median(fl[fil])))

        # candidate explanations
        halo = float(np.median(img[(rr > 1.02) & (rr < 1.12)]))          # the band outside
        halo_frac = float(np.mean(img[(rr > 1.02) & (rr < 1.12)] > 10))  # how much of it is lit
        prof90 = float(np.median(img[(rr > 0.88) & (rr < 0.92)]))
        prof10 = float(np.median(img[rr < 0.12]))
        limb_depth = prof90 / max(prof10, 1e-6)                          # brightness fall-off
        granulation = float(np.std(fl[qui]))                             # background roughness
        fil_contrast = float(np.median(fl[qui]) - np.median(fl[fil]))    # how dark the filaments are
        med_area = float(np.median([a["area"] for a in by_img[k]]))
        n_fil = len(by_img[k])
        station = fname[k][14:16]

        rec.append(dict(key=k, file=fname[k], station=station, confusion=confusion,
                        halo=halo, halo_frac=halo_frac, limb_depth=limb_depth,
                        granulation=granulation, fil_contrast=fil_contrast,
                        med_area=med_area, n_fil=n_fil))

    conf = np.array([r["confusion"] for r in rec])
    print(f"{len(rec)} images | confusion after flattening: "
          f"mean {conf.mean():.3f}  median {np.median(conf):.3f}  max {conf.max():.3f}\n")

    print("Correlation of post-flattening confusion with each candidate cause")
    print("(Spearman rank correlation; |rho| near 1 means that factor explains it)\n")
    from scipy.stats import spearmanr
    factors = ["halo", "halo_frac", "limb_depth", "granulation",
               "fil_contrast", "med_area", "n_fil"]
    labels = {
        "halo": "brightness of the halo band outside the disk",
        "halo_frac": "how much of the halo band is lit at all",
        "limb_depth": "depth of limb darkening (0.9R / 0.1R)",
        "granulation": "roughness of the flattened background",
        "fil_contrast": "how dark the filaments are vs quiet sun",
        "med_area": "median filament size in that image",
        "n_fil": "number of filaments annotated",
    }
    out = []
    for fkey in factors:
        v = np.array([r[fkey] for r in rec])
        rho, p = spearmanr(v, conf)
        out.append((abs(rho), rho, p, fkey))
    for arho, rho, p, fkey in sorted(out, reverse=True):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
        print(f"  rho={rho:+.3f} p={p:7.5f} {star}  {labels[fkey]}")

    print("\nBy station (mean confusion):")
    bys = collections.defaultdict(list)
    for r in rec:
        bys[r["station"]].append(r["confusion"])
    for st in sorted(bys):
        print(f"  {st}: {np.mean(bys[st]):.3f}  (n={len(bys[st])})")

    worst = sorted(rec, key=lambda r: -r["confusion"])[:5]
    best = sorted(rec, key=lambda r: r["confusion"])[:5]
    print("\nWorst 5:")
    for r in worst:
        print(f"  {r['file']}  confusion={r['confusion']:.3f}  "
              f"fil_contrast={r['fil_contrast']:+.4f}  granulation={r['granulation']:.4f}  "
              f"halo={r['halo']:.0f}  med_area={r['med_area']:.0f}")
    print("Best 5:")
    for r in best:
        print(f"  {r['file']}  confusion={r['confusion']:.3f}  "
              f"fil_contrast={r['fil_contrast']:+.4f}  granulation={r['granulation']:.4f}  "
              f"halo={r['halo']:.0f}  med_area={r['med_area']:.0f}")

    # ---- figure
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    pairs = [("fil_contrast", "filament darkness vs quiet sun"),
             ("granulation", "background roughness (flattened)"),
             ("med_area", "median filament area, px"),
             ("halo", "halo brightness outside the disk"),
             ("limb_depth", "limb darkening depth"),
             ("n_fil", "number of filaments")]
    for a_, (fkey, lab) in zip(ax.ravel(), pairs):
        v = np.array([r[fkey] for r in rec])
        a_.scatter(v, conf, s=26, alpha=.75, edgecolor="k", linewidth=.3)
        rho, p = spearmanr(v, conf)
        a_.set_xlabel(lab); a_.set_ylabel("confusion after flattening")
        a_.set_title(f"rho = {rho:+.2f}  (p={p:.4f})", fontsize=10)
        a_.grid(alpha=.3)
        if fkey == "med_area":
            a_.set_xscale("log")
    fig.suptitle("What still makes an image hard after photometric flattening?", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, .96])
    plt.savefig(FIGS / "05_hard_images.png", dpi=85); plt.close()
    print(f"\nwrote {FIGS / '05_hard_images.png'}")


if __name__ == "__main__":
    main()
