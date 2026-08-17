"""Does flattening actually make filaments easier to separate? Measure it."""
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
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from geometry import fit_disk, radial_coordinates      # noqa: E402
from photometry import flatten, quiet_sun_profile      # noqa: E402


def masks_for(anns, shape=(2048, 2048)):
    m = Image.new("L", shape[::-1], 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        p = a["segmentation"][0]
        d.polygon(list(zip(p[0::2], p[1::2])), fill=1)
    return np.array(m, bool)


def separability(values_fil, values_quiet):
    """Fraction of quiet-sun pixels darker than the median filament pixel.
    Lower is better: it is the confusion a brightness threshold cannot avoid."""
    return float(np.mean(values_quiet < np.median(values_fil)))


def main(n_images=40):
    d = json.load(open(BASE / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    keys = sorted(by_img)
    random.seed(0); random.shuffle(keys)

    raw_sep, flat_sep, raw_c, flat_c = [], [], [], []
    used = 0
    for k in keys:
        if used >= n_images:
            break
        f = IMGS / fname[k]
        if not f.exists():
            continue
        img = np.array(Image.open(f).convert("L")).astype(np.float32)
        fit = fit_disk(img)
        rr, _ = radial_coordinates(img.shape, fit)
        fl = flatten(img, rr)
        M = masks_for(by_img[k])
        disk = rr <= 0.98
        fil, qui = M & disk, (~M) & disk
        if fil.sum() < 500:
            continue
        raw_sep.append(separability(img[fil], img[qui]))
        flat_sep.append(separability(fl[fil], fl[qui]))
        # contrast: how many quiet-sun sigmas below the quiet mean a filament sits
        raw_c.append((np.median(img[qui]) - np.median(img[fil])) / np.std(img[qui]))
        flat_c.append((np.median(fl[qui]) - np.median(fl[fil])) / np.std(fl[qui]))
        used += 1

    print(f"{used} images\n")
    print("Fraction of quiet-sun pixels darker than the median filament pixel")
    print("  (this is the irreducible confusion for a brightness threshold; lower is better)")
    print(f"    raw       : {np.mean(raw_sep):.3f}   (median {np.median(raw_sep):.3f})")
    print(f"    flattened : {np.mean(flat_sep):.3f}   (median {np.median(flat_sep):.3f})")
    print()
    print("Filament contrast, in units of quiet-sun standard deviation (higher is better)")
    print(f"    raw       : {np.mean(raw_c):.3f}")
    print(f"    flattened : {np.mean(flat_c):.3f}")
    print()
    print(f"image-to-image spread of the confusion figure (lower = more consistent)")
    print(f"    raw       : std {np.std(raw_sep):.4f}   range {min(raw_sep):.3f}-{max(raw_sep):.3f}")
    print(f"    flattened : std {np.std(flat_sep):.4f}   range {min(flat_sep):.3f}-{max(flat_sep):.3f}")

    # ---------- figure ----------
    k = keys[0]
    img = np.array(Image.open(IMGS / fname[k]).convert("L")).astype(np.float32)
    fit = fit_disk(img)
    rr, _ = radial_coordinates(img.shape, fit)
    fl = flatten(img, rr)
    M = masks_for(by_img[k]); disk = rr <= 0.98

    fig, ax = plt.subplots(2, 3, figsize=(19, 12))
    ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title(f"raw   {fname[k]}", fontsize=11); ax[0, 0].axis("off")

    v = fl[disk]
    lo, hi = np.percentile(v, [1, 99])
    ax[0, 1].imshow(np.where(disk, fl, np.nan), cmap="gray", vmin=lo, vmax=hi)
    ax[0, 1].set_title("flattened (quiet sun = 1.0 everywhere)", fontsize=11); ax[0, 1].axis("off")

    centres, prof = quiet_sun_profile(img, rr)
    ax[0, 2].plot(centres, prof, lw=2, label="quiet-sun profile (75th pct)")
    bins = np.linspace(0, 1, 101)
    idx = np.digitize(rr.ravel(), bins)
    med = [np.median(img.ravel()[idx == i]) if (idx == i).sum() > 40 else np.nan
           for i in range(1, len(bins))]
    ax[0, 2].plot(0.5 * (bins[1:] + bins[:-1]), med, lw=1.2, ls="--", label="median profile")
    ax[0, 2].set_xlabel("r / R_sun"); ax[0, 2].set_ylabel("brightness")
    ax[0, 2].legend(); ax[0, 2].grid(alpha=.3)
    ax[0, 2].set_title("what gets divided out", fontsize=11)

    # zoom on a limb-side filament, raw vs flattened
    anns = sorted(by_img[k], key=lambda a: -(a["bbox"][0] - 1024) ** 2 - (a["bbox"][1] - 1024) ** 2)
    a = anns[0]
    x, y, w, h = a["bbox"]; pad = 80
    X0, Y0 = int(max(x - pad, 0)), int(max(y - pad, 0))
    X1, Y1 = int(min(x + w + pad, 2048)), int(min(y + h + pad, 2048))
    for j, (arr, name) in enumerate(((img, "raw"), (fl, "flattened"))):
        sub = arr[Y0:Y1, X0:X1]
        ax[1, j].imshow(sub, cmap="gray")
        p = a["segmentation"][0]
        ax[1, j].plot([q - X0 for q in p[0::2]] + [p[0] - X0],
                      [q - Y0 for q in p[1::2]] + [p[1] - Y0], "r-", lw=1.5)
        ax[1, j].set_title(f"{name} — filament near the limb", fontsize=11); ax[1, j].axis("off")

    ax[1, 2].hist(fl[M & disk], bins=70, range=(0.7, 1.3), density=True, alpha=.6, label="filaments")
    ax[1, 2].hist(fl[(~M) & disk], bins=70, range=(0.7, 1.3), density=True, alpha=.6, label="quiet disk")
    ax[1, 2].axvline(np.median(fl[M & disk]), color="k", ls="--", lw=1)
    ax[1, 2].legend(); ax[1, 2].set_xlabel("flattened brightness")
    ax[1, 2].set_title("after flattening: still heavily overlapping", fontsize=11)

    plt.tight_layout(); plt.savefig(FIGS / "04_flatten.png", dpi=82); plt.close()
    print(f"\nwrote {FIGS / '04_flatten.png'}")


if __name__ == "__main__":
    main()
