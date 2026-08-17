"""How many annotated filaments actually sit on dark material?

A filament is dark chromospheric absorption. After flattening, the pixels inside
a correct annotation must be darker than the ring of quiet sun around it. Any
annotation where that is not true is either misplaced, degenerate, or not a
filament -- and training on it teaches the model to call quiet sun a filament.

Contrast is measured locally (against a ring around each polygon) rather than
against the whole disk, so plage and active regions nearby cannot fake it.
"""
from __future__ import annotations

import sys, json, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
PRE = ROOT / "preprocessed" / "train"
FIGS = ROOT / "figures"

OFFSET, SCALE = 0.5, 255.0


def decode(byte_img):
    return byte_img.astype(np.float32) / SCALE + OFFSET


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    by_file = collections.defaultdict(list)
    for iid, f in fname.items():
        by_file[f].append(iid)

    rows = []
    done = 0
    for f, iids in sorted(by_file.items()):
        p = PRE / (Path(f).stem + ".png")
        if not p.exists():
            continue
        flat = decode(np.asarray(Image.open(p)))
        ondisk = flat > (OFFSET + 0.02)      # off-disk was written as 0 -> decodes to 0.5
        for iid in iids:
            anns = by_img[iid]
            # union of everything annotated on this image, to keep it out of backgrounds
            allm = Image.new("L", (2048, 2048), 0)
            dr = ImageDraw.Draw(allm)
            for a in anns:
                pp = a["segmentation"][0]
                dr.polygon(list(zip(pp[0::2], pp[1::2])), fill=1)
            allm = np.asarray(allm, bool)

            for a in anns:
                pp = a["segmentation"][0]
                x, y, w, h = a["bbox"]
                pad = int(max(20, 0.6 * max(w, h)))
                X0, Y0 = int(max(x - pad, 0)), int(max(y - pad, 0))
                X1, Y1 = int(min(x + w + pad, 2048)), int(min(y + h + pad, 2048))
                if X1 - X0 < 4 or Y1 - Y0 < 4:
                    continue
                sub = flat[Y0:Y1, X0:X1]
                m = Image.new("L", (X1 - X0, Y1 - Y0), 0)
                ImageDraw.Draw(m).polygon(
                    [(px - X0, py - Y0) for px, py in zip(pp[0::2], pp[1::2])], fill=1)
                m = np.asarray(m, bool)
                if m.sum() < 8:
                    continue
                ring = binary_dilation(m, iterations=12) & ~allm[Y0:Y1, X0:X1] \
                    & ondisk[Y0:Y1, X0:X1]
                if ring.sum() < 50:
                    continue
                inside = float(np.median(sub[m]))
                around = float(np.median(sub[ring]))
                rows.append(dict(iid=iid, file=f, batch=iid.split("-")[0],
                                 area=a["area"], contrast=around - inside))
        done += 1

    print(f"{done} photographs processed, {len(rows)} annotations measured\n")
    c = np.array([r["contrast"] for r in rows])
    a = np.array([r["area"] for r in rows])

    print("Local contrast (quiet ring minus filament interior, flattened units)")
    print(f"  median {np.median(c):+.4f}   mean {c.mean():+.4f}")
    for q in (1, 5, 10, 25):
        print(f"  p{q:<2}    {np.percentile(c, q):+.4f}")
    print(f"\n  annotations NOT darker than their surroundings (contrast <= 0): "
          f"{100*np.mean(c <= 0):.2f}%  ({int((c<=0).sum())} of {len(c)})")
    print(f"  barely darker (0 < contrast < 0.01):                            "
          f"{100*np.mean((c>0)&(c<0.01)):.2f}%")

    print("\nBy filament size:")
    for lo, hi, lab in ((0, 500, "< 500 px"), (500, 1500, "500-1500"),
                        (1500, 4000, "1500-4000"), (4000, 1e9, "> 4000 px")):
        s = (a >= lo) & (a < hi)
        if s.sum():
            print(f"  {lab:<10} n={s.sum():5d}  median contrast {np.median(c[s]):+.4f}  "
                  f"non-dark {100*np.mean(c[s] <= 0):5.2f}%")

    print("\nWorst annotator batches (by share of non-dark annotations, n>=40):")
    byb = collections.defaultdict(list)
    for r in rows:
        byb[r["batch"]].append(r["contrast"])
    stats = [(np.mean(np.array(v) <= 0), np.median(v), len(v), b)
             for b, v in byb.items() if len(v) >= 40]
    for frac, med, n, b in sorted(stats, reverse=True)[:8]:
        print(f"  batch {b}: n={n:4d}  non-dark {100*frac:5.2f}%  median contrast {med:+.4f}")
    print("Best batches:")
    for frac, med, n, b in sorted(stats)[:4]:
        print(f"  batch {b}: n={n:4d}  non-dark {100*frac:5.2f}%  median contrast {med:+.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist(c, bins=90, range=(-0.15, 0.35), color="#4a7fb5", edgecolor="k", lw=.3)
    ax[0].axvline(0, color="r", lw=2)
    ax[0].set_xlabel("local contrast (positive = darker than surroundings)")
    ax[0].set_ylabel("annotations")
    ax[0].set_title(f"{100*np.mean(c<=0):.1f}% of annotations are not darker "
                    f"than their surroundings", fontsize=11)

    ax[1].scatter(a, c, s=4, alpha=.18, edgecolor="none")
    ax[1].axhline(0, color="r", lw=1.5); ax[1].set_xscale("log")
    ax[1].set_xlabel("annotation area, px"); ax[1].set_ylabel("local contrast")
    ax[1].set_title("small annotations are the unreliable ones", fontsize=11)
    ax[1].grid(alpha=.3)

    fr = [100 * np.mean(np.array(v) <= 0) for b, v in sorted(byb.items()) if len(v) >= 40]
    bl = [b for b, v in sorted(byb.items()) if len(v) >= 40]
    ax[2].bar(range(len(fr)), fr, color="#c1543a")
    ax[2].set_xticks(range(len(bl))); ax[2].set_xticklabels(bl, rotation=90, fontsize=7)
    ax[2].set_ylabel("% annotations not darker than surroundings")
    ax[2].set_title("annotator batches differ", fontsize=11); ax[2].grid(alpha=.3, axis="y")

    plt.tight_layout(); plt.savefig(FIGS / "07_label_quality.png", dpi=85); plt.close()
    print(f"\nwrote {FIGS / '07_label_quality.png'}")

    json.dump(rows, open(ROOT / "label_contrast.json", "w"))
    print(f"wrote {ROOT / 'label_contrast.json'} ({len(rows)} records)")


if __name__ == "__main__":
    main()
