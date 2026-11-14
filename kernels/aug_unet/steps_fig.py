"""The user's pipeline drawn step by step on one validation photograph (figures/pixel_steps.png):

  1. every pixel is classified: probability of being filament (here a fold-0 U-Net; the GBM and PixNet
     give the same kind of map)
  2. pixels above the threshold are filament pixels
  3. pieces lying within a few px of each other are joined (a pixel missed in the middle no longer splits a filament)
  4. groups smaller than the minimum area are dropped (isolated pixels are not filaments)
  5. what is left, each group one filament, against the expert's outlines

    python kernels/aug_unet/steps_fig.py [stem]
"""
import collections, json, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

T, K, A = 0.65, 7, 350           # the fold-0 ensemble's tuned point (kfold.ENS_POINT["d4"])
MAPS = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "out" / "dump" / "flip" / "val_d4"   # a folder of val_d4 maps
OUT = HERE.parents[1] / "figures" / "pixel_steps.png"


def main():
    ann, tr_dir, _ = run.find_data()
    coco = json.load(open(ann))
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    by_file = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_file[fname[a["image_id"]]].append(a)
    stems = sorted(p.stem for p in MAPS.glob("*.png"))
    stem = sys.argv[1] if len(sys.argv) > 1 else max(stems, key=lambda s: len(by_file[s + ".jpeg"]))
    img = cv2.imread(str(tr_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    p = cv2.imread(str(MAPS / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    # the expert with the most filaments on this photograph
    sets = collections.defaultdict(list)
    for a in by_file[stem + ".jpeg"]:
        sets[a["image_id"]].append(a)
    expert = max(sets.values(), key=len)
    gt = np.zeros((run.FULL, run.FULL), np.uint8)
    for a in expert:
        gt |= run.ann_mask(a).astype(np.uint8)

    b = (p >= int(round(255 * T))).astype(np.uint8)
    raw_lab, raw_area = run.components(p, T, 0)
    lab, area = run.components(p, T, K)
    keep = area >= A
    keep[0] = False
    final = np.where(keep[lab], lab, 0)

    # the crop: the 900 px window holding the most expert filament pixels
    s = cv2.boxFilter(gt.astype(np.float32), -1, (900, 900), normalize=False)
    cy, cx = np.unravel_index(np.argmax(s), s.shape)
    y0, x0 = int(np.clip(cy - 450, 0, run.FULL - 900)), int(np.clip(cx - 450, 0, run.FULL - 900))
    win = (slice(y0, y0 + 900), slice(x0, x0 + 900))

    rng = np.random.default_rng(3)
    colours = rng.uniform(0.25, 1.0, (lab.max() + 1, 3))
    colours[0] = 0

    def groups(L, ids_small=None):
        rgb = colours[L[win] % len(colours)] if L.max() < len(colours) else colours[L[win] % len(colours)]
        rgb = rgb.copy()
        if ids_small is not None:
            small = ids_small[L[win]]
            rgb[small] = (1.0, 0.1, 0.1)
        return rgb

    fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
    ax = ax.ravel()
    ax[0].imshow(img[win], cmap="gray")
    ax[0].set_title("1. Hα image (900 px crop)")
    ax[1].imshow(p[win] / 255.0, cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("2. per-pixel probability of 'filament'")
    n_raw = int((raw_area[1:] > 0).sum())
    ax[2].imshow(b[win], cmap="gray_r")
    ax[2].set_title(f"3. classified as filament (p ≥ {T}): {n_raw} pieces")
    small_raw = np.zeros(len(raw_area), bool)
    small_raw[1:] = raw_area[1:] < A
    colours_raw = rng.uniform(0.25, 1.0, (raw_lab.max() + 1, 3))
    colours_raw[0] = 1.0
    rgb = colours_raw[raw_lab[win]]
    rgb[small_raw[raw_lab[win]] & (raw_lab[win] > 0)] = (1.0, 0.1, 0.1)
    ax[3].imshow(rgb)
    ax[3].set_title(f"4. pieces (red: smaller than {A} px)")
    rgb = np.ones((900, 900, 3))
    L = lab[win]
    on = L > 0
    rgb[on] = colours[L[on]]
    rgb[on & ~keep[L]] = (1.0, 0.1, 0.1)
    ax[4].imshow(rgb)
    ax[4].set_title(f"5. joined within {K} px; red groups dropped\n"
                    f"{int(keep.sum())} filaments on the whole disk")
    ax[5].imshow(img[win], cmap="gray")
    ov = np.zeros((900, 900, 4))
    F = final[win]
    # thin filaments: opaque fill, no resampling blur (antialiasing averaged the alpha away)
    ov[F > 0] = np.concatenate([colours[F[F > 0]], np.ones((int((F > 0).sum()), 1))], 1)
    ax[5].imshow(ov, interpolation="nearest")
    cs = cv2.findContours(gt[win].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    for c in cs:
        c = c[:, 0]
        ax[5].plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="cyan", lw=1.2)
    ax[5].set_title(f"6. result (colour) vs one expert (cyan)\nexpert: {len(expert)} filaments on the disk, "
                    f"model: {int(keep.sum())}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{stem}: from pixel classification to filaments (fold-0 validation photograph, "
                 f"threshold {T}, join {K} px, minimum {A} px)", fontsize=13, y=0.995)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT, dpi=110)
    print("wrote", OUT, "stem", stem, "pieces", n_raw, "after join", int((area[1:] > 0).sum()), "kept", int(keep.sum()))


if __name__ == "__main__":
    main()
