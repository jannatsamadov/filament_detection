"""Does predicting at more than one scale help? Fold 0's R50 on its 141 validation photographs.

D4 at scale 1.0 is already dumped (dump/r50/val_d4). This adds D4 at 0.875 and 1.125,
each mapped back to 2048 and written as its own "model", so post_local's grid scores
every mix of them the same way as everything before. Runs on the local GPU (opt-in).
    python kernels/aug_unet/scale_tta.py
"""
from __future__ import annotations

import os, subprocess, sys, time
from pathlib import Path

os.environ["FIL_LOCAL_GPU"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import post_local as P  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

D = HERE.parent / "aug_post" / "out"
CKPT = HERE.parent / "unet_r2" / "out_kaggle" / "best_r50.pt"
SCALES = (0.875, 1.125)


class Serial:
    def imap_unordered(self, f, jobs, chunksize=1):
        return map(f, jobs)


def gpu():
    q = "--query-gpu=temperature.gpu,utilization.gpu,power.draw"
    return subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def predict_at(model, im, device, s):
    """D4 prediction with the frame resized by s, returned on the 2048 grid."""
    n = int(round(run.FULL * s / 32)) * 32
    small = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    yy, xx = np.mgrid[:n, :n]
    c, r = (n - 1) / 2, run.R_DISK * n / run.FULL
    ondisk = torch.from_numpy((((xx - c) ** 2 + (yy - c) ** 2) <= (r * 0.995) ** 2).astype(np.float32)).to(device)
    with torch.no_grad():
        p = run.predict(model, small, device, ondisk, tta="d4")
    return cv2.resize(p, (run.FULL, run.FULL), interpolation=cv2.INTER_LINEAR)


def main():
    meta, gt = P.ground_truth()
    stems = [Path(f).stem for f in meta["val_files"]]
    img = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    device = torch.device("cuda")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = run.UNet("resnet50", pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    print("start", gpu(), flush=True)
    t0, hot = time.time(), 0
    for s in SCALES:
        out = D / "dump" / f"r50s{int(s * 1000)}" / "val_d4"
        out.mkdir(parents=True, exist_ok=True)
        for i, stem in enumerate(stems):
            if not (out / f"{stem}.png").exists():
                cv2.imwrite(str(out / f"{stem}.png"), predict_at(model, np.asarray(img[i]), device, s))
            if i % 20 == 0:
                g = gpu()
                hot = max(hot, int(g.split(",")[0]))
                print(f"scale {s} {i}/{len(stems)} {time.time() - t0:.0f}s gpu {g}", flush=True)
    print(f"max temperature seen {hot} C", flush=True)
    del model
    torch.cuda.empty_cache()
    a, b = (f"r50s{int(s * 1000)}" for s in SCALES)
    for v in ("r50/d4", f"ens:r50+{a}+{b}/d4", f"ens:r50+{b}/d4", f"ens:r50+{a}/d4"):
        best, _ = P.grid(D, v, meta, gt, Serial())
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in best.items() if k != "variant"}, flush=True)


if __name__ == "__main__":
    main()
