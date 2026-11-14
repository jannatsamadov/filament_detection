"""Can a general vision-language model tell a real filament from a false candidate?

The keep-rule judges each candidate from 10-21 numbers; the crop-CNN (keep_cnn.py)
from the pixels, and it matched the 17-feature rule without beating it. This asks
Qwen3-VL, zero-shot: shown the crop around a candidate, with and without the
candidate outlined in red, does the outline hold a solar filament? Its P("yes")
is scored three ways on fold 0's validation candidates (R50 fold 0, D4, the
rule's setting t 0.6 / merge 3 / >= 100 px):

  1. AUC against "matched in at least one annotation set", next to the GBM's
  2. the GBM keep-rule with P(yes) as one more feature, out-of-fold
  3. PQ at the best cut for each, and the paired bootstrap of the difference

Local GPU (FIL_LOCAL_GPU=1 is set here). Candidates and scores are cached, so a
crash of this laptop costs nothing but the frame it was on.
    python kernels/aug_unet/qwen_probe.py --model Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")          # before run.py hides the card
import argparse, json, pickle, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if not Path("/kaggle/working").exists():
    os.environ.setdefault("HF_HOME", str(ROOT / ".hfhome"))   # the model cache stays on D:
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from post_local import comp_features, ground_truth, matched_iou, pq_of  # noqa: E402

T, K, AMIN = 0.6, 3, 100
PAD, WIN, SIDE = 24, (96, 512), 448
# on Kaggle the bulky candidates stay in /tmp; results go to /kaggle/working (RES)
OUT = Path("/tmp/qwen") if run.ON_KAGGLE else HERE / "out" / "qwen_probe"
RES = run.WORK / "qwen" if run.ON_KAGGLE else OUT
LOCAL_DUMPS = {"r50": "unet_r2", "r50f1": "unet_r5", "r50f2": "unet_r5", "r50f3": "unet_r6", "r50f4": "unet_r6"}


def find_dump(model):
    """The dump tar of one model: an attached kernel's output on Kaggle, the downloaded copy here."""
    if run.ON_KAGGLE:
        return next(Path("/kaggle/input").rglob(f"dump_{model}.tar"))
    return ROOT / "kernels" / LOCAL_DUMPS[model] / "out_kaggle" / f"dump_{model}.tar"
PROMPT = (
    "Both panels show the same crop of a full-disk H-alpha image of the Sun (GONG network). "
    "Left: the crop as observed. Right: the same crop with a red outline around a candidate "
    "region proposed by an automatic detector.\n"
    "A solar filament is a dark, elongated, often thread-like or ribbon-like absorption feature "
    "on the bright chromospheric disk. Not filaments: sunspots (compact, very dark, roundish), "
    "dark gaps between bright plage, small dark mottles, noise, and darkening at the limb.\n"
    "Does the red outline enclose a solar filament (most of one, not just a fragment of noise)? "
    "Answer with one word: yes or no.")


def candidates():
    """Fold 0's validation candidates: crops, the 10 hand features, matched IoU per set."""
    path = OUT / "candidates.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    meta, gt = ground_truth()
    by = {}
    for i, g in enumerate(gt):
        by.setdefault(g["frame"], []).append((i, g["crops"]))
    tar = tarfile.open(find_dump("r50"))
    names = {Path(n).stem: n for n in tar.getnames() if "val_d4/" in n and n.endswith(".png")}
    val_img = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    from scipy import ndimage
    rows = []
    for f, sets in sorted(by.items()):
        stem = Path(meta["val_files"][f]).stem
        p = cv2.imdecode(np.frombuffer(tar.extractfile(names[stem]).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = np.asarray(val_img[f])
        lab, area = run.components(p, T, K)
        ids = [j for j in range(1, len(area)) if area[j] >= AMIN]
        if not ids:
            rows.append(dict(stem=stem, X=np.zeros((0, 10)), Y=np.zeros((0, len(sets))), n_gt=[len(c) for _, c in sets], crops=[]))
            continue
        boxes = ndimage.find_objects(lab)
        crops = []
        for j in ids:
            sy, sx = boxes[j - 1]
            side = int(np.clip(max(sy.stop - sy.start, sx.stop - sx.start) + 2 * PAD, *WIN))
            y0, x0 = (sy.start + sy.stop - side) // 2, (sx.start + sx.stop - side) // 2
            y0, x0 = int(np.clip(y0, 0, run.FULL - side)), int(np.clip(x0, 0, run.FULL - side))
            crops.append((img[y0:y0 + side, x0:x0 + side].copy(), (lab[y0:y0 + side, x0:x0 + side] == j)))
        rows.append(dict(stem=stem, X=comp_features(p, img, lab, area, ids), Y=matched_iou(lab, area, ids, sets),
                         n_gt=[len(c) for _, c in sets], crops=crops))
        print(f"candidates: {len(rows)}/{len(by)} frames, {sum(len(r['crops']) for r in rows)} so far", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(path, "wb"))
    return rows


def panel(im, m):
    """Left the crop, right the crop with the candidate's outline in red; contrast stretched."""
    x = im.astype(np.float32)
    lo, hi = np.percentile(x[x > 0], (1, 99.5)) if (x > 0).any() else (0, 255)
    g = np.clip((x - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    g = cv2.resize(g, (SIDE, SIDE), interpolation=cv2.INTER_AREA if g.shape[0] > SIDE else cv2.INTER_CUBIC)
    mm = cv2.resize(m.astype(np.uint8), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    left = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    right = left.copy()
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(right, cs, -1, (255, 0, 0), 2)
    gap = np.full((SIDE, 8, 3), 255, np.uint8)
    return np.concatenate([left, gap, right], 1)


def qwen_scores(rows, model_id, four_bit=False):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    tag = model_id.split("/")[-1] + ("-4bit" if four_bit else "")
    path = OUT / f"scores_{tag}.npy"
    flat = [c for r in rows for c in r["crops"]]
    done = np.load(path) if path.exists() else np.full(len(flat), np.nan)
    if np.isfinite(done).all():
        return done
    dtype = torch.float16
    extra = {}
    if four_bit:                         # 4B and up do not fit the laptop's 6 GB in fp16
        from transformers import BitsAndBytesConfig
        extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="cuda:0", **extra)
    model.eval()
    proc = AutoProcessor.from_pretrained(model_id)
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    for n, (im, m) in enumerate(flat):
        if np.isfinite(done[n]):
            continue
        inp = proc(text=[text], images=[Image.fromarray(panel(im, m))], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        py, pn = torch.logsumexp(lp[yes], 0), torch.logsumexp(lp[no], 0)
        done[n] = float(torch.sigmoid(py - pn))
        if (n + 1) % 100 == 0 or n + 1 == len(flat):
            np.save(path, done)
            print(f"{tag}: {n + 1}/{len(flat)}  {(time.time() - t0) / (n + 1):.2f} s each", flush=True)
    np.save(path, done)
    return done


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, s)), 4)


def evaluate(rows, q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([r["X"] for r in rows])
    Y = [r["Y"] for r in rows]
    y = np.concatenate([v.mean(axis=1) for v in Y])
    groups = np.concatenate([[n] * len(r["X"]) for n, r in enumerate(rows)])
    hit = (y > 0).astype(int)

    def oof(F):
        o = np.zeros(len(y))
        for fold in range(5):
            te = groups % 5 == fold
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=fold)
            o[te] = m.fit(F[~te], y[~te]).predict(F[te])
        return o

    def counts(s, cut):
        c = np.zeros((len(rows), 4))
        for n, r in enumerate(rows):
            keep = s[groups == n] >= cut
            yy = r["Y"][keep]
            for qq, g in enumerate(r["n_gt"]):
                tp = (yy[:, qq] > 0).sum()
                c[n] += (tp, keep.sum() - tp, g - tp, yy[:, qq].sum())
        return c

    def best(s):
        cuts = np.unique(np.quantile(s, np.linspace(0, 0.95, 96)))
        pq = [float(pq_of(counts(s, c).sum(axis=0))) for c in cuts]
        b = int(np.argmax(pq))
        return counts(s, cuts[b]), round(pq[b], 4)

    gbm, both = oof(X), oof(np.concatenate([X, q[:, None]], 1))
    cg, pg = best(gbm)
    cb, pb = best(both)
    cq, pqq = best(q)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        s = rng.integers(0, len(rows), len(rows))
        d.append(float(pq_of(cb[s].sum(axis=0)) - pq_of(cg[s].sum(axis=0))))
    return dict(candidates=int(len(y)), matched=int(hit.sum()),
                auc=dict(qwen=auc(q, hit), gbm10=auc(gbm, hit), gbm10_plus_qwen=auc(both, hit)),
                pq_best_cut=dict(qwen_alone=pqq, gbm10=pg, gbm10_plus_qwen=pb),
                plus_qwen_minus_gbm=dict(mean=round(float(np.mean(d)), 4), lo=round(float(np.percentile(d, 2.5)), 4),
                                         hi=round(float(np.percentile(d, 97.5)), 4)),
                qwen_mean_matched=round(float(q[hit == 1].mean()), 4), qwen_mean_unmatched=round(float(q[hit == 0].mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--panels", type=int, default=0, help="only write this many example panels and stop")
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()
    rows = candidates()
    if a.panels:
        flat = [(r["stem"], c, r["Y"][i].mean()) for r in rows for i, c in enumerate(r["crops"])]
        for n, (stem, (im, m), y) in enumerate(flat[:a.panels]):
            cv2.imwrite(str(OUT / f"panel_{n:02d}_{'hit' if y > 0 else 'miss'}.png"), panel(im, m)[:, :, ::-1])
        return
    q = qwen_scores(rows, a.model, a.four_bit)
    res = evaluate(rows, q)
    res["model"] = a.model + (" (4-bit)" if a.four_bit else "")
    json.dump(res, open(OUT / f"result_{a.model.split('/')[-1]}{'-4bit' if a.four_bit else ''}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
