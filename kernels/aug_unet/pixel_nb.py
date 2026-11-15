"""The user's pixel table with every neighbour's own value (2026-11-14, built as the user specified).

One row per pixel of the 2048 px photograph, on the solar disk (off the disk is cut first, so a
neighbour beyond the limb is 0):
  - the 35 columns of the earlier pixel table (pixel_gbm.PIX_FEAT_A, computed at 1024, read at the pixel)
  - the pixel's position x, y (px) and its own contrast c_dx0_dy0
  - every pixel within 12 px of it (440 neighbours): its contrast c_dx.._dy.., and its position
    x_dx.._dy.., y_dx.._dy.. (all three 0 when the neighbour is off the disk)
    contrast = brightness / the quiet-Sun brightness at the same distance from the disk centre
  - answer columns, never inputs: how many experts drew the pixel, of how many, and the two targets
    (share of experts; the user's scale 1 -> 0.45, 2 -> 0.65, 3 -> 0.95)
A training photograph enters five times: as it is, and copies with only contrast, only sharpness,
only brightness, and all three changed; every column of a copy is taken from the copy. Validation
photographs are used as they are.

Arms (random forest, the earlier settings: 100 trees, 10% of the rows each, leaf >= 50, sqrt columns;
as a regression forest on the soft target, whose splits and leaf values equal the classifier's on
weighted positive/negative copies of each row):
  nb        the user's table, target = share of experts
  nb_count  the user's table, target = the user's expert-count scale
  ctrl      the 35 earlier columns only, the same rows and target as `nb` (the control that says what
            the neighbour columns add)
Fold 0: 566 training photographs, 141 validation photographs; the maps go to aug_unet/pixel_cls.py.
    kernels/build.py builds kernels/pixel_nb/run.py
    python pixel_nb.py --head     the table of two photographs: data.head(), data.info()  (local)
"""
from __future__ import annotations

import collections, gc, json, multiprocessing as mp, os, shutil, sys, tarfile, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import pixel_gbm as G  # noqa: E402

SMOKE = os.environ.get("FIL_NB_SMOKE") == "1"
NB_LEARNER = "reg"                            # "cls" in pixel_nbc/run.py: the classifier the user asked for (11-14)
if os.environ.get("FIL_NB_CLS") == "1":       # local smoke of the classifier
    NB_LEARNER = "cls"
RADIUS = 12                                   # px at 2048 (user)
VERSIONS = ("orig", "contrast", "sharp", "bright", "all")
N_ROWS = dict(pos=240, ring=90, far=240)      # per photograph and version: 566 x 5 x 570 = 1.6 M rows, ~8.8 GB
N_VAL_ROWS = dict(pos=600, ring=200, far=600)
COUNT_SCALE = np.array([0.0, 0.45, 0.65, 0.95], np.float32)
RF = dict(n_estimators=5 if SMOKE else 100, max_samples=0.1, min_samples_leaf=50, max_features="sqrt")
CHUNK = 60000
WORK = run.WORK
# the Sun's rotation (user, 11-14): how far the pixel moves per hour, and the same solar point in other
# photographs carried there by differential rotation (kfold.carry: Snodgrass, synodic, B0 of each frame)
NB_ROT = False                                # True in pixel_nbr/run.py
if os.environ.get("FIL_NB_ROT") == "1":       # smoke
    NB_ROT = True
ROT_R = 6                                     # px at 2048 around the carried point (user: 3 or 6; 6 covers ~2 days' error)
ROT_BANDS = ((0, 24), (24, 48), (48, 84))     # hours, each band its own columns (user: up to 84 h)
ROT_SLOTS = [(d, lo, hi) for d in ("before", "after") for lo, hi in ROT_BANDS]
_ro = [(dy, dx) for dy in range(-ROT_R, ROT_R + 1) for dx in range(-ROT_R, ROT_R + 1) if dy * dy + dx * dx <= ROT_R ** 2]
ROFFS = sorted(_ro, key=lambda o: (o[0] ** 2 + o[1] ** 2, np.arctan2(o[0], o[1])))

# offsets within RADIUS, the pixel itself first, then by distance
_o = [(dy, dx) for dy in range(-RADIUS, RADIUS + 1) for dx in range(-RADIUS, RADIUS + 1) if dy * dy + dx * dx <= RADIUS ** 2]
OFFS = sorted(_o, key=lambda o: (o[0] ** 2 + o[1] ** 2, np.arctan2(o[0], o[1])))
DY = np.array([o[0] for o in OFFS], np.int32)
DX = np.array([o[1] for o in OFFS], np.int32)
OLD = list(G.PIX_FEAT_A)
COLS = OLD + ["x", "y", "c_dx0_dy0"]
for dy, dx in OFFS[1:]:
    COLS += [f"c_dx{dx}_dy{dy}", f"x_dx{dx}_dy{dy}", f"y_dx{dx}_dy{dy}"]
ANSWER = ["experts_marked", "n_experts", "target_share", "target_count"]
BASE_N = len(COLS)                            # the user's 1358 columns; the rotation columns follow
if NB_ROT:
    COLS += ["vx_px_per_h", "vy_px_per_h"]
    for _d, _lo, _hi in ROT_SLOTS:
        _p = f"{_d}{_lo}_{_hi}h"
        COLS += [f"{_p}_dt_h", f"{_p}_x", f"{_p}_y"] + [f"{_p}_c_dx{dx}_dy{dy}" for dy, dx in ROFFS]
NCOL = len(COLS)
ARM_COLS = {"nb": slice(0, NCOL), "nb_count": slice(0, NCOL), "ctrl": slice(0, len(OLD)),
            "nbc": slice(0, NCOL), "ctrlc": slice(0, len(OLD)), "nbcr": slice(0, NCOL)}
# Claude's arm (11-15), next to the user's table, not instead of it: nbc fell below its 35-column
# control (pixel F1 0.443 vs 0.530). The 880 neighbour positions are the pixel's own x, y plus a
# constant, and a forest that draws sqrt(1358) = 37 columns per split rarely sees the useful ones
# among them. nbcc keeps everything but those positions: 35 + x, y, own contrast + 440 contrasts.
ARM_COLS["nbcc"] = np.r_[0:len(OLD) + 3, len(OLD) + 3:BASE_N:3]
ARM_TARGET = {"nb": "share", "nb_count": "count", "ctrl": "share", "nbc": "share", "ctrlc": "share", "nbcr": "share",
              "nbcc": "share"}
NB_CC = os.environ.get("FIL_NB_CC") == "1"     # kernel filament-pixel-nbcc (with NB_LEARNER = "cls")
# the user (11-15): the 1358 columns need a learner sized for them, not the 35-column forest's settings
# (100 trees, sqrt = 37 columns per split, leaves of 50). Kernels filament-pixel-nbc-rf / -xgb, the classifier's rows
NB_TUNE = os.environ.get("FIL_NB_TUNE", "")
RF_TUNED = dict(n_estimators=300, max_samples=0.1, min_samples_leaf=20, max_features=0.1)   # 136 columns per split
XGB = dict(n_estimators=400, max_depth=8, learning_rate=0.08, subsample=0.7, colsample_bytree=0.3, min_child_weight=5,
           max_bin=64, tree_method="hist")
ARMS = ("ctrl", "nb", "nb_count") if NB_LEARNER == "reg" else ("ctrlc", "nbc")
MAIN_ARM = "nb" if NB_LEARNER == "reg" else "nbc"
TAG = "nb" if NB_LEARNER == "reg" else "nbc"
if NB_ROT:                                    # pixel_nbc (same rows, same forest, no rotation columns) is its control
    ARMS, MAIN_ARM, TAG = ("nbcr",), "nbcr", "nbcr"
if NB_CC:                                     # its control is pixel_nbc's ctrlc and nbc: same rows, same forest
    ARMS, MAIN_ARM, TAG = ("nbcc",), None, "nbcc"
if NB_TUNE:                                   # its control is pixel_nbc's nbc: same rows, the default forest
    ARMS, MAIN_ARM, TAG = ("nbc",), None, f"nbc_{NB_TUNE}"
TAG += "_smoke" if SMOKE else ""
MAPS = (Path("/tmp") if run.ON_KAGGLE else WORK) / f"{TAG}_maps"
ROT_SLOTS_OF = {}                             # file -> per slot (other file, gap in days) or None

_c = (np.arange(run.FULL, dtype=np.float32) - run.CX) / run.R_DISK
RHO2 = np.hypot(_c[None, :], _c[:, None])
ON2 = RHO2 < 1.0                              # the cut: the solar disk at 2048
ON2P = np.pad(ON2, RADIUS)
SETS, MODELS = {}, {}
TR_DIR = None


# ---------------------------------------------------------------- the table

def truth(f):
    cnt = np.zeros((run.FULL, run.FULL), np.uint8)
    for anns in SETS[f]:
        u = np.zeros((run.FULL, run.FULL), bool)
        for a in anns:
            u |= run.ann_mask(a)
        cnt += u
    return cnt, len(SETS[f])


def augment(img, kind, rng):
    """Only contrast, only sharpness, only brightness, or all three, on the disk; ranges of run.photometric
    (measured between frames: contrast 1.7-1.85x, sharpness 2.2-2.3x, brightness 1.05x)."""
    if kind == "orig":
        return img
    x = img.astype(np.float32)
    y = x
    if kind in ("contrast", "all"):
        med = float(np.median(x[724:1324, 724:1324]))
        y = med + float(np.exp(rng.uniform(np.log(0.6), np.log(1.65)))) * (y - med)
    if kind in ("bright", "all"):
        y = y + float(rng.choice([-1, 1]) * rng.uniform(2.0, 6.0))
    if kind in ("sharp", "all"):
        if rng.random() < 0.5:                 # softer (worse seeing)
            y = cv2.GaussianBlur(y, (0, 0), float(rng.uniform(0.5, 1.8)))
        else:                                  # sharper (unsharp mask)
            s, a = float(rng.uniform(1.0, 3.0)), float(rng.uniform(0.3, 1.2))
            y = y + a * (y - cv2.GaussianBlur(y, (0, 0), s))
    w = run.disk_weight()
    return np.clip(x + w * (y - x) + 0.5, 0, 255).astype(np.uint8)


def maps_of(img, stem):
    """The 35 earlier columns as a (1024, 1024, 35) map, and the contrast at 2048 (0 off the disk), padded."""
    F = np.zeros((G.PIX_HALF, G.PIX_HALF, len(OLD)), np.float32)
    F[G.PIX_G[3]] = G.pix_features(img, stem)
    return F, np.pad(contrast(img), RADIUS)


def contrast(img):
    """Brightness / the quiet-Sun brightness at the same distance from the centre, at 2048; 0 off the disk."""
    u, v, rho, on = G.PIX_G
    im = cv2.resize(img, (G.PIX_HALF, G.PIX_HALF), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vals = im[on]
    med = float(np.median(vals))
    bins = np.minimum((rho * G.PIX_NBINS).astype(np.int32), G.PIX_NBINS - 1)
    b_on = bins[on]
    order = np.argsort(b_on, kind="stable")
    edges = np.searchsorted(b_on[order], np.arange(G.PIX_NBINS + 1))
    prof = np.full(G.PIX_NBINS, med, np.float32)
    for b in range(G.PIX_NBINS):                # the limb-darkening profile, as pixel_gbm.pix_features
        s = vals[order[edges[b]:edges[b + 1]]]
        if s.size > 50:
            prof[b] = np.median(s)
    b2 = np.minimum((RHO2 * G.PIX_NBINS).astype(np.int32), G.PIX_NBINS - 1)
    return np.where(ON2, img.astype(np.float32) / 255.0 / np.maximum(prof[b2], 1e-3), 0.0).astype(np.float32)


# ---------------------------------------------------------------- the rotation columns

def rot_slots(files):
    """Per photograph, in each of the six slots (before/after x 0-24, 24-48, 48-84 h), the nearest other
    photograph of the same pool, with its gap in days (+ = later)."""
    import kfold as KF
    t = {f: KF.stamp(f) for f in files}
    out = {}
    for f in files:
        slots = []
        for d, lo, hi in ROT_SLOTS:
            best = None
            for g in files:
                dt = t[g] - t[f]
                if g != f and (dt < 0) == (d == "before") and lo < abs(dt) * 24 <= hi:
                    if best is None or abs(dt) < abs(best[1]):
                        best = (g, dt)
            slots.append(best)
        out[f] = slots
    return out


def rot_context(f):
    """One photograph's B0 and, per slot, the other photograph's contrast map (padded), B0 and gap. The other
    photographs are read as they are (not augmented)."""
    b0 = run.b0_deg(Path(f).stem)
    slots = []
    for s in ROT_SLOTS_OF.get(f, [None] * len(ROT_SLOTS)):
        if s is None:
            slots.append(None)
            continue
        g, dt = s
        img = cv2.imread(str(TR_DIR / g), cv2.IMREAD_GRAYSCALE)
        slots.append((np.pad(contrast(img), ROT_R), run.b0_deg(Path(g).stem), dt))
    return b0, slots


def rot_cols(ys, xs, ctx):
    """(n, NCOL - BASE_N): displacement per hour, and per slot: gap (h), where the point is in that photograph,
    and the contrast of every pixel within ROT_R of it there. 0 where the slot is empty or the point is hidden."""
    import kfold as KF
    b0, slots = ctx
    out = np.zeros((len(ys), NCOL - BASE_N), np.float32)
    pts = np.stack([ys, xs], axis=1)
    y1, x1, _ = KF.carry(pts, b0, b0, 1.0 / 24)
    out[:, 0], out[:, 1] = x1 - xs, y1 - ys
    j = 2
    for s in slots:
        if s is not None:
            cpB, b0B, dt = s
            yb, xb, vis = KF.carry(pts, b0, b0B, dt)
            yi = np.where(vis, np.rint(yb), 0).astype(np.int32)
            xi = np.where(vis, np.rint(xb), 0).astype(np.int32)
            out[:, j] = dt * 24.0
            out[:, j + 1], out[:, j + 2] = np.where(vis, xb, 0), np.where(vis, yb, 0)
            for k, (dy, dx) in enumerate(ROFFS):
                out[:, j + 3 + k] = np.where(vis, cpB[yi + ROT_R + dy, xi + ROT_R + dx], 0)
        j += 3 + len(ROFFS)
    return out


def build(F, cp, ys, xs, ctx=None):
    """(n, NCOL) rows for the pixels (ys, xs) at 2048."""
    X = np.empty((len(ys), NCOL), np.float32)
    if NB_ROT:
        X[:, BASE_N:] = rot_cols(ys, xs, ctx)
    X[:, :len(OLD)] = F[ys // 2, xs // 2]
    j = len(OLD)
    X[:, j], X[:, j + 1], X[:, j + 2] = xs, ys, cp[ys + RADIUS, xs + RADIUS]
    j += 3
    for dy, dx in OFFS[1:]:
        yy, xx = ys + RADIUS + dy, xs + RADIUS + dx
        inside = ON2P[yy, xx]
        X[:, j] = cp[yy, xx]
        X[:, j + 1] = np.where(inside, xs + dx, 0)
        X[:, j + 2] = np.where(inside, ys + dy, 0)
        j += 3
    return X


def sample(cnt, rng, n_rows):
    """Disk pixels an expert drew, an 8 px ring around them and the rest, each weighted back to its count."""
    pos2d = (cnt > 0) & ON2
    ring2d = cv2.dilate(pos2d.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~pos2d & ON2
    ys, xs, w = [], [], []
    for name, sel in (("pos", pos2d), ("ring", ring2d), ("far", ON2 & ~pos2d & ~ring2d)):
        cy, cx = np.nonzero(sel)
        if cy.size == 0:
            continue
        k = rng.choice(cy.size, min(n_rows[name], cy.size), replace=False)
        ys.append(cy[k]); xs.append(cx[k]); w.append(np.full(k.size, cy.size / k.size, np.float32))
    return np.concatenate(ys).astype(np.int32), np.concatenate(xs).astype(np.int32), np.concatenate(w)


def answers(cnt, n, ys, xs):
    m = cnt[ys, xs]
    return np.stack([m, np.full(m.size, n), m / n, COUNT_SCALE[np.minimum(m, 3)]], axis=1).astype(np.float32)


def seed_of(stem):
    return int(stem[:14]) % (2 ** 32)


def photo_rows(f, versions=VERSIONS, n_rows=None):
    """One training photograph: (X, answers, weight, version index) over its versions."""
    stem = Path(f).stem
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    rng = np.random.default_rng(seed_of(stem))
    ctx = rot_context(f) if NB_ROT else None
    out = []
    for vi, kind in enumerate(versions):
        F, cp = maps_of(augment(img, kind, rng), stem)
        ys, xs, w = sample(cnt, rng, n_rows or N_ROWS)
        out.append((build(F, cp, ys, xs, ctx), answers(cnt, n, ys, xs), w / len(versions), np.full(len(w), vi, np.int8)))
    return [np.concatenate([o[i] for o in out]) for i in range(4)]


def rows_job(f):
    return photo_rows(f)


# ---------------------------------------------------------------- forests, maps, scores

def fit(X, t, w):
    """Regression on the soft target, or (cls) a classifier on 0/1 labels with their weights."""
    if NB_LEARNER == "cls":
        if NB_TUNE == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(**{**XGB, "n_estimators": 5 if SMOKE else XGB["n_estimators"]}, n_jobs=-1,
                                 random_state=0).fit(X, t, sample_weight=w)
        from sklearn.ensemble import RandomForestClassifier
        params = {**RF_TUNED, "n_estimators": 5 if SMOKE else RF_TUNED["n_estimators"]} if NB_TUNE == "rf" else RF
        return RandomForestClassifier(**params, n_jobs=-1, random_state=0).fit(X, t, sample_weight=w)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**RF, n_jobs=-1, random_state=0).fit(X, t, sample_weight=w)


def prob(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)


def mdi(m, names):
    """The forest's own (impurity) importance per column, and summed over the column groups."""
    v = m.feature_importances_
    out = {"per_column": {n: round(float(x), 7) for n, x in zip(names, v)}}
    if len(names) == NCOL:
        j = len(OLD)
        out["groups"] = {"35 earlier columns": round(float(v[:j].sum()), 5),
                         "pixel x, y": round(float(v[j:j + 2].sum()), 5),
                         "own contrast": round(float(v[j + 2]), 5),
                         "neighbours' contrast (440)": round(float(v[j + 3:BASE_N:3].sum()), 5),
                         "neighbours' positions (880)": round(float(v[j + 4:BASE_N:3].sum() + v[j + 5:BASE_N:3].sum()), 5)}
        if NB_ROT:
            out["groups"]["rotation: vx, vy"] = round(float(v[BASE_N:BASE_N + 2].sum()), 5)
            k = 3 + len(ROFFS)
            for i, (d, lo, hi) in enumerate(ROT_SLOTS):
                a = BASE_N + 2 + i * k
                out["groups"][f"other photograph, {d} {lo}-{hi} h"] = round(float(v[a:a + k].sum()), 5)
    return out


def val_job(f):
    stem = Path(f).stem
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    F, cp = maps_of(img, stem)
    ctx = rot_context(f) if NB_ROT else None
    ys, xs = np.nonzero(ON2)
    ys, xs = ys.astype(np.int32), xs.astype(np.int32)
    full = {a: np.zeros((run.FULL, run.FULL), np.float32) for a in MODELS}
    for s in range(0, len(ys), CHUNK):
        X = build(F, cp, ys[s:s + CHUNK], xs[s:s + CHUNK], ctx)
        for a, m in MODELS.items():
            full[a][ys[s:s + CHUNK], xs[s:s + CHUNK]] = prob(m, np.ascontiguousarray(X[:, ARM_COLS[a]]))
        del X
    for a, p in full.items():
        (MAPS / a).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(MAPS / a / f"{stem}.png"), np.round(np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8))
    vy, vx, w = sample(cnt, np.random.default_rng(seed_of(stem) + 1), N_VAL_ROWS)
    return build(F, cp, vy, vx, ctx), answers(cnt, n, vy, vx), w, {a: p[vy, vx] for a, p in full.items()}


def auc(t, w, s):
    from sklearn.metrics import roc_auc_score
    soft = (t > 0) & (t < 1)
    y = np.concatenate([(t > 0).astype(np.int8), np.zeros(int(soft.sum()), np.int8)])
    ww = np.concatenate([np.where(t > 0, w * t, w), w[soft] * (1.0 - t[soft])])
    return round(float(roc_auc_score(y, np.concatenate([s, s[soft]]), sample_weight=ww)), 4)


def importance(model, X, t, w, rng):
    """Increase of the weighted squared error (target: share) when a group of columns is shuffled."""
    def err(p):
        return float((w * (p - t) ** 2).sum() / w.sum())
    base = err(prob(model, X))
    dist = np.hypot(DY, DX)
    c_idx = [len(OLD) + 2] + [len(OLD) + 3 + 3 * k for k in range(len(OFFS) - 1)]
    groups = {"[35 earlier columns]": list(range(len(OLD))), "[pixel x, y]": [len(OLD), len(OLD) + 1],
              "[neighbours' contrast, all 441]": c_idx,
              "[neighbours' positions, 880]": [len(OLD) + 3 + 3 * k + q for k in range(len(OFFS) - 1) for q in (1, 2)]}
    for lo, hi in ((0, 0), (0.5, 1.5), (1.5, 3), (3, 6), (6, 9), (9, 12.01)):
        groups[f"[contrast, neighbours {lo}-{hi:g} px]"] = [c_idx[k] for k in range(len(OFFS)) if lo <= dist[k] <= hi]
    if NB_ROT:
        k = 3 + len(ROFFS)
        groups["[rotation: vx, vy]"] = [BASE_N, BASE_N + 1]
        groups["[other photographs, all slots]"] = list(range(BASE_N + 2, NCOL))
        for i, (d, lo, hi) in enumerate(ROT_SLOTS):
            a = BASE_N + 2 + i * k
            groups[f"[other photograph, {d} {lo}-{hi} h]"] = list(range(a, a + k))
    out = {}
    for name, cols in groups.items():
        Xs = X.copy()
        perm = rng.permutation(len(Xs))
        Xs[:, cols] = X[np.ix_(perm, cols)]
        out[name] = round(err(prob(model, Xs)) - base, 6)
        del Xs
    return dict(base_mse=round(base, 6), increase=dict(sorted(out.items(), key=lambda kv: -kv[1])))


def load_sets():
    global TR_DIR
    ann, TR_DIR, _ = run.find_data()
    coco = json.load(open(ann))
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    for sid in sorted(fname):
        SETS.setdefault(fname[sid], []).append(by_id[sid])
    return fname


def main(procs):
    t_start = time.time()

    def log(*a):
        print(f"[{(time.time() - t_start) / 60:6.1f} min]", *a, flush=True)

    fname = load_sets()
    train_files, val_files = run.photograph_split(fname, 0)
    if NB_ROT:      # the other photographs come from the same pool: training from training, validation from
        for pool in (train_files, val_files):           # validation (the test photographs will see only test ones)
            ROT_SLOTS_OF.update(rot_slots(pool))
    if SMOKE:                       # with the rotation, photographs whose slots are filled, so the smoke tries them
        has = (lambda f: any(ROT_SLOTS_OF.get(f, []))) if NB_ROT else (lambda f: True)
        train_files, val_files = [f for f in train_files if has(f)][:4], [f for f in val_files if has(f)][:2]
    rot_fill = {f"{d}{lo}_{hi}h": {"train": round(float(np.mean([ROT_SLOTS_OF[f][i] is not None for f in train_files])), 3),
                                   "val": round(float(np.mean([ROT_SLOTS_OF[f][i] is not None for f in val_files])), 3)}
                for i, (d, lo, hi) in enumerate(ROT_SLOTS)} if NB_ROT else None
    res = dict(learner=NB_LEARNER, arms=ARMS, rotation=NB_ROT, rotation_slot_fill=rot_fill, rot_radius_px=ROT_R, columns=NCOL, column_groups={"earlier": len(OLD), "pixel x, y, own contrast": 3,
                                            "per neighbour contrast, x, y": 3 * (len(OFFS) - 1)},
               neighbours=len(OFFS) - 1, radius_px_2048=RADIUS, versions=VERSIONS, rows_per_version=N_ROWS,
               rf=RF, train=len(train_files), val=len(val_files), smoke=SMOKE)
    if MAPS.exists():
        shutil.rmtree(MAPS)

    # 1. the training table, filled in place (two copies of 9 GB would not fit)
    n_max = len(train_files) * len(VERSIONS) * sum(N_ROWS.values())
    n_dup = len(train_files) * len(VERSIONS) * N_ROWS["pos"] if NB_LEARNER == "cls" else 0   # room for 0-copies
    X = np.empty((n_max + n_dup, NCOL), np.float32)
    A = np.empty((n_max, len(ANSWER)), np.float32)
    W = np.empty(n_max, np.float32)
    V = np.empty(n_max, np.int8)
    n = 0
    pool = mp.Pool(procs) if procs > 1 else None
    for i, (x, a, w, v) in enumerate((pool.imap if pool else map)(rows_job, train_files)):
        X[n:n + len(w)], A[n:n + len(w)], W[n:n + len(w)], V[n:n + len(w)] = x, a, w, v
        n += len(w)
        if i % 100 == 0:
            log(f"rows {i + 1}/{len(train_files)}: {n}")
    if pool:
        pool.close(); pool.join()
    A, W, V = A[:n], W[:n], V[:n]
    res["table"] = dict(rows=n, columns=NCOL, GB=round(n * NCOL * 4 / 2 ** 30, 2),
                        rows_by_version={k: int((V == i).sum()) for i, k in enumerate(VERSIONS)},
                        filament_rows=int((A[:, 0] > 0).sum()))
    log("table", res["table"])

    # 2. the forests
    if NB_LEARNER == "cls":         # a soft target t: a 1-row of weight w t and a 0-row of weight w (1 - t), as rf0
        share = A[:, 2]
        soft = np.flatnonzero((share > 0) & (share < 1))
        for s in range(0, len(soft), 100000):
            idx = soft[s:s + 100000]
            X[n + s:n + s + len(idx)] = X[idx]
        m = n + len(soft)
        y = np.concatenate([(share > 0).astype(np.int8), np.zeros(len(soft), np.int8)])
        w = np.concatenate([np.where(share > 0, W * share, W), W[soft] * (1.0 - share[soft])]).astype(np.float32)
        targets = {"share": (y, w)}
        res["table"]["classifier_rows"] = int(m)
    else:
        m = n
        targets = {"share": (A[:, 2], W), "count": (A[:, 3], W)}
    Xt = X[:m]
    res["fit_minutes"], res["mdi"] = {}, {}
    for arm in ARMS:
        t0 = time.time()
        t, w = targets[ARM_TARGET[arm]]
        cols = ARM_COLS[arm]
        whole = isinstance(cols, slice) and cols.start == 0 and cols.stop == NCOL
        MODELS[arm] = fit(Xt if whole else np.ascontiguousarray(Xt[:, cols]), t, w)
        res["fit_minutes"][arm] = round((time.time() - t0) / 60, 1)
        res["mdi"][arm] = mdi(MODELS[arm], COLS[cols] if isinstance(cols, slice) else [COLS[i] for i in cols])
        log("fitted", arm, res["fit_minutes"][arm], "min", res["mdi"][arm].get("groups", ""))
    del X, Xt, A, W, V, targets
    gc.collect()

    # 3. maps of the validation photographs
    for m in MODELS.values():
        m.n_jobs = 1
        if hasattr(m, "get_booster"):                 # xgboost reads its own thread count, not the attribute
            m.get_booster().set_param({"nthread": 1})
    pool = mp.Pool(procs, initializer=G.pix_one_thread) if procs > 1 else None
    parts = []
    for i, p in enumerate((pool.imap if pool else map)(val_job, val_files)):
        parts.append(p)
        if i % 20 == 0:
            log(f"validation {i + 1}/{len(val_files)}")
    if pool:
        pool.close(); pool.join()
    VX = np.concatenate([p[0] for p in parts])
    VA = np.concatenate([p[1] for p in parts])
    VW = np.concatenate([p[2] for p in parts])
    VP = {a: np.concatenate([p[3][a] for p in parts]) for a in MODELS}
    del parts
    gc.collect()
    res["row_auc_vs_share"] = {a: auc(VA[:, 2], VW, VP[a]) for a in MODELS}
    log("row AUC on fold 0", res["row_auc_vs_share"])
    rng = np.random.default_rng(0)
    sub = rng.choice(len(VW), min(150000, len(VW)), replace=False)
    if MAIN_ARM is not None:                  # the group importance reads the full table's column layout
        MODELS[MAIN_ARM].n_jobs = -1
        res["importance_" + MAIN_ARM] = importance(MODELS[MAIN_ARM], VX[sub], VA[sub, 2], VW[sub], rng)
        log("importance", res["importance_" + MAIN_ARM])

    tar = WORK / f"dump_{TAG}.tar"
    with tarfile.open(tar, "w") as tf:
        for arm in MODELS:
            tf.add(str(MAPS / arm), arcname=arm)
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(WORK / f"pixel_{TAG}.json", "w"), indent=1)
    log("done", tar)
    return res


def kaggle_main():
    main(os.cpu_count() or 4)


# ---------------------------------------------------------------- CLI
# Everything below is local only.

def head():
    """The table of two training photographs, every version: data.head(), data.info(), a CSV of the head."""
    import pandas as pd
    fname = load_sets()
    train_files, _ = run.photograph_split(fname, 0)
    pick = [f for f in train_files if len(SETS[f]) == 3][:2]
    parts = [photo_rows(f) + [Path(f).stem] for f in pick]
    df = pd.DataFrame(np.concatenate([p[0] for p in parts]), columns=COLS)
    ans = pd.DataFrame(np.concatenate([p[1] for p in parts]), columns=ANSWER)
    meta = pd.DataFrame({"photo": np.concatenate([[p[4]] * len(p[2]) for p in parts]),
                         "version": [VERSIONS[i] for i in np.concatenate([p[3] for p in parts])],
                         "weight": np.concatenate([p[2] for p in parts])})
    data = pd.concat([meta, df, ans], axis=1)
    out = HERE / "out"
    data.head(20).to_csv(out / "pixel_nb_head.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 16)
    print(data.head())
    print()
    data.info(verbose=False)
    print("\nrows by version:", data["version"].value_counts().to_dict())
    print("experts_marked:", data["experts_marked"].value_counts().sort_index().to_dict())
    print("first columns:", list(data.columns[:45]))
    print("last columns:", list(data.columns[-10:]))
    print("wrote", out / "pixel_nb_head.csv")


def fig():
    """figures/pixel_nb_table.png: three rows of the table drawn back as their 12 px neighbourhoods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fname = load_sets()
    train_files, _ = run.photograph_split(fname, 0)
    f = [f for f in train_files if len(SETS[f]) == 3][0]
    stem = Path(f).stem
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    rng = np.random.default_rng(1)
    F, cp = maps_of(img, stem)
    ring = cv2.dilate((cnt > 0).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & (cnt == 0) & ON2
    picks = []
    for name, sel in (("3 of 3 experts drew it", (cnt == 3) & ON2), ("background, 8 px from a filament", ring),
                      ("background, far from filaments", ON2 & ~cv2.dilate((cnt > 0).astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool))):
        yy, xx = np.nonzero(sel)
        k = rng.integers(len(yy))
        picks.append((name, int(yy[k]), int(xx[k])))
    ys, xs = np.array([p[1] for p in picks], np.int32), np.array([p[2] for p in picks], np.int32)
    X = build(F, cp, ys, xs)
    c_idx = [len(OLD) + 2] + [len(OLD) + 3 + 3 * k for k in range(len(OFFS) - 1)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
    for j, (name, y, x) in enumerate(picks):
        g = np.full((2 * RADIUS + 1, 2 * RADIUS + 1), np.nan)
        g[DY + RADIUS, DX + RADIUS] = X[j, c_idx]
        ax = axs[0, j]
        lo, hi = max(y - 60, 0), max(x - 60, 0)
        ax.imshow(img[lo:y + 61, hi:x + 61], cmap="gray")
        ax.contour((cnt[lo:y + 61, hi:x + 61] > 0).astype(float), [0.5], colors="#E69F00", linewidths=1)
        ax.add_patch(plt.Circle((x - hi, y - lo), RADIUS, fill=False, color="#56B4E9", lw=1.5))
        ax.set_title(f"{name}\npixel x {x}, y {y}; orange = expert outline", fontsize=9.5, loc="left")
        ax.axis("off")
        ax = axs[1, j]
        im = ax.imshow(g, cmap="gray", vmin=0.7, vmax=1.15, extent=(-RADIUS - 0.5, RADIUS + 0.5, RADIUS + 0.5, -RADIUS - 0.5))
        ax.plot(0, 0, "+", color="#D55E00", ms=10)
        ax.set_title(f"its 441 contrast columns (c_dx.._dy..)\nown {X[j, len(OLD) + 2]:.2f}, mean of the 8 adjacent "
                     f"{np.mean([X[j, c_idx[k]] for k in range(1, 9)]):.2f}", fontsize=9.5, loc="left")
        ax.set_xlabel("dx, px at 2048")
        if j == 0:
            ax.set_ylabel("dy, px at 2048")
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, label="contrast = brightness / quiet Sun at the same mu (1 = quiet Sun)")
    fig.suptitle(f"The user's table: one row per pixel; its own and its neighbours' contrast within 12 px, with their "
                 f"positions ({stem}, {n} experts)", fontsize=11, x=0.01, ha="left")
    out = HERE.parents[1] / "figures" / "pixel_nb_table.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    if "--fig" in sys.argv:
        fig()
    elif "--head" in sys.argv:
        head()
    else:
        main(1)
