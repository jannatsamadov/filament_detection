"""Time, the simple way: carry a neighbour photograph's map forward by the Sun's known rotation (11-16).

No learning. A photograph taken dt days before shows the same filaments, each moved in longitude by
omega(phi) * dt; mapping it pixel by pixel onto this photograph's sphere (each with its own B0) puts them where
they should be now.

(A) The rotation rate the data itself shows: the experts' union mask of photograph B, carried forward to the time
    of photograph A at a trial equatorial rate (Snodgrass's latitude shape), overlapped with A's experts' mask.
    The Dice over all pairs within 84 h peaks at the filaments' synodic rate if the geometry (north up, the
    longitude sign, B0) is right; a flat curve would say it is not.
(B) The R50 U-Net's out-of-fold D4 maps (707 photographs, five folds): each photograph's map averaged with its
    neighbours' maps carried forward, neighbours from the same fold only (the same model saw neither, and at
    test the neighbours are other test photographs only, rule 1). PQ with the fixed rules, on all 707 and on
    the photographs that have a neighbour; controls: the same averaging without rotation.
    python kernels/aug_unet/time_derot.py      -> out/time_derot.json
"""
import collections, json, pickle, sys, tarfile, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from run import np, cv2  # noqa: E402

KER = HERE.parent
R50 = [KER / "unet_r2/out_kaggle/dump_r50.tar", KER / "unet_r5/out_kaggle/dump_r50f1.tar",
       KER / "unet_r5/out_kaggle/dump_r50f2.tar", KER / "unet_r6/out_kaggle/dump_r50f3.tar",
       KER / "unet_r6/out_kaggle/dump_r50f4.tar"]
MAX_DT_H = 84.0
SHAPE = (-2.39, -1.78)          # Snodgrass's latitude terms, deg/day
EARTH = 0.9856                  # deg/day: sidereal -> synodic
N_A = 512                       # resolution of the rate measurement (1 px = 1.8 Mm at disk centre)
OUT = HERE / "out" / "time_derot.json"


def hours(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").timestamp() / 3600.0


def sphere(b0, n):
    """Per pixel of an n x n frame: sin(latitude), longitude from the CM (rad, + west), mu, on-disk."""
    c = ((np.arange(n) + 0.5) * (run.FULL / n) - 0.5 - run.CX) / run.R_DISK
    dx, dy = np.meshgrid(c, -c)
    rho2 = dx * dx + dy * dy
    z = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
    b = np.radians(b0)
    sl = np.clip(dy * np.cos(b) + z * np.sin(b), -1.0, 1.0)
    lon = np.arctan2(dx, z * np.cos(b) - dy * np.sin(b))
    return sl, lon, z, rho2 < 1.0


def carry(src, b0_src, b0_dst, dt_days, a_rate, n):
    """src (n x n float32) seen at the destination's time: dt_days = t_dst - t_src. Returns map, valid."""
    sl, lon, z, on = sphere(b0_dst, n)
    w = np.radians(a_rate + SHAPE[0] * sl ** 2 + SHAPE[1] * sl ** 4) if a_rate is not None else 0.0
    lon_s = lon - w * dt_days
    cl = np.sqrt(1.0 - sl * sl)
    b = np.radians(b0_src)
    dxs = cl * np.sin(lon_s)
    dys = sl * np.cos(b) - cl * np.cos(lon_s) * np.sin(b)
    zs = sl * np.sin(b) + cl * np.cos(lon_s) * np.cos(b)
    col = ((run.CX + dxs * run.R_DISK) + 0.5) * n / run.FULL - 0.5
    row = ((run.CY - dys * run.R_DISK) + 0.5) * n / run.FULL - 0.5
    out = cv2.remap(src, col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    return out, on & (z > 0.15) & (zs > 0.15)


def folds():
    """Each fold's validation photographs and every annotation set's filaments (as run.prepare keeps them), straight
    from the annotation file: the local fold caches hold only a few photographs each."""
    cache = HERE / "out" / "time_derot_gt.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    ann_path, _, _ = run.find_data()
    coco = json.load(open(ann_path))
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    out = []
    for k in range(5):
        _, val_files = run.photograph_split(fname, k)
        index = {f: i for i, f in enumerate(val_files)}
        by = collections.defaultdict(list)
        for sid in sorted(fname):
            if fname[sid] not in index:
                continue
            crops = []
            for a in by_id[sid]:
                m = run.ann_mask(a)
                ys, xs = np.nonzero(m)
                if ys.size:
                    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    crops.append((int(y0), int(x0), m[y0:y1, x0:x1].copy()))
            by[index[fname[sid]]].append(crops)
        out.append(dict(stems=[Path(f).stem for f in val_files], sets=dict(by)))
        print(f"  fold {k}: {len(val_files)} photographs, {sum(len(v) for v in by.values())} annotation sets", flush=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def union(sets, n):
    m = np.zeros((run.FULL, run.FULL), np.uint8)
    for crops in sets:
        for y0, x0, c in crops:
            m[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c] = 1
    return cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) if n != run.FULL else m.astype(np.float32)


def pairs_of(stems, max_h=MAX_DT_H):
    t = np.array([hours(s) for s in stems])
    out = []
    for i in range(len(stems)):
        for j in range(len(stems)):
            if i != j and 1.0 <= abs(t[i] - t[j]) <= max_h:
                out.append((i, j, (t[i] - t[j]) / 24.0))
    return out


def part_a(fs):
    stems = [s for f in fs for s in f["stems"]]
    masks = [union(f["sets"].get(i, []), N_A) for f in fs for i in range(len(f["stems"]))]
    pr = pairs_of(stems)
    print(f"  (A) {len(pr)} ordered pairs within {MAX_DT_H:.0f} h among {len(stems)} photographs", flush=True)
    rates = [-13.0, 0.0, 6.0, 10.0, 12.0, 13.0, 13.5, 14.0, 15.0, 16.0, 18.0]
    res = []
    for a in rates:
        inter = tot = 0.0
        for i, j, dt in pr:
            m, v = carry(masks[j], run.b0_deg(stems[j]), run.b0_deg(stems[i]), dt, a, N_A)
            mi = masks[i] * v
            mj = (m > 0.5) * v
            inter += float((mi * mj).sum())
            tot += float(mi.sum() + mj.sum())
        res.append(dict(rate=a, dice=2 * inter / max(tot, 1.0)))
        print(f"  (A) equatorial rate {a:6.1f} deg/day (synodic): Dice {res[-1]['dice']:.4f}", flush=True)
    return dict(pairs=len(pr), photographs=len(stems), curve=res)


def load_maps(k, stems):
    out = {}
    with tarfile.open(R50[k]) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == "val_d4" and Path(m.name).stem in stems:
                out[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8),
                                                      cv2.IMREAD_GRAYSCALE)
    assert len(out) == len(stems), (k, len(out), len(stems))      # the dump's fold is this fold
    return out


GRID = [(t, k, a) for t in (0.5, 0.6, 0.7) for k in (3, 7) for a in (250, 350)]


def counts(p_u8, sets):
    c = {}
    for t, k in {(t, k) for t, k, _ in GRID}:
        lab, area = run.components(p_u8, t, k)
        for a in (250, 350):
            keep = area >= a
            keep[0] = False
            c[(t, k, a)] = sum(np.array(run.pq_counts(cr, lab, area, keep), float) for cr in sets)
    return c


def part_b(fs, a_rate):
    # the control: the same averaging, the neighbour put on this photograph's sphere but not rotated
    arms = {"plain": None, "w0.5 no rotation": (0.5, 0.0), "w0.5 rotation": (0.5, a_rate),
            "w0.25 rotation": (0.25, a_rate), "w1 rotation": (1.0, a_rate)}
    tot = {arm: collections.defaultdict(lambda: np.zeros(4)) for arm in arms}
    sub = {arm: collections.defaultdict(lambda: np.zeros(4)) for arm in arms}
    n_nb = 0
    t0 = time.time()
    for k, f in enumerate(fs):
        maps = load_maps(k, set(f["stems"]))
        t = np.array([hours(s) for s in f["stems"]])
        for i, s in enumerate(f["stems"]):
            near = [j for j in range(len(f["stems"])) if j != i and 1.0 <= abs(t[i] - t[j]) <= MAX_DT_H]
            n_nb += bool(near)
            p = maps[s].astype(np.float32)
            carried, base = {}, None
            for arm, spec in arms.items():
                if spec is None or not near:
                    base = counts(maps[s], f["sets"].get(i, [])) if base is None else base
                    c = base
                else:
                    w, rate = spec
                    acc, wt = p.copy(), np.ones_like(p)
                    for j in near:
                        if (j, rate) not in carried:
                            carried[(j, rate)] = carry(maps[f["stems"][j]].astype(np.float32),
                                                       run.b0_deg(f["stems"][j]), run.b0_deg(s),
                                                       (t[i] - t[j]) / 24.0, rate, run.FULL)
                        m, v = carried[(j, rate)]
                        acc += w * m * v
                        wt += w * v
                    c = counts(np.round(acc / wt).astype(np.uint8), f["sets"].get(i, []))
                for g, v in c.items():
                    tot[arm][g] += v
                    if near:
                        sub[arm][g] += v
        print(f"  (B) fold {k}: {len(f['stems'])} photographs, {time.time() - t0:.0f}s", flush=True)

    def best(d):
        rows = [dict(run.summarise(v), thresh=g[0], merge=g[1], min_area=g[2]) for g, v in d.items()]
        return max(rows, key=lambda r: r["PQ"])
    out = {}
    for arm in arms:
        b_all = best(tot[arm])
        g = (b_all["thresh"], b_all["merge"], b_all["min_area"])
        out[arm] = dict(all_707=b_all, with_neighbour=dict(run.summarise(sub[arm][g]), at=g),
                        with_neighbour_best=best(sub[arm]))
        print(f"  (B) {arm:18s} all 707 PQ {b_all['PQ']:.4f} at {g}; the {n_nb} with a neighbour "
              f"{out[arm]['with_neighbour']['PQ']:.4f} (their own best {out[arm]['with_neighbour_best']['PQ']:.4f})",
              flush=True)
    return dict(photographs_with_neighbour=n_nb, arms=out)


def main():
    fs = folds()
    res = dict(A=part_a(fs))
    curve = [r for r in res["A"]["curve"] if r["rate"] is not None]
    a_best = max(curve, key=lambda r: r["dice"])["rate"]
    res["A"]["peak_rate"] = a_best
    res["A"]["snodgrass_synodic"] = round(14.71 - EARTH, 2)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    res["B"] = part_b(fs, a_best)
    json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
