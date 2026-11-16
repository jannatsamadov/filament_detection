"""FilNet's own filament head: the keep decision inside the model (11-16, the user's request).

The 21-number rule reads hand-made numbers of each candidate. Here the decision is made from the network's own
features: FilNet's trunk is frozen, its feature maps at every size are averaged inside each candidate and in a
ring around it, and a small head (an MLP) learns each candidate's mean matched IoU from them. The trunk and the
head are then one model: frame -> channels -> trunk -> map -> candidates -> head -> kept filaments.
The trunk's features belong to one fold model (two trained networks do not share their channels), so a head is
fitted per fold model, on that model's own unseen photographs: four fifths of the fold's validation photographs
for the remaining fifth. The rule's 21 numbers mean the same for every model, so the rule can also be fitted on
the other folds' rows (what the submitted file does). Every score is of a photograph the scorer never saw:
  rule         trees on the 21 numbers, fitted on the other folds          (the submitted file's kind)
  rule-in      trees on the 21 numbers, fitted inside the fold             (the head's control: same rows)
  linear       ridge regression on the trunk's features, inside the fold
  head         MLP on the trunk's features, inside the fold                (the selector inside the model)
  head+21      MLP on the features and the 21 numbers, inside the fold
  trees+21     trees on the features and the 21 numbers, inside the fold
  rule+linear, rule+head   trees on the 21 numbers and the head's score as a 22nd, fitted on the other folds
    python kernels/aug_unet/filnet_head.py extract <fold>   # GPU: that fold's rows with features
    python kernels/aug_unet/filnet_head.py test             # GPU: the test candidates with every fold model's features
    python kernels/aug_unet/filnet_head.py [kind]           # the table -> out/filnet_head.json; with a kind, its test CSV
"""
import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")
os.environ.setdefault("OMP_NUM_THREADS", "3")
import json, pickle, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from run import torch, cv2, np, nn  # noqa: E402
import filnet as FN  # noqa: E402
import solar_unet as S  # noqa: E402  -- pooled(), the rule's protocol
import candidate_pool as C  # noqa: E402
import qwen_seg as Q  # noqa: E402
import time_derot as TD  # noqa: E402

K = HERE.parent
OUT = HERE / "out"
SRC = {0: ("filnet", "filnet_phys"), 1: ("filnet_b", "filnet_phys_f1"), 2: ("filnet_b", "filnet_phys_f2"),
       3: ("filnet_c", "filnet_phys_f3"), 4: ("filnet_c", "filnet_phys_f4")}
CUTS = np.round(np.arange(0.02, 0.6, 0.01), 2)


def have(k):
    d, n = SRC[k]
    return (K / d / "out_kaggle" / f"{n}.pt").exists() and (K / d / "out_kaggle" / f"dump_{n}.tar").exists()


def load_model(k, device):
    d, n = SRC[k]
    sd = torch.load(K / d / "out_kaggle" / f"{n}.pt", map_location="cpu")
    net = FN.FilNet(FN.IN_CH["phys"])
    net.load_state_dict({a: (v.float() if v.is_floating_point() else v) for a, v in sd.items()})
    return net.to(device).eval().to(memory_format=torch.channels_last)


def tar_maps(k, sub):
    d, n = SRC[k]
    out = {}
    with tarfile.open(K / d / "out_kaggle" / f"dump_{n}.tar") as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == sub:
                out[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return out


@torch.no_grad()
def pooled(net, im_u8, stem, cands, device, gov):
    """(n, D): the trunk's features inside each candidate at every size, and in a ring at 1/8."""
    if not cands:
        return np.zeros((0, sum(FN.WIDTHS) + FN.WIDTHS[3]), np.float16)
    with gov.step():
        x = run.to_input(im_u8)[None, None].to(device)
        geo = run.geo_channels(torch.tensor([[1.0, 0.0, 0.0, 1.0, run.b0_deg(stem)]]), device)
        inp = FN.front(x, geo, "phys").contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, fm = net(inp, feats=True)
        z = S.pooled(fm, cands)
        del fm
        if device.type == "cuda":
            torch.cuda.synchronize()
    return z


def extract(k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gov = Q.governor()
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    net = load_model(k, device)
    f = TD.folds()[k]
    path_of = {p.stem: p for p in run.find_data()[1].iterdir()}
    maps = tar_maps(k, "val_d4")
    rows, t0 = [], time.time()
    for i, s in enumerate(f["stems"]):
        img = cv2.imread(str(path_of[s]), cv2.IMREAD_GRAYSCALE)
        sets = f["sets"].get(i, [])
        cands, X = C.candidates(maps[s], img, run.b0_deg(s))
        one = np.array([C.ONE_SETTING in c["from"] for c in cands], bool)
        keep = [c for c, o in zip(cands, one) if o]              # the head is measured on the one setting
        Y, _ = C.targets(keep, sets)
        rows.append(dict(stem=s, fold=k, X=X[one], Z=pooled(net, img, s, keep, device, gov), Y=Y.astype(np.float32),
                         n_gt=[len(x) for x in sets], conflict=C.conflicts(keep), one=np.ones(len(keep), bool)))
        if (i + 1) % 20 == 0:
            print(f"fold {k}: {i + 1}/{len(f['stems'])} photographs, {time.time() - t0:.0f}s", flush=True)
    pickle.dump(rows, open(OUT / f"filnet_head_rows_f{k}.pkl", "wb"))
    print(f"fold {k}: {sum(len(r['X']) for r in rows)} candidates on {len(rows)} photographs, {time.time() - t0:.0f}s", flush=True)


def extract_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gov = Q.governor()
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    folds = [k for k in SRC if have(k)]
    te_dir = run.find_data()[2]
    files = sorted(p.name for p in te_dir.glob("*.jpeg"))
    tm = [tar_maps(k, "test_d4") for k in folds]
    rows = []
    for name in files:
        s = Path(name).stem
        p = np.mean([m[s] for m in tm], 0).round().astype(np.uint8)
        img = cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE)
        cands, X = C.candidates(p, img, run.b0_deg(s))
        one = np.array([C.ONE_SETTING in c["from"] for c in cands], bool)
        keep = [c for c, o in zip(cands, one) if o]
        rows.append(dict(stem=s, X=X[one], rle=[C.cand_rle(c) for c in keep], cands=keep, Z={}))
    for k in folds:                                           # one model on the card at a time
        net = load_model(k, device)
        t0 = time.time()
        for r, name in zip(rows, files):
            r["Z"][k] = pooled(net, cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE), r["stem"], r["cands"], device, gov)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"test features of fold {k}'s model: {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        del r["cands"]
    pickle.dump(dict(folds=folds, rows=rows), open(OUT / "filnet_head_test.pkl", "wb"))


# ---------------------------------------------------------------- the head

class Head:
    """A small MLP on standardised features; three seeds averaged."""

    def __init__(self, X, y, seeds=(0, 1, 2), epochs=60):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32))
        self.nets = []
        for seed in seeds:
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.SiLU(),
                                nn.Linear(64, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * ((len(Xn) + 255) // 256))
            for _ in range(epochs):
                net.train()
                perm = torch.randperm(len(Xn))
                for i in range(0, len(Xn), 256):
                    b = perm[i:i + 256]
                    loss = nn.functional.mse_loss(net(Xn[b])[:, 0], yt[b])
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            self.nets.append(net.eval())

    @torch.no_grad()
    def predict(self, X):
        if not len(X):
            return np.zeros(0)
        Xn = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        return np.mean([n(Xn)[:, 0].numpy() for n in self.nets], 0)


class Linear:
    """A linear head: ridge regression on standardised features, its strength chosen by leave-one-out."""

    def __init__(self, X, y):
        from sklearn.linear_model import RidgeCV
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit((X - self.mu) / self.sd, y)

    def predict(self, X):
        return self.m.predict((X - self.mu) / self.sd) if len(X) else np.zeros(0)


# name -> (what reads the candidate, on which numbers, fitted where)
#   "other folds": on the other folds' rows, possible only for numbers that mean the same for every fold model
#   "same fold":   on four fifths of the same fold's photographs for the remaining fifth -- the trunk's features
#                  belong to one fold model, so a head can only be fitted on that model's own unseen photographs
KINDS = {
    "rule":        ("trees", "21 numbers", "other folds"),
    "rule-in":     ("trees", "21 numbers", "same fold"),
    "linear":      ("linear head", "the trunk's features", "same fold"),
    "head":        ("MLP head", "the trunk's features", "same fold"),
    "head+21":     ("MLP head", "the trunk's features + 21 numbers", "same fold"),
    "trees+21":    ("trees", "the trunk's features + 21 numbers", "same fold"),
    "rule+linear": ("trees", "21 numbers + the linear head's score", "other folds"),
    "rule+head":   ("trees", "21 numbers + the MLP head's score", "other folds"),
}


def design(f, kind):
    z = f["Z"].astype(np.float32)
    if kind in ("rule", "rule-in"):
        return f["X"]
    if kind in ("head", "linear"):
        return z
    if kind == "head+21":
        return np.hstack([z, np.nan_to_num(f["X"])])
    if kind == "trees+21":
        return np.hstack([f["X"], z])
    return np.hstack([f["X"], np.asarray(f["H"][kind.split("+")[1]], np.float32)[:, None]])   # rule+head, rule+linear


def fit(frames, kind):
    X = np.vstack([design(f, kind) for f in frames if len(f["X"])])
    y = np.concatenate([f["Y"].mean(1) if f["Y"].shape[1] else np.zeros(len(f["X"])) for f in frames if len(f["X"])])
    if KINDS[kind][0] == "trees":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
                                             random_state=0).fit(X, y)
    return Linear(X, y) if kind == "linear" else Head(X, y)


def parts(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(idx[j::k]) for j in range(k)]


def oof(frames, folds, kind):
    """Every photograph's candidates scored by a model that never saw that photograph's outlines."""
    scores = [None] * len(frames)
    for k in folds:
        idx = [n for n, f in enumerate(frames) if f["fold"] == k]
        if KINDS[kind][2] == "other folds":
            groups = [(idx, [n for n, f in enumerate(frames) if f["fold"] != k])]
        else:
            groups = []
            for part in parts(len(idx)):
                te = [idx[j] for j in part]
                groups.append((te, [n for n in idx if n not in set(te)]))
        for te, tr in groups:
            m = fit([frames[n] for n in tr], kind)
            for n in te:
                scores[n] = m.predict(design(frames[n], kind)) if len(frames[n]["X"]) else np.zeros(0)
    return scores


def per_photo(frames, scores, cut):
    out = np.zeros((len(frames), 4))
    for n, (f, s) in enumerate(zip(frames, scores)):
        out[n] = S.tally([f], [s], cut, True)
    return out


def pq_of(c):
    tp, fp, fn, iou = c.sum(axis=0)
    return iou / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)


def boot(a, b, n=2000):
    rng = np.random.default_rng(0)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a))
        d.append(pq_of(a[s]) - pq_of(b[s]))
    return [round(float(pq_of(a) - pq_of(b)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    folds = [k for k in SRC if (OUT / f"filnet_head_rows_f{k}.pkl").exists()]
    frames = [f for k in folds for f in pickle.load(open(OUT / f"filnet_head_rows_f{k}.pkl", "rb"))]
    for f in frames:
        f["H"] = {}
    print(f"folds {folds}: {len(frames)} photographs, {sum(len(f['X']) for f in frames)} candidates, "
          f"{frames[0]['Z'].shape[1]} network features", flush=True)
    res, counts, kept = {}, {}, {}
    for kind in KINDS:
        t0 = time.time()
        scores = oof(frames, folds, kind)
        if kind in ("head", "linear"):                         # the head's score becomes a 22nd number of the rule
            for f, s in zip(frames, scores):
                f["H"][kind] = s
        pqs = [S.summary(S.tally(frames, scores, c, True)) for c in CUTS]
        b = int(np.argmax([r["PQ"] for r in pqs]))
        per = {k: S.summary(S.tally([f for f in frames if f["fold"] == k],
                                    [s for f, s in zip(frames, scores) if f["fold"] == k], CUTS[b], True))["PQ"] for k in folds}
        counts[kind] = per_photo(frames, scores, CUTS[b])
        kept[kind] = scores
        res[kind] = dict(pqs[b], cut=float(CUTS[b]), per_fold=per, reads=KINDS[kind][1], by=KINDS[kind][0], fitted=KINDS[kind][2])
        if kind != "rule":
            res[kind]["minus_rule"] = boot(counts[kind], counts["rule"])
        if kind not in ("rule", "rule-in"):
            res[kind]["minus_rule_in"] = boot(counts[kind], counts["rule-in"])
        print(f"{kind:12s} PQ {pqs[b]['PQ']:.4f} SQ {pqs[b]['SQ']:.3f} RQ {pqs[b]['RQ']:.3f} TP {pqs[b]['TP']} FP {pqs[b]['FP']} "
              f"FN {pqs[b]['FN']} cut {CUTS[b]:.2f}  per fold " + " ".join(f"{v:.4f}" for v in per.values())
              + (f"  minus rule {res[kind]['minus_rule']}" if kind != "rule" else "")
              + (f"  minus rule-in {res[kind]['minus_rule_in']}" if "minus_rule_in" in res[kind] else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(OUT / "filnet_head.json", "w"), indent=1, default=float)

    tp = OUT / "filnet_head_test.pkl"
    want = [a for a in sys.argv[1:] if a in KINDS]
    if tp.exists() and want:
        # the test file: every fold model has its own head, fitted on all of that model's unseen photographs;
        # a test candidate is read by each model's head on that model's features, and the answers are averaged
        t = pickle.load(open(tp, "rb"))
        kind = want[0]
        base = kind.split("+")[1] if kind.startswith("rule+") else kind
        heads = {k: fit([f for f in frames if f["fold"] == k], base) for k in t["folds"] if k in folds}
        final = fit(frames, kind) if kind.startswith("rule+") else None
        cut = res[kind]["cut"]
        lines, per = [], []
        for r in t["rows"]:
            s = []
            if len(r["X"]):
                h = np.mean([heads[k].predict(design(dict(X=r["X"], Z=r["Z"][k]), base)) for k in heads], 0)
                s = final.predict(design(dict(X=r["X"], Z=r["Z"][t["folds"][0]], H={base: h}), kind)) if final is not None else h
            n = 0
            for rle, q in zip(r["rle"], s):
                if q >= cut:
                    n += 1
                    lines.append(f"{r['stem']}_{n},{rle['counts'].decode('ascii')}")
            per.append(n)
        csv = OUT / f"submission_filnet{len(heads)}_{kind.replace('+', '_')}.csv"
        csv.write_text("filament_id,segmentation_rle\n" + "\n".join(lines) + "\n")
        v = subprocess.run([sys.executable, str(HERE.parents[1] / "src" / "verify_submission.py"), str(csv),
                            "--test-dir", str(run.find_data()[2])], capture_output=True, text=True)
        res["csv"] = dict(file=str(csv), kind=kind, rows=len(lines), per_image=round(float(np.mean(per)), 2), cut=cut,
                          verify_exit=v.returncode)
        print(f"{csv.name}: {len(lines)} rows, cut {cut}; verify exit {v.returncode}\n{v.stdout[-250:]}", flush=True)
        json.dump(res, open(OUT / "filnet_head.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "extract":
        extract(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        extract_test()
    else:
        main()
