"""The filament head's protocol on the existing models' maps, for a like-for-like comparison (11-15).

solar_unet.stage2_eval scores the new network's outlines on held-out halves of fold 0's 141 validation
photographs. This runs the same pool, the same 21 numbers, the same selector and the same halves on the maps
of the submitted mix, one R50 U-Net and PixNet. These maps carry no network features, so only the variants
that use the 21 numbers are run.
    python kernels/aug_unet/stage2_baselines.py      -> out/stage2_baselines.json, out/stage2_<model>.pkl
"""
import collections, json, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402  -- also points run.CACHE at the full fold-0 cache
import pixel_cls as PC  # noqa: E402
from run import np  # noqa: E402
import run  # noqa: E402
import candidate_pool as C  # noqa: E402

MODELS = ("5 U-Nets + Qwen (submitted mix)", "U-Net R50 (one model)", "PixNet, 30 epochs")


def main():
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    vimg = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    D = PC.Dumps()
    have = set(D.stems())
    S.VARIANTS = {k: v for k, v in S.VARIANTS.items() if v[1] in (None, "X")}
    out = {}
    for name in MODELS:
        frames, t0 = [], time.time()
        for i, f in enumerate(meta["val_files"]):
            stem = Path(f).stem
            if stem not in have:
                continue
            p = D.model(name, stem, {})
            im = np.array(vimg[i])
            cands, X = C.candidates(p, im, run.b0_deg(stem))
            Y, _ = C.targets(cands, by_frame[i])
            frames.append(dict(stem=stem, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                               n_gt=[len(s) for s in by_frame[i]], conflict=C.conflicts(cands),
                               one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool),
                               area=np.array([c["area"] for c in cands], np.int32)))
        tag = name.split(" (")[0].replace(" ", "_").replace("+", "").lower()
        pickle.dump(frames, open(HERE / "out" / f"stage2_{tag}.pkl", "wb"))
        print(f"\n{name}: {len(frames)} photographs, {sum(len(f['one']) for f in frames)} outlines, "
              f"{time.time() - t0:.0f}s", flush=True)
        out[name] = S.stage2_eval(frames)
        json.dump(out, open(HERE / "out" / "stage2_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
