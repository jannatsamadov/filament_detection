"""One table of every model's fold-0 maps: object level (PQ, SQ, RQ) and pixel level (AUC, AP, F1, IoU, accuracy).

PQ: each map thresholded on a small grid (t 0.3-0.8, join 3 or 7 px, min area 100-500), the best point
reported -- the same post-processing for every model, no selection rule. Pixel metrics come from
pixel_cls.py (out/pixel_cls.json), against the union of the experts, at t 0.5.
    python kernels/aug_unet/model_table.py      -> out/model_table.md
"""
import json, multiprocessing as mp, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel_cls as X  # noqa: E402
import post_local as P  # noqa: E402
import run  # noqa: E402


def pq_job(name):
    meta, gt = P.ground_truth()
    run.THRESH, run.MERGE, run.MIN_AREA = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (3, 7), (100, 250, 500)
    D = X.Dumps()
    stems = [Path(f).stem for f in meta["val_files"]]
    cache = {}
    probs = []
    for s in stems:
        cache.clear()
        probs.append(D.model(name, s, cache))
    r = run.sweep(probs, gt)[0]
    return name, {k: r[k] for k in ("PQ", "SQ", "RQ", "thresh", "merge", "min_area")}


def main():
    pix = json.loads((HERE / "out" / "pixel_cls.json").read_text())["models"]
    names = [m for m in X.MODELS if m in pix]
    with mp.Pool(4) as pool:
        pq = dict(pool.imap_unordered(pq_job, names))
    rows = ["| model | PQ | SQ | RQ | pixel AUC | AUC, hard pixels | AP | precision | recall | F1 | IoU | accuracy |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(names, key=lambda m: -pq[m]["PQ"]):
        a, b = pix[m], pq[m]
        rows.append(f"| {m} | {b['PQ']:.3f} | {b['SQ']:.3f} | {b['RQ']:.3f} | {a['AUC']:.3f} | {a['band']['AUC']:.3f} | "
                    f"{a['AP']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} | {a['F1']:.3f} | {a['IoU']:.3f} | "
                    f"{a['accuracy']:.4f} |")
    out = HERE / "out" / "model_table.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    json.dump(pq, open(HERE / "out" / "model_table_pq.json", "w"), indent=1)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
