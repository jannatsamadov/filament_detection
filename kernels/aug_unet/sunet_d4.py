"""The compact network (filament-solar-unet, both arms) seen from all eight sides of the square (D4), on the
laptop's GPU: fold 0's 141 validation photographs and the 180 test photographs (11-15).

Each view is the frame mirrored and/or transposed; its geometry channels are computed for that view (the same
output-to-source map the training's rotations and mirrors used), and the eight probability maps are mapped
back and averaged, exactly as the U-Nets' D4 maps were. The plain view is kept too, as a check against the
maps the Kaggle run wrote.
GPU: the shared governor (target 72 % utilisation, pause above 78 C). Resumable: maps already written are
skipped.
    python kernels/aug_unet/sunet_d4.py [pre scratch]   -> out/sunet_maps/<arm>/{val,test}_{plain,d4}/<stem>.png
"""
import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")            # before run.py hides the card
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402  -- also points run.CACHE at the full fold-0 cache
import run  # noqa: E402
from run import torch, cv2, np  # noqa: E402
import pixnet as PX  # noqa: E402
import qwen_seg as Q  # noqa: E402

SRC = HERE.parent / "solar_unet" / "out_kaggle"
OUT = HERE / "out" / "sunet_maps"


def views():
    """(transpose?, flipped dims, the view's output-to-source map a, b, c, d in image axes)."""
    for tr in (False, True):
        for dims in ([], [3], [2], [2, 3]):
            su = -1.0 if 3 in dims else 1.0      # columns mirrored
            sv = -1.0 if 2 in dims else 1.0      # rows mirrored
            yield tr, dims, ((0.0, sv, su, 0.0) if tr else (su, 0.0, 0.0, sv))


@torch.no_grad()
def predict_d4(model, im_u8, stem, device, ondisk, gov):
    b0 = run.b0_deg(stem)
    x = run.to_input(im_u8)
    meta = PX.meta_vec(stem, x, b0)[None].to(device)
    x = x[None, None].to(device)
    acc, plain = 0.0, None
    for tr, dims, g in views():
        with gov.step():
            xi = x.transpose(2, 3) if tr else x
            xi = torch.flip(xi, dims) if dims else xi
            geo = run.geo_channels(torch.tensor([[*g, b0]]), device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p = torch.sigmoid(model(xi.contiguous(), geo, meta).float())
            p = torch.flip(p, dims) if dims else p
            p = p.transpose(2, 3) if tr else p
            plain = p if plain is None else plain
            acc = acc + p
            torch.cuda.synchronize()

    def u8(p):
        return (p[0, 0] * ondisk * 255.0).round().byte().cpu().numpy()
    return u8(plain), u8(acc / 8)


def main(arms):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.9)
    meta = json.load(open(run.CACHE / "meta.json"))
    tr_dir = run.find_data()[1]
    ondisk = Q.disk(device)
    gov = Q.governor()
    log = {}
    for arm in arms:
        model = S.SolarUNet().to(device)
        sd = {k: v.float() for k, v in torch.load(SRC / f"sunet_{arm}.pt", map_location="cpu").items()}
        model.load_state_dict(sd)
        model.eval()
        t0, n_done = time.time(), 0
        for split, folder, files in (("val", tr_dir, meta["val_files"]),
                                     ("test", Path(meta["test_dir"]), meta["test_files"])):
            for view in ("plain", "d4"):
                (OUT / arm / f"{split}_{view}").mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(files):
                s = Path(f).stem
                if (OUT / arm / f"{split}_d4" / f"{s}.png").exists():
                    continue
                p1, p8 = predict_d4(model, cv2.imread(str(folder / f), cv2.IMREAD_GRAYSCALE), s, device, ondisk, gov)
                cv2.imwrite(str(OUT / arm / f"{split}_plain" / f"{s}.png"), p1)
                cv2.imwrite(str(OUT / arm / f"{split}_d4" / f"{s}.png"), p8)
                n_done += 1
                if (k + 1) % 20 == 0:
                    t = (time.time() - t0) / max(n_done, 1)
                    temp = os.popen("nvidia-smi --query-gpu=temperature.gpu,utilization.gpu "
                                    "--format=csv,noheader").read().strip()
                    print(f"[{arm}] {split} {k + 1}/{len(files)}: {t:.2f} s per photograph (8 views); GPU {temp}",
                          flush=True)
        log[arm] = dict(photographs=n_done, sec_per_photo_d4=round((time.time() - t0) / max(n_done, 1), 2))
        print(f"[{arm}] done: {log[arm]}", flush=True)
        del model
        torch.cuda.empty_cache()
    # the plain maps against the ones the Kaggle run wrote (same network, T4 vs the laptop)
    import tarfile
    for arm in arms:
        diffs = []
        with tarfile.open(SRC / f"dump_sunet_{arm}.tar") as t:
            for m in t.getmembers():
                if m.isfile() and Path(m.name).parent.name == "val_plain":
                    ref = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                    mine = cv2.imread(str(OUT / arm / "val_plain" / Path(m.name).name), cv2.IMREAD_GRAYSCALE)
                    diffs.append(np.abs(ref.astype(int) - mine.astype(int)).max())
        log[arm]["plain_vs_kaggle_max_abs_diff_u8"] = dict(median=float(np.median(diffs)), max=int(max(diffs)))
        print(f"[{arm}] plain map vs Kaggle's: max |diff| (0-255) median {np.median(diffs):.0f}, worst {max(diffs)}",
              flush=True)
    json.dump(log, open(OUT / "sunet_d4.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["pre", "scratch"])
