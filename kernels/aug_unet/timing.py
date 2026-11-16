"""Inference time per test photograph, per part of the pipeline (the challenge also weighs efficiency).

Weights do not change the time, so the networks are built untrained. Each forward pass runs inside the
GPU governor (<= 72 % load); only the pass itself is timed (CUDA-synchronised), median of 5 after 2 warm-ups.
    FIL_LOCAL_GPU=1 python kernels/aug_unet/timing.py      -> out/timing.json
"""
import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")
import json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from run import torch, cv2, np  # noqa: E402
import candidate_pool as C  # noqa: E402
import qwen_seg as Q  # noqa: E402
import pixnet as PX  # noqa: E402

sys.path.insert(0, str(HERE.parents[2]))
from gpu_guard import Governor  # noqa: E402


def timed(fn, gov, n=5, warm=2, cuda=True):
    out = []
    for i in range(warm + n):
        with gov.step():
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t
        if i >= warm:
            out.append(dt)
    return round(statistics.median(out), 3)


def main():
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.9)
    gov = Governor(target_util=72)
    te = run.find_data()[2]
    stem = "20130530133654Bh"
    im = cv2.imread(str(te / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    ondisk = Q.disk(dev)
    res = dict(gpu=torch.cuda.get_device_name(0), photo=stem, seconds={})
    S = res["seconds"]

    for enc in ("resnet34", "resnet50"):
        m = run.UNet(enc, pretrained=False).to(dev).eval()
        S[f"U-Net {enc}, one pass"] = timed(lambda: run.predict(m, im, dev, ondisk, "plain"), gov)
        S[f"U-Net {enc}, D4 (8 passes)"] = timed(lambda: run.predict(m, im, dev, ondisk, "d4"), gov, n=3)
        del m
        torch.cuda.empty_cache()
    print(S, flush=True)

    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = AutoConfig.from_pretrained(Q.MODEL).vision_config
    qm = Q.QwenSeg(Qwen3VLVisionModel._from_config(cfg, attn_implementation="sdpa")).to(dev).eval()
    S["Qwen3-VL segmenter, one pass"] = timed(lambda: Q.predict(qm, im, dev, ondisk), gov, n=3)
    del qm
    torch.cuda.empty_cache()
    print(S, flush=True)

    pm = PX.PixNet().to(dev).eval()
    S[f"PixNet, one pass ({sum(p.numel() for p in pm.parameters()) / 1e6:.1f} M, tiled on this card)"] = \
        timed(lambda: PX.predict(pm, im, stem, dev, ondisk), gov)
    del pm
    torch.cuda.empty_cache()

    import tarfile                                # a real map of this photograph (one R34 fold model, D4)
    with tarfile.open(HERE.parent / "aug_dump/out_kaggle/dump_full.tar") as t:
        mem = next(x for x in t.getmembers() if x.name.endswith(f"test_d4/{stem}.png"))
        p = cv2.imdecode(np.frombuffer(t.extractfile(mem).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    S["candidates + 21 features (CPU)"] = timed(lambda: C.candidates(p, im, run.b0_deg(stem)), gov, n=3, cuda=False)
    (HERE / "out" / "timing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
