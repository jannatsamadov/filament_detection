"""Qwen3-VL taught on our candidates: does a fine-tuned vision-language model judge
them better than the keep-rule?

qwen_probe.py asked the model zero-shot and it said "yes" to nearly everything
(AUC 0.68 against the GBM's 0.78, nothing added). This trains it. QLoRA: the
2B model in 4-bit, low-rank adapters on every linear layer of the vision encoder
and of the language model. Each candidate is the same two-panel image and
question as the probe; the training signal is the logit of "yes" against "no",
pulled by binary cross-entropy towards the share of annotation sets in which
the candidate is matched.

One honest split: train on the candidates of folds 1-4 (each fold's own R50
model, which never saw it), test on fold 0's 1340. The 10-feature GBM keep-rule
is trained and tested on exactly the same split, so the three -- GBM, fine-tuned
Qwen, and their rank average (fixed 50/50, nothing tuned on fold 0) -- compare
on the same candidates, with a paired bootstrap over fold 0's photographs.

Local GPU only; candidates and the adapter are cached.
    python kernels/aug_unet/qwen_finetune.py
"""
from __future__ import annotations

import argparse, json, math, pickle, random, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))     # the shared folder: gpu_guard
import qwen_probe as QP  # noqa: E402  -- sets FIL_LOCAL_GPU and HF_HOME, imports run (torch before cv2)
from qwen_probe import run, torch, cv2, np  # noqa: E402
from post_local import comp_features, matched_iou, pq_of  # noqa: E402

SIDE = 320                                   # per panel: 10 x 20 visual tokens for the pair
OUT, RES = QP.OUT, QP.RES


class _NoGovernor:
    """Kaggle's T4s need no duty cycling."""
    peak_temp = last_temp = pauses = 0

    def step(self):
        import contextlib
        return contextlib.nullcontext()

    def summary(self):
        return "none (Kaggle)"


def governor():
    """72% / pause at 78 C on the laptop: the unguarded zero-shot 4B run reached 80 C there."""
    if run.ON_KAGGLE:
        return _NoGovernor()
    from gpu_guard import Governor
    return Governor(target_util=72)
TARGETS = (r".*(visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
           r"|language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj))")


def fold_candidates(fold):
    """As qwen_probe.candidates, for one of folds 1-4 and its own R50 model."""
    run.set_fold(fold)
    if fold == 0:
        return QP.candidates()
    path = OUT / f"candidates_fold{fold}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    from scipy import ndimage
    if not (run.CACHE / "val_gt.pkl").exists():
        run.prepare(False, with_train=False)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_img = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    by = {}
    for i, g in enumerate(gt):
        by.setdefault(g["frame"], []).append((i, g["crops"]))
    tar = tarfile.open(QP.find_dump(f"r50f{fold}"))
    names = {Path(n).stem: n for n in tar.getnames() if "val_d4/" in n and n.endswith(".png")}
    rows = []
    for f, sets in sorted(by.items()):
        stem = Path(meta["val_files"][f]).stem
        p = cv2.imdecode(np.frombuffer(tar.extractfile(names[stem]).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = np.asarray(val_img[f])
        lab, area = run.components(p, QP.T, QP.K)
        ids = [j for j in range(1, len(area)) if area[j] >= QP.AMIN]
        n_gt = [len(c) for _, c in sets]
        if not ids:
            rows.append(dict(stem=stem, X=np.zeros((0, 10)), Y=np.zeros((0, len(sets))), n_gt=n_gt, crops=[]))
            continue
        boxes = ndimage.find_objects(lab)
        crops = []
        for j in ids:
            sy, sx = boxes[j - 1]
            side = int(np.clip(max(sy.stop - sy.start, sx.stop - sx.start) + 2 * QP.PAD, *QP.WIN))
            y0 = int(np.clip((sy.start + sy.stop - side) // 2, 0, run.FULL - side))
            x0 = int(np.clip((sx.start + sx.stop - side) // 2, 0, run.FULL - side))
            crops.append((img[y0:y0 + side, x0:x0 + side].copy(), lab[y0:y0 + side, x0:x0 + side] == j))
        rows.append(dict(stem=stem, X=comp_features(p, img, lab, area, ids), Y=matched_iou(lab, area, ids, sets),
                         n_gt=n_gt, crops=crops))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(path, "wb"))
    print(f"fold {fold}: {len(rows)} photographs, {sum(len(r['crops']) for r in rows)} candidates", flush=True)
    return rows


def panel(im, m, aug=False):
    if aug:                                  # the Sun has no preferred orientation in a crop
        k = random.randrange(4)
        im, m = np.rot90(im, k), np.rot90(m, k)
        if random.random() < 0.5:
            im, m = im[:, ::-1], m[:, ::-1]
    old, QP.SIDE = QP.SIDE, SIDE
    try:
        return QP.panel(np.ascontiguousarray(im), np.ascontiguousarray(m))
    finally:
        QP.SIDE = old


def load(model_id, adapter=None):
    """4-bit on the laptop's 6 GB (QLoRA); fp16 on Kaggle's 16 GB T4s."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    quant = not run.ON_KAGGLE
    extra = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)) if quant else {}
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16, device_map="cuda:0", **extra)
    proc = AutoProcessor.from_pretrained(model_id)
    if adapter is not None:
        return PeftModel.from_pretrained(model, adapter), proc
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, bias="none")
    return get_peft_model(model, cfg), proc


def yes_no(proc):
    tok = proc.tokenizer
    return ([tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")],
            [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")])


def margin(model, proc, text, image, yes, no):
    """log P(yes) - log P(no) at the answer position."""
    inp = proc(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return torch.logsumexp(lp[yes], 0) - torch.logsumexp(lp[no], 0)


def train(model, proc, items, epochs, lr, accum):
    from PIL import Image
    yes, no = yes_no(proc)
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QP.PROMPT}]}],
                                    add_generation_prompt=True, tokenize=False)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * len(items) // accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps, 1)))))
    scaler = torch.amp.GradScaler()
    gov = governor()
    model.train()
    t0, run_loss, n = time.time(), 0.0, 0
    for ep in range(epochs):
        order = list(range(len(items)))
        random.shuffle(order)
        for i, k in enumerate(order):
            im, m, t = items[k]
            with gov.step():
                s = margin(model, proc, text, Image.fromarray(panel(im, m, aug=True)), yes, no)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(t, device=s.device)) / accum
                scaler.scale(loss).backward()
                run_loss += loss.item() * accum; n += 1
                if (i + 1) % accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                torch.cuda.synchronize()
            if (i + 1) % 200 == 0:
                print(f"epoch {ep + 1} {i + 1}/{len(items)}  loss {run_loss / n:.4f}  "
                      f"{(time.time() - t0) / (ep * len(items) + i + 1):.2f} s each  "
                      f"peak {gov.peak_temp} C, last {gov.last_temp} C, pauses {gov.pauses}", flush=True)
                run_loss, n = 0.0, 0
    print("governor:", gov.summary(), flush=True)
    model.eval()


@torch.no_grad()
def predict(model, proc, rows):
    from PIL import Image
    yes, no = yes_no(proc)
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QP.PROMPT}]}],
                                    add_generation_prompt=True, tokenize=False)
    gov, out = governor(), []
    for r in rows:
        for im, m in r["crops"]:
            with gov.step():
                out.append(float(margin(model, proc, text, Image.fromarray(panel(im, m)), yes, no)))
    print("governor (predict):", gov.summary(), flush=True)
    return np.array(out)


def evaluate(train_rows, test_rows, qft):
    from scipy.stats import rankdata
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    Xtr = np.concatenate([r["X"] for r in train_rows]); ytr = np.concatenate([r["Y"].mean(axis=1) for r in train_rows])
    Xte = np.concatenate([r["X"] for r in test_rows]); yte = np.concatenate([r["Y"].mean(axis=1) for r in test_rows])
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                        l2_regularization=1.0, random_state=0).fit(Xtr, ytr).predict(Xte)
    blend = 0.5 * rankdata(gbm) / len(gbm) + 0.5 * rankdata(qft) / len(qft)
    groups = np.concatenate([[n] * len(r["X"]) for n, r in enumerate(test_rows)])
    hit = (yte > 0).astype(int)

    def counts(s, cut):
        c = np.zeros((len(test_rows), 4))
        for n, r in enumerate(test_rows):
            keep = s[groups == n] >= cut
            yy = r["Y"][keep]
            for q, g in enumerate(r["n_gt"]):
                tp = (yy[:, q] > 0).sum()
                c[n] += (tp, keep.sum() - tp, g - tp, yy[:, q].sum())
        return c

    def best(s):
        cuts = np.unique(np.quantile(s, np.linspace(0, 0.95, 96)))
        pq = [float(pq_of(counts(s, c).sum(axis=0))) for c in cuts]
        b = int(np.argmax(pq))
        return counts(s, cuts[b]), round(pq[b], 4)

    def boot(a, b):
        rng = np.random.default_rng(0)
        d = [float(pq_of(a[s].sum(axis=0)) - pq_of(b[s].sum(axis=0)))
             for s in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        return dict(mean=round(float(np.mean(d)), 4), lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))

    cg, pg = best(gbm); cq, pq = best(qft); cb, pb = best(blend)
    return dict(train_candidates=int(len(ytr)), test_candidates=int(len(yte)),
                auc=dict(gbm10=round(float(roc_auc_score(hit, gbm)), 4), qwen_ft=round(float(roc_auc_score(hit, qft)), 4),
                         blend=round(float(roc_auc_score(hit, blend)), 4)),
                pq_best_cut=dict(gbm10=pg, qwen_ft=pq, blend=pb),
                qwen_ft_minus_gbm=boot(cq, cg), blend_minus_gbm=boot(cb, cg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: train on this many candidates only")
    ap.add_argument("--candidates-only", action="store_true")
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    train_rows = [r for f in (1, 2, 3, 4) for r in fold_candidates(f)]
    test_rows = fold_candidates(0)
    if a.candidates_only:
        return
    items = [(im, m, float((r["Y"][i] > 0).mean())) for r in train_rows for i, (im, m) in enumerate(r["crops"])]
    if a.limit:
        random.shuffle(items); items = items[:a.limit]
    tag = a.model.split("/")[-1] + (f"-smoke{a.limit}" if a.limit else "")
    RES.mkdir(parents=True, exist_ok=True)
    adapter = RES / f"lora_{tag}"
    if (adapter / "adapter_config.json").exists():
        model, proc = load(a.model, str(adapter))
    else:
        model, proc = load(a.model)
        train(model, proc, items, a.epochs, a.lr, a.accum)
        model.save_pretrained(str(adapter))
    model.eval()
    qft = predict(model, proc, test_rows if not a.limit else test_rows[:20])
    np.save(RES / f"ft_scores_{tag}.npy", qft)
    if a.limit:
        print("smoke ok:", qft[:10].round(3), flush=True)
        return
    res = evaluate(train_rows, test_rows, qft)
    res.update(model=a.model, epochs=a.epochs, lr=a.lr, train_items=len(items))
    json.dump(res, open(RES / f"ft_result_{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


KAGGLE_ARMS = ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct")


def kaggle_main():
    """The kernel filament-qwen-ft: candidates once on the CPU, then one model per T4."""
    import subprocess
    t0 = time.time()
    for f in (1, 2, 3, 4, 0):
        fold_candidates(f)
    print(f"candidates ready in {time.time() - t0:.0f}s", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, m in enumerate(KAGGLE_ARMS):
        log = open(RES / f"log_{m.split('/')[-1]}.txt", "w")
        env = dict(run.os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((m, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--model", m],
                                          env=env, stdout=log, stderr=subprocess.STDOUT), log))
    last = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for m, _, _ in procs:
                lines = [x for x in (RES / f"log_{m.split('/')[-1]}.txt").read_text().splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {m.split('/')[-1]}: {lines[-1][:160] if lines else ''}", flush=True)
    for m, p, _ in procs:
        print(f"\n===== {m} (exit {p.returncode}) =====")
        print((RES / f"log_{m.split('/')[-1]}.txt").read_text()[-2500:], flush=True)


if __name__ == "__main__":
    main()
