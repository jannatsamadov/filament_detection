"""Generate the single-file Kaggle kernels from the two sources of truth.

A Kaggle script kernel uploads exactly one file, so these kernels are built from
`aug_unet/run.py` and `aug_unet/post_local.py` rather than kept as copies that
could drift:

  aug_dump/run.py   run.py with MODE = "dump"  (GPU: write probability maps)
  aug_post/run.py   run.py with MODE = "post", plus post_local.py's analysis
                    (CPU only: grid, ensemble, keep-rule, submission CSVs)
  unet_r2/run.py,   run.py with RUN = "r2" / "r3" (later pairs of trainings);
  unet_r3/run.py    --r2-smoke / --r3-smoke build one with SMOKE = True
  keep_cnn/run.py   aug_post's file plus aug_unet/keep_cnn.py (the image-based
                    keep-rule), entering at kaggle_keep_cnn() instead of main()

Run it before every push of any of them. The generated files are gitignored.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAIL = 'if __name__ == "__main__":\n    main()\n'
CLI = "# ---------------------------------------------------------------- CLI"
CNN_START = "# ---------------------------------------------------------------- keep-cnn settings"
POOL_START = "# ---------------------------------------------------------------- pool settings"
KF_START = "# ---------------------------------------------------------------- k-fold settings"
PIX_START = "# ---------------------------------------------------------------- pixel settings"
XM_START = "# ---------------------------------------------------------------- pixnet-mix settings"
RUN_LINE = 'RUN = "r1"                                   # kernels/build.py sets this per kernel'


def with_const(src, line, new):
    """Replace one whole top-level line, which must occur exactly once."""
    old = "\n" + line + "\n"
    assert src.count(old) == 1, f"expected exactly one {line!r}"
    return src.replace(old, "\n" + new + "\n")


def main():
    run = (HERE / "aug_unet" / "run.py").read_text(encoding="utf-8")
    post = (HERE / "aug_unet" / "post_local.py").read_text(encoding="utf-8")
    assert run.endswith(TAIL)

    dump = with_const(run, 'MODE = "train"', 'MODE = "dump"')
    (HERE / "aug_dump" / "run.py").write_text(dump, encoding="utf-8")

    body = post[post.index("THRESH = "):post.index(CLI)]
    merged = (with_const(run, 'MODE = "train"', 'MODE = "post"')[:-len(TAIL)]
              + "\n\n# ================================================================ post-processing\n"
              + "# copied from aug_unet/post_local.py by kernels/build.py -- edit it there\n"
              + "import multiprocessing as mp, tarfile  # noqa: E402\n"
              + "run = sys.modules[__name__]          # post_local.py calls run.<name>\n\n"
              + body + "\n" + TAIL)
    (HERE / "aug_post").mkdir(exist_ok=True)
    (HERE / "aug_post" / "run.py").write_text(merged, encoding="utf-8")

    cnn = (HERE / "aug_unet" / "keep_cnn.py").read_text(encoding="utf-8")
    keep = (merged[:-len(TAIL)]
            + "\n\n# ================================================================ image-based keep-rule\n"
            + "# copied from aug_unet/keep_cnn.py by kernels/build.py -- edit it there\n\n"
            + cnn[cnn.index(CNN_START):cnn.index(CLI)]
            + 'if __name__ == "__main__":\n    kaggle_keep_cnn()\n')
    (HERE / "keep_cnn").mkdir(exist_ok=True)
    (HERE / "keep_cnn" / "run.py").write_text(keep, encoding="utf-8")

    pool = (HERE / "aug_unet" / "candidate_pool.py").read_text(encoding="utf-8")
    kf = (HERE / "aug_unet" / "kfold.py").read_text(encoding="utf-8")
    kfold = (merged[:-len(TAIL)]
             + "\n\n# ================================================================ k-fold evaluation\n"
             + "# copied from aug_unet/candidate_pool.py and kfold.py by kernels/build.py -- edit them there\n"
             + "from scipy import ndimage  # noqa: E402\nP = C = run\n"
             + "VERIFY_SRC = " + repr((HERE.parent / "src" / "verify_submission.py").read_text(encoding="utf-8")) + "\n\n"
             + pool[pool.index(POOL_START):pool.index(CLI)]
             + kf[kf.index(KF_START):kf.index(CLI)]
             + 'if __name__ == "__main__":\n    kaggle_kfold()\n')
    (HERE / "kfold").mkdir(exist_ok=True)
    (HERE / "kfold" / "run.py").write_text(kfold, encoding="utf-8")
    # v8: the same file entering at the YOLO comparison (needs filament-yolo and -yolo-b)
    kfold_yolo = kfold.replace('if __name__ == "__main__":\n    kaggle_kfold()\n',
                               'if __name__ == "__main__":\n    kaggle_kfold_yolo()\n')
    assert kfold_yolo != kfold
    (HERE / "kfold_yolo").mkdir(exist_ok=True)
    (HERE / "kfold_yolo" / "run.py").write_text(kfold_yolo, encoding="utf-8")
    # its plumbing check, a kernel of its own (ten photographs a fold, missing YOLO folds empty)
    kfold_yolo_smoke = with_const(kfold_yolo, "KFY_SMOKE = False                             "
                                  "# kernels/build.py sets this in kfold_yolo_smoke/run.py", "KFY_SMOKE = True")
    (HERE / "kfold_yolo_smoke").mkdir(exist_ok=True)
    (HERE / "kfold_yolo_smoke" / "run.py").write_text(kfold_yolo_smoke, encoding="utf-8")
    # 11-15: which Qwen members the final map carries (needs filament-qwen-seg2 for its last arms)
    kfold_q2 = kfold.replace('if __name__ == "__main__":\n    kaggle_kfold()\n',
                             'if __name__ == "__main__":\n    kaggle_kfold_q2()\n')
    assert kfold_q2 != kfold
    (HERE / "kfold_q2").mkdir(exist_ok=True)
    (HERE / "kfold_q2" / "run.py").write_text(kfold_q2, encoding="utf-8")

    # v9: the user's pixel table (aug_unet/pixel_gbm.py) on top of the same merged file
    pix = (HERE / "aug_unet" / "pixel_gbm.py").read_text(encoding="utf-8")
    pixel = (kfold[:-len('if __name__ == "__main__":\n    kaggle_kfold()\n')]
             + "\n\n# ================================================================ pixel table\n"
             + "# copied from aug_unet/pixel_gbm.py by kernels/build.py -- edit it there\nK = run\n\n"
             + pix[pix.index(PIX_START):pix.index(CLI)]
             + 'if __name__ == "__main__":\n    kaggle_pixel()\n')
    (HERE / "pixel").mkdir(exist_ok=True)
    (HERE / "pixel" / "run.py").write_text(pixel, encoding="utf-8")
    smoke_line = next(ln for ln in pix.splitlines() if ln.startswith("PIX_SMOKE = False"))
    pixel_smoke = with_const(pixel, smoke_line, "PIX_SMOKE = True")
    (HERE / "pixel_smoke").mkdir(exist_ok=True)
    (HERE / "pixel_smoke" / "run.py").write_text(pixel_smoke, encoding="utf-8")
    # the same table with a random forest instead of gradient boosting (user, 11-12)
    learner_line = next(ln for ln in pix.splitlines() if ln.startswith('PIX_LEARNER = "gbm"'))
    pixel_rf = with_const(pixel, learner_line, 'PIX_LEARNER = "rf"')
    (HERE / "pixel_rf").mkdir(exist_ok=True)
    (HERE / "pixel_rf" / "run.py").write_text(pixel_rf, encoding="utf-8")
    # the forest for fold 0 only, its 141 validation maps tarred out for the pixel metrics (11-14)
    fold0_line = next(ln for ln in pix.splitlines() if ln.startswith("PIX_FOLD0 = False"))
    pixel_rf0 = with_const(pixel_rf, fold0_line, "PIX_FOLD0 = True")
    (HERE / "pixel_rf0").mkdir(exist_ok=True)
    (HERE / "pixel_rf0" / "run.py").write_text(pixel_rf0, encoding="utf-8")
    # PixNet (filament-pixnet's maps) averaged into fold 0's ensemble, and the gap filling (11-12)
    xm = (HERE / "aug_unet" / "pixnet_mix.py").read_text(encoding="utf-8")
    pixnet_mix = (kfold[:-len('if __name__ == "__main__":\n    kaggle_kfold()\n')]
                  + "\n\n# ================================================================ PixNet in the ensemble\n"
                  + "# copied from aug_unet/pixnet_mix.py by kernels/build.py -- edit it there\nK = run\n\n"
                  + xm[xm.index(XM_START):xm.index(CLI)]
                  + 'if __name__ == "__main__":\n    kaggle_pixnet_mix()\n')
    (HERE / "pixnet_mix").mkdir(exist_ok=True)
    (HERE / "pixnet_mix" / "run.py").write_text(pixnet_mix, encoding="utf-8")
    xm_smoke_line = next(ln for ln in xm.splitlines() if ln.startswith("XM_SMOKE = False"))
    pixnet_mix_smoke = with_const(pixnet_mix, xm_smoke_line, "XM_SMOKE = True")
    (HERE / "pixnet_mix_smoke").mkdir(exist_ok=True)
    (HERE / "pixnet_mix_smoke" / "run.py").write_text(pixnet_mix_smoke, encoding="utf-8")

    # the Qwen fine-tune: its modules import each other by name, so instead of merging
    # them the kernel carries their sources, writes them out and imports them as usual
    mods = {n: (HERE / "aug_unet" / n).read_text(encoding="utf-8")
            for n in ("run.py", "post_local.py", "qwen_probe.py", "qwen_finetune.py", "qwen_geo.py", "yolo_seg.py",
                      "qwen_seg.py", "pixnet.py")}
    qwen_pip = ('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "peft", "accelerate"])\n'
                "# Kaggle's torchao 0.10 makes peft refuse to build LoRA layers (needs > 0.16); nothing here uses it\n"
                'subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])\n')
    yolo_pip = 'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])\n'

    def launcher(kernel, entry, pip=qwen_pip, env=""):
        return (f'"""Kaggle kernel {kernel}, generated by kernels/build.py from aug_unet/'
                '{run,post_local,qwen_probe,qwen_finetune,qwen_geo,yolo_seg}.py -- edit those."""\n'
                "import os, subprocess, sys\nfrom pathlib import Path\n\n" + env + pip
                + "SRC = " + repr(mods) + "\n"
                'd = Path("/kaggle/working/src")\nd.mkdir(parents=True, exist_ok=True)\n'
                "for name, text in SRC.items():\n    (d / name).write_text(text, encoding=\"utf-8\")\n"
                f"sys.path.insert(0, str(d))\nimport {entry}  # noqa: E402\n{entry}.kaggle_main()\n")

    qwen, qgeo = launcher("filament-qwen-ft", "qwen_finetune"), launcher("filament-qwen-geo", "qwen_geo")
    smoke = "--yolo-smoke" in sys.argv
    yolo = launcher("filament-yolo", "yolo_seg", pip=yolo_pip,
                    env='os.environ["FIL_YOLO_SMOKE"] = "1"\n' if smoke else "")
    # folds 2-4 in a kernel of their own, so filament-yolo keeps its folds 0-1 output
    yolo_b = launcher("filament-yolo-b", "yolo_seg", pip=yolo_pip, env='os.environ["FIL_YOLO_FOLDS"] = "2,3,4"\n'
                      + ('os.environ["FIL_YOLO_SMOKE"] = "1"\n' if smoke else ""))
    qsmoke = "--qseg-smoke" in sys.argv
    qseg = launcher("filament-qwen-seg", "qwen_seg", env='os.environ["FIL_QSEG_SMOKE"] = "1"\n' if qsmoke else "")
    # both Qwen arms continued from their saved weights (11-15)
    qseg2 = launcher("filament-qwen-seg2", "qwen_seg", env='os.environ["FIL_QSEG_CONT"] = "1"\n'
                     + ('os.environ["FIL_QSEG_SMOKE"] = "1"\n' if qsmoke else ""))
    assert "FIL_QSEG_CONT" in qseg2 and "CONT_EPOCHS" in qseg2
    # the user's from-scratch per-pixel network (aug_unet/pixnet.py): needs nothing beyond Kaggle's image
    psmoke = "--pixnet-smoke" in sys.argv
    pixnet = launcher("filament-pixnet", "pixnet", pip="", env='os.environ["FIL_PIXNET_SMOKE"] = "1"\n' if psmoke else "")
    # PixNet with the user's table as channels, and with the rotation channels too: one arm per T4 (11-14)
    pmods = dict(mods, **{n: (HERE / "aug_unet" / n).read_text(encoding="utf-8")
                          for n in ("candidate_pool.py", "kfold.py", "pixel_gbm.py", "pixel_nb.py")})
    prot_env = 'os.environ["FIL_PIXNET_ARMS"] = "table,tablerot"\n'
    pixnet_rot = launcher("filament-pixnet-rot", "pixnet", pip="", env=prot_env).replace(
        "SRC = " + repr(mods), "SRC = " + repr(pmods))
    pixnet_rot_smoke = launcher("filament-pixnet-rot-smoke", "pixnet", pip="",
                                env=prot_env + 'os.environ["FIL_PIXNET_SMOKE"] = "1"\n').replace(
        "SRC = " + repr(mods), "SRC = " + repr(pmods))
    assert "pixel_nb.py" in pixnet_rot and "pixel_nb.py" in pixnet_rot_smoke
    for name, text in (("qwen_ft", qwen), ("qwen_geo", qgeo), ("yolo", yolo), ("yolo_b", yolo_b), ("qwen_seg", qseg),
                       ("qwen_seg2", qseg2),
                       ("pixnet", pixnet), ("pixnet_rot", pixnet_rot), ("pixnet_rot_smoke", pixnet_rot_smoke)):
        (HERE / name).mkdir(exist_ok=True)
        (HERE / name / "run.py").write_text(text, encoding="utf-8")

    # the user's questions about the pixel forest, one arm each (aug_unet/pixel_ideas.py, 11-14)
    imods = {n: (HERE / "aug_unet" / n).read_text(encoding="utf-8")
             for n in ("run.py", "post_local.py", "candidate_pool.py", "kfold.py", "pixel_gbm.py", "pixel_ideas.py")}
    ideas = ('"""Kaggle kernel filament-pixel-ideas, generated by kernels/build.py from aug_unet/pixel_ideas.py -- edit that."""\n'
             "import sys\nfrom pathlib import Path\n\nSRC = " + repr(imods) + "\n"
             'd = Path("/kaggle/working/src")\nd.mkdir(parents=True, exist_ok=True)\n'
             "for name, text in SRC.items():\n    (d / name).write_text(text, encoding=\"utf-8\")\n"
             "sys.path.insert(0, str(d))\nimport pixel_ideas  # noqa: E402\npixel_ideas.kaggle_main()\n")
    (HERE / "pixel_ideas").mkdir(exist_ok=True)
    (HERE / "pixel_ideas" / "run.py").write_text(ideas, encoding="utf-8")

    # the user's table with every neighbour's own value within 12 px (aug_unet/pixel_nb.py, 11-14)
    nmods = {n: (HERE / "aug_unet" / n).read_text(encoding="utf-8")
             for n in ("run.py", "post_local.py", "candidate_pool.py", "kfold.py", "pixel_gbm.py", "pixel_nb.py")}
    nbk = ('"""Kaggle kernel filament-pixel-nb, generated by kernels/build.py from aug_unet/pixel_nb.py -- edit that."""\n'
           "import sys\nfrom pathlib import Path\n\nSRC = " + repr(nmods) + "\n"
           'd = Path("/kaggle/working/src")\nd.mkdir(parents=True, exist_ok=True)\n'
           "for name, text in SRC.items():\n    (d / name).write_text(text, encoding=\"utf-8\")\n"
           "sys.path.insert(0, str(d))\nimport pixel_nb  # noqa: E402\npixel_nb.kaggle_main()\n")
    (HERE / "pixel_nb").mkdir(exist_ok=True)
    (HERE / "pixel_nb" / "run.py").write_text(nbk, encoding="utf-8")
    # the same table with a random-forest classifier, as the user asked (11-14)
    nb_line = next(ln for ln in nmods["pixel_nb.py"].splitlines() if ln.startswith('NB_LEARNER = "reg"'))
    cmods = dict(nmods, **{"pixel_nb.py": with_const(nmods["pixel_nb.py"], nb_line, 'NB_LEARNER = "cls"')})
    nbc = nbk.replace("SRC = " + repr(nmods), "SRC = " + repr(cmods)).replace("filament-pixel-nb,", "filament-pixel-nbc,")
    assert nbc.count('NB_LEARNER = "cls"') == nbk.count('NB_LEARNER = "cls"') + 1
    (HERE / "pixel_nbc").mkdir(exist_ok=True)
    (HERE / "pixel_nbc" / "run.py").write_text(nbc, encoding="utf-8")
    # the classifier with the rotation columns (user, 11-14), and its plumbing check
    rot_line = next(ln for ln in nmods["pixel_nb.py"].splitlines() if ln.startswith("NB_ROT = False"))
    rmods = dict(cmods, **{"pixel_nb.py": with_const(cmods["pixel_nb.py"], rot_line, "NB_ROT = True")})
    nbr = nbk.replace("SRC = " + repr(nmods), "SRC = " + repr(rmods)).replace("filament-pixel-nb,", "filament-pixel-nbr,")
    nbr_smoke = nbr.replace("import sys\n", 'import os, sys\nos.environ["FIL_NB_SMOKE"] = "1"\n', 1)
    assert nbr != nbk and nbr_smoke != nbr
    # Claude's arm next to the user's table: the classifier without the 880 neighbour positions (11-15)
    nbcc = nbc.replace("import sys\n", 'import os, sys\nos.environ["FIL_NB_CC"] = "1"\n', 1).replace(
        "filament-pixel-nbc,", "filament-pixel-nbcc,")
    assert nbcc != nbc and 'NB_LEARNER = "cls"' in nbcc
    for name, text in (("pixel_nbr", nbr), ("pixel_nbr_smoke", nbr_smoke), ("pixel_nbcc", nbcc)):
        (HERE / name).mkdir(exist_ok=True)
        (HERE / name / "run.py").write_text(text, encoding="utf-8")

    built = [(nbcc, "pixel_nbcc"), (nbr, "pixel_nbr"), (nbr_smoke, "pixel_nbr_smoke"), (nbc, "pixel_nbc"), (nbk, "pixel_nb"), (ideas, "pixel_ideas"), (dump, "aug_dump"), (merged, "aug_post"), (keep, "keep_cnn"), (kfold, "kfold"), (kfold_yolo, "kfold_yolo"),
             (kfold_yolo_smoke, "kfold_yolo_smoke"), (kfold_q2, "kfold_q2"), (pixel, "pixel"), (pixel_smoke, "pixel_smoke"), (pixel_rf, "pixel_rf"),
             (pixel_rf0, "pixel_rf0"), (pixnet_mix, "pixnet_mix"), (pixnet_mix_smoke, "pixnet_mix_smoke"),
             (qwen, "qwen_ft"), (qgeo, "qwen_geo"), (yolo, "yolo" + (" (smoke)" if smoke else "")),
             (yolo_b, "yolo_b" + (" (smoke)" if smoke else "")), (qseg, "qwen_seg" + (" (smoke)" if qsmoke else "")),
             (qseg2, "qwen_seg2" + (" (smoke)" if qsmoke else "")),
             (pixnet, "pixnet" + (" (smoke)" if psmoke else ""))]
    for r in ("r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"):               # one training kernel per later run
        src = with_const(run, RUN_LINE, f'RUN = "{r}"')
        if f"--{r}-smoke" in sys.argv:
            src = with_const(src, "SMOKE = False", "SMOKE = True")
        (HERE / f"unet_{r}").mkdir(exist_ok=True)
        (HERE / f"unet_{r}" / "run.py").write_text(src, encoding="utf-8")
        built.append((src, f"unet_{r}" + (" (smoke)" if f"--{r}-smoke" in sys.argv else "")))

    for src, name in built:
        compile(src, f"{name}/run.py", "exec")
    print("built " + ", ".join(name for _, name in built))


if __name__ == "__main__":
    main()
