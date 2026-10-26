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

    built = [(dump, "aug_dump"), (merged, "aug_post"), (keep, "keep_cnn")]
    for r in ("r2", "r3", "r4"):                     # one training kernel per later run
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
