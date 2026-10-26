"""Generate the single-file Kaggle kernels from the two sources of truth.

A Kaggle script kernel uploads exactly one file, so these kernels are built from
`aug_unet/run.py` and `aug_unet/post_local.py` rather than kept as copies that
could drift:

  aug_dump/run.py   run.py with MODE = "dump"  (GPU: write probability maps)
  aug_post/run.py   run.py with MODE = "post", plus post_local.py's analysis
                    (CPU only: grid, ensemble, keep-rule, submission CSVs)
  unet_r2/run.py    run.py with RUN = "r2" (the second pair of trainings);
                    --r2-smoke builds it with SMOKE = True

Run it before every push of any of them. The generated files are gitignored.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAIL = 'if __name__ == "__main__":\n    main()\n'
CLI = "# ---------------------------------------------------------------- CLI"
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

    r2 = with_const(run, RUN_LINE, 'RUN = "r2"')
    smoke = "--r2-smoke" in sys.argv
    if smoke:
        r2 = with_const(r2, "SMOKE = False", "SMOKE = True")
    (HERE / "unet_r2").mkdir(exist_ok=True)
    (HERE / "unet_r2" / "run.py").write_text(r2, encoding="utf-8")

    for src, name in ((dump, "aug_dump"), (merged, "aug_post"), (r2, "unet_r2")):
        compile(src, f"{name}/run.py", "exec")
    print("built aug_dump/run.py, aug_post/run.py and unet_r2/run.py" + (" (r2 smoke)" if smoke else ""))


if __name__ == "__main__":
    main()
