Dump kernel: the same `../aug_unet/run.py`, pushed with `MODE = "dump"`. It attaches
the training kernel's output (`best_*.pt`) and writes every validation and test
probability map, plain and D4 TTA, for offline post-processing
(`../aug_unet/post_local.py`). `run.py` here is generated -- do not edit it:

    sed 's/^MODE = "train"$/MODE = "dump"/' kernels/aug_unet/run.py > kernels/aug_dump/run.py
    kaggle kernels push -p kernels/aug_dump --accelerator NvidiaTeslaT4
