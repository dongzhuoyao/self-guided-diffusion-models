import os
from pathlib import Path

from loguru import logger

root_dir = "~/data/sg_fid_eval"

FID_DEBUG_DIR = Path(os.path.join(root_dir, "in32_4debug")
                     ).expanduser().resolve()
FID64_DEBUG_DIR = Path(os.path.join(
    root_dir, "in64_4debug")).expanduser().resolve()
FID128_DEBUG_DIR = Path(os.path.join(
    root_dir, "in128_4debug")).expanduser().resolve()


############################################

CIFAR10_TRAIN_DIR = Path(os.path.join(
    root_dir, "cifar10_train")).expanduser().resolve()
CIFAR10_VAL_DIR = Path(os.path.join(
    root_dir, "cifar10_val")).expanduser().resolve()

