import numpy as np
import torch
from click import Path
from loguru import logger


def prepare_feat(pl_module, batch_data):
    condition_method = pl_module.hparams.condition_method
    return batch_data




