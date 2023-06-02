

import os
from dataset.DS_CONSTANT import IN32_TRAIN_DIR, IN32_VAL_DIR
from loguru import logger

real_val_dir = str(IN32_VAL_DIR)
real_train_dir = str(IN32_TRAIN_DIR)
sample_dir = '../outputs/v1.6_dino_vits16_in32p_unetfast_ep20/26-07-2022/16-15-47/eval_test_ddim250_s2 _sample'

assert len(os.listdir(sample_dir))>0

from cleanfid import fid as clean_fid

result = clean_fid.compute_fid(sample_dir, real_val_dir)
logger.warning(f'fid with val:{result}')

result = clean_fid.compute_fid(sample_dir, real_train_dir)
logger.warning(f'fid with train:{result}')