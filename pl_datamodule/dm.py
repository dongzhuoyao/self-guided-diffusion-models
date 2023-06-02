import os
import sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from functools import partial
from diffusion_utils.util import instantiate_from_config
from loguru import logger


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        pin_memory,
        train=None,
        validation=None,
        test=None,
        predict=None,
        crossds=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        predict, crossds = None, None  # we dont use it now
        logger.warning("DM:   predict, crossds = None, None")
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        if crossds is not None:
            self.dataset_configs["crossds"] = crossds
            self.crossds_dataloader = self._crossds_dataloader
        self.wrap = wrap

    def prepare_data(self):
        pass
        # for data_cfg in self.dataset_configs.values():
        #    instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = False
        # https://github.com/Lightning-AI/lightning/issues/11910
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else True,
            drop_last=True,
            worker_init_fn=None,
        )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            worker_init_fn=None,
            drop_last=True,
            shuffle=shuffle,
        )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = False
        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=0,
            worker_init_fn=None,
            shuffle=shuffle,
            drop_last=True,
        )

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=0,
            worker_init_fn=None,
            drop_last=True,
        )

    def _crossds_dataloader(self):
        return DataLoader(
            self.datasets["crossds"],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=0,
            worker_init_fn=None,
            drop_last=True,
        )
