# ------------------------------------------------------------------------------------
# Modified from VQGAN (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy
import random
from utils.general import initialize_from_config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class ZipDataLoader:
    def __init__(self, *data_loaders):
        self.data_loaders = data_loaders

    def __len__(self):
        return min(len(dl) for dl in self.data_loaders)

    def __iter__(self):
        return zip(*self.data_loaders)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size: int, train: Optional[OmegaConf] = None,
                 train2: Optional[OmegaConf] = None,
                 validation: Optional[OmegaConf] = None,
                 test: Optional[OmegaConf] = None,
                 num_workers: Optional[int] = None):
        super().__init__()
        self.dataset_configs = dict()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if train2 is not None:
            self.dataset_configs["train"] = train
            self.dataset_configs["train2"] = train2
            self.train_dataloader = self._train2_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            initialize_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, initialize_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,drop_last=True,worker_init_fn=seed_worker)
    def _train2_dataloader(self):
        src_trainset = self.datasets["train"]
        tgt_trainset = self.datasets["train2"]
        # 补齐数量
        src_ids_len = len(src_trainset.ids)
        tgt_ids_len = len(tgt_trainset.ids)
        if src_ids_len < tgt_ids_len:
            num_copies = tgt_ids_len // src_ids_len
            src_trainset.ids = src_trainset.ids * num_copies
            src_trainset.ids = src_trainset.ids[:tgt_ids_len]

        return ZipDataLoader(DataLoader(src_trainset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,drop_last=True,worker_init_fn=seed_worker),
            DataLoader(tgt_trainset, batch_size=self.batch_size,
                       num_workers=self.num_workers, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
            )
    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,worker_init_fn=seed_worker)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers,worker_init_fn=seed_worker)
