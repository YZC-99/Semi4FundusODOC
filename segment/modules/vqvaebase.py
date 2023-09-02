import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import normal
from torch.optim import SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from segment.util import meanIOU
from segment.losses.vae.loss import vqvae_loss
from typing import List,Tuple, Dict, Any, Optional
import pytorch_lightning as pl
import torchmetrics
from torchvision import transforms as T
from segment.modules.VQVAE.vqvae import ResVectorQuantizedVAE
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.general import initialize_from_config

from segment.modules.pyl_utils import *


class Base(pl.LightningModule):
    def __init__(self,
                 model:str,
                 backbone: str,
                 num_classes: int,
                 in_channels: int,
                 dim: int,
                 z_dim: int,
                 cfg,
                 ):
        super(Base, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = ResVectorQuantizedVAE(backbone,in_channels,dim,z_dim)
        self.init_from_ckpt = init_from_ckpt
        self.loss = vqvae_loss
        
        self.gray2rgb = gray2rgb


        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = T.Normalize((0.0,), (1.0,))(x)
        out = self.model(x)
        return out

    def get_mask_as_imput(self,batch):
        y = batch['mask']
        y = y.unsqueeze(1).float() / 2.0
        return y


    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        y = self.get_mask_as_imput(batch)
        output = self(y)
        loss = self.loss(y,output)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,)
        return loss


    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        y = self.get_mask_as_imput(batch)
        output = self(y)
        loss = self.loss(y,output)
        return {'val_loss':loss,'preds':output['x_tilde'],'y':y}

    def validation_step_end(self, outputs):
        loss = outputs['val_loss']
        self.log("val/loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True,
                      )
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=False, on_epoch=True,
                      sync_dist=True,
                      )
    def setup(self, stage: str):
        if stage == 'fit':
            limit_batches = self.trainer.limit_train_batches
            batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
            batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
            num_devices = max(1, self.trainer.num_devices)
            effective_accum = self.trainer.accumulate_grad_batches * num_devices
            self.train_steps = (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        # total_iters = self.trainer.max_steps
        total_iters = self.train_steps
        optimizers = [SGD(self.model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-4)]
        # lambda_lr = lambda iters: lr * (1 - iters / total_iters) ** 0.9
        # scheduler = LambdaLR(optimizers[0],lr_lambda=lambda_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizers[0], total_iters=total_iters,power=0.9)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]

        print(">>>>>>>>>>>>>total iters:{}<<<<<<<<<<<<<<<<".format(total_iters))
        return optimizers, schedulers


    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        y = self.get_mask_as_imput(batch)
        out = self(y)['x_tilde']
        y_color, out_color = self.gray2rgb(self, y.squeeze(1) * 2.0, out.squeeze(1))
        log["label"] = y_color
        log["predict"] = out
        print(torch.unique(out))
        return log
