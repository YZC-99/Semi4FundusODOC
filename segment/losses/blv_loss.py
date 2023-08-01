import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

# Copyright (c) OpenMMLab. All rights reserved.
import functools

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
class BlvLoss(pl.LightningModule):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4):
        super(BlvLoss, self).__init__()

        cls_num_list = torch.tensor(cls_num_list)
        cls_list = torch.tensor(cls_num_list, dtype=torch.float)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
        self.sampler = normal.Normal(0, sigma)

    def forward(self, pred, target):


        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(self.device)
        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, reduction='none')

        return loss


