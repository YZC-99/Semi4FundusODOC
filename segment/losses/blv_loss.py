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
class BlvLoss(nn.Module):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4):
        super(BlvLoss, self).__init__()

        cls_num_list = torch.tensor(cls_num_list)
        cls_list = torch.tensor(cls_num_list, dtype=torch.float)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = 'BlvLoss'

    def forward(self, pred, target,viariation):
        # viariation = self.sampler.sample(pred.shape).clamp(-1, 1)
        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)
        loss = F.cross_entropy(pred, target, reduction='none')
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

