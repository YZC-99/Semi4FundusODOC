import torch
from mmcv.cnn import build_norm_layer
from torch import nn

class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.conv = nn.Conv2d(3, 3, padding=1)


    def forward(self, skip,x_loacal,x):
        pass