import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import sys
import os
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
from segment.demo.attentions.cbam import CBAMBlock
from segment.modules.semseg.nn import CrissCrossAttention
from PIL import Image

class AxialDAM(nn.Module):
    def __init__(self,in_channels=512):
        super(AxialDAM, self).__init__()

        self.dam1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                  dilation=1, bias=False)
        self.dam2 = nn.Conv2d(in_channels, in_channels, 3, padding=2,
                  dilation=2, bias=False)
        self.dam4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                  dilation=4, bias=False)
        self.dam8 = nn.Conv2d(in_channels, in_channels, 3, padding=8,
                  dilation=8, bias=False)

        self.cbam = CBAMBlock(channel=512*5, reduction=16, kernel_size=7)

        self.block = nn.Sequential(nn.Conv2d(in_channels*5, in_channels, 3, padding=1,
                                    bias=False),
                          nn.BatchNorm2d(in_channels),
                          nn.ReLU(True))

    def forward(self,x):
        d1 = self.dam1(x)
        d2 = self.dam2(x)
        d4 = self.dam4(x)
        d8 = self.dam8(x)

        _d = torch.cat([x,d1,d2,d4,d8],dim=1)

        # _d = F.interpolate(_d, (65, 65), mode="bilinear", align_corners=True)
        _c = self.cbam(_d)
        _c = self.block(_c)
        # _c = F.interpolate(_c, (64, 64), mode="bilinear", align_corners=True)
        return _c


class DAM(nn.Module):
    def __init__(self,in_channels=512):
        super(DAM, self).__init__()

        self.dam1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                  dilation=1, bias=False)
        self.dam2 = nn.Conv2d(in_channels, in_channels, 3, padding=2,
                  dilation=2, bias=False)
        self.dam4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                  dilation=4, bias=False)
        self.dam8 = nn.Conv2d(in_channels, in_channels, 3, padding=8,
                  dilation=8, bias=False)

        self.cbam = CBAMBlock(channel=512*5, reduction=16, kernel_size=7)

        self.block = nn.Sequential(nn.Conv2d(in_channels*5, in_channels, 3, padding=1,
                                    bias=False),
                          nn.BatchNorm2d(in_channels),
                          nn.ReLU(True))

    def forward(self,x):
        d1 = self.dam1(x)
        d2 = self.dam2(d1)
        d4 = self.dam4(d2)
        d8 = self.dam8(d4)

        _d = torch.cat([x,d1,d2,d4,d8],dim=1)

        # _d = F.interpolate(_d, (65, 65), mode="bilinear", align_corners=True)
        _c = self.cbam(_d)
        _c = self.block(_c)
        # _c = F.interpolate(_c, (64, 64), mode="bilinear", align_corners=True)
        return _c

class DAM_criss(nn.Module):
    def __init__(self,in_channels=512):
        super(DAM_criss, self).__init__()

        self.dam1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                  dilation=1, bias=False)
        self.dam2 = nn.Conv2d(in_channels, in_channels, 3, padding=2,
                  dilation=2, bias=False)
        self.dam4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                  dilation=4, bias=False)
        self.dam8 = nn.Conv2d(in_channels, in_channels, 3, padding=8,
                  dilation=8, bias=False)

        self.cca1 = CrissCrossAttention(512*5)
        self.cca2 = CrissCrossAttention(512*5)

        self.block = nn.Sequential(nn.Conv2d(in_channels*5, in_channels, 3, padding=1,
                                    bias=False),
                          nn.BatchNorm2d(in_channels),
                          nn.ReLU(True))

    def forward(self,x):
        d1 = self.dam1(x)
        d2 = self.dam2(d1)
        d4 = self.dam4(d2)
        d8 = self.dam8(d4)

        _d = torch.cat([x,d1,d2,d4,d8],dim=1)

        # _d = F.interpolate(_d, (65, 65), mode="bilinear", align_corners=True)
        _c = self.cca1(_d)
        _c = self.cca2(_c)
        _c = self.block(_c)
        # _c = F.interpolate(_c, (64, 64), mode="bilinear", align_corners=True)
        return _c

if __name__ == '__main__':
    input = torch.randn((2,512,8,8))
    # dam = DAM(512)
    dam = AxialDAM(512)
    out = dam(input)
    print(out.shape)
