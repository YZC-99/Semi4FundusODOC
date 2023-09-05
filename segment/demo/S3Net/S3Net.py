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

from segment.demo.S3Net.utils import dice_metric, xor_metric, hm_metric, create_mask
from segment.demo.S3Net.utils import read_image, compute_sobel_gradients
from model_utils import AttentionModule, DeformableConv
from PIL import Image

class S3Net(nn.Module):
    def __init__(self, in_channels, nChannel, nConv, BHW=(1, 256, 256), color=False):
        """
        Initialize the S^3Net model.

        :param in_channels: Number of input channels.
        :param nChannel: Number of output channels.
        :param nConv: Number of convolutional layers.
        :param BHW: (Batch size, Height of the input image, Width of the input image).
        :param color: Flag to indicate whether to use color processing (good for binary segmentation).
        :type in_channels: int
        :type nChannel: int
        :type nConv: int
        :type BHW: tuple (int, int, int)
        :type color: bool
        """
        super(S3Net, self).__init__()

        self.color = color

        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(in_channels, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)

        # Attention Convolution Layers
        self.conv2 = nn.ModuleList([AttentionModule(nChannel) for _ in range(nConv)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(nChannel) for _ in range(nConv)])

        # Deformable Convolution Layer
        self.convdeform = DeformableConv(nChannel, batch=BHW[0], height=BHW[1], width=BHW[2])
        self.bn3 = nn.BatchNorm2d(nChannel)

        # Additional Convolution Layer
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(nChannel)

        # Auxiliary Convolution for the Affine Transformation Branch
        self.conv_aux = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn_aux = nn.BatchNorm2d(nChannel)

        if self.color:
            # Linear Mapping for Color Processing
            self.map = nn.Linear(3, nChannel, bias=False)
            self.cf = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
            self.bf = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        inp = x.clone()

        # Initial Convolution
        x = self.bn1(self.conv1(x))

        # Attention Convolution Blocks
        for i in range(len(self.conv2)):
            x = self.bn2[i](self.conv2[i](x))

        # Auxiliary Convolution for the Affine Transformation Branch
        x_aux = self.bn_aux(self.conv_aux(x))

        # Deformable Convolution
        x = self.bn3(self.convdeform(x))

        # Additional Convolution Layer
        x = self.bn4(self.conv3(x))

        if self.color:
            # Color Processing
            x1 = x.view(x.size(0), x.size(1), -1)
            x1 = F.normalize(x1, dim=-1)
            x1 = torch.matmul(x1, F.softmax(x1.transpose(-2, -1)))

            inp = inp.view(inp.size(0), inp.size(1), -1)
            inp = self.map(inp.permute(0, 2, 1)).permute(0, 2, 1)

            inp = F.normalize(inp, dim=-1)
            inp = torch.matmul(inp, F.softmax(inp.transpose(-2, -1)))
            att = inp + x1

            v = x.view(x.size(0), x.size(1), -1)
            xatt = torch.matmul(att, v)
            xatt = xatt.view(x.size(0), x.size(1), x.size(2), x.size(3))
            x = self.bf(self.cf(xatt))

        # Affine Transformation
        x_aux = torchvision.transforms.functional.affine(x_aux, angle=90, translate=(0, 0),
                                                         scale=1.0, shear=0.0)

        return x, x_aux
if __name__ == '__main__':
    img_path = 'T0001.jpg'
    img_rgb = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) / 255.
    sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)

    print(sobelx)
