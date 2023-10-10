from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import os
from torchmetrics import JaccardIndex,Dice
import torch.nn as nn
from copy import deepcopy
from segment.util import count_params, meanIOU, color_map
from PIL import Image
from skimage import measure,draw
import numpy as np
from segment.modules.semseg.deeplabv2 import DeepLabV2
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus,My_DeepLabV3PlusPlus
from segment.modules.semseg.segformer import SegFormer
from segment.modules.semseg.pspnet import PSPNet
from segment.dataloader.od_oc_dataset import SupTrain
from torch.utils.data import DataLoader
from thop import profile

num_classes = 3
attention = 'org'
input_tensor = torch.randn(1, 3, 256, 256)
# model = SegFormer(num_classes=num_classes, phi='b4',attention=attention)
model = DeepLabV3Plus(nclass=num_classes,backbone='resnet50')
# 计算 FLOPs
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops / 1e9} G FLOPs")  # 以十亿次FLOPs为单位打印结果
print(f"参数数量: {params / 1e6} 百万参数")  # 以百万为单位打印参数数量


