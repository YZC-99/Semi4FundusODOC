from segment.modules.semseg.segformer import SegFormer
import torch
from PIL import Image
import torchvision.transforms as T
from segment.util import count_params, meanIOU, color_map
import numpy as np

cmap = color_map("c")
num_classes = 3
backbone = "b2"
model = SegFormer(num_classes=num_classes, phi=backbone,attention='subv1',pretrained=True)






