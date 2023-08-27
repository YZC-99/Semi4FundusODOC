import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import os

from copy import deepcopy
from segment.util import count_params, meanIOU, color_map
from PIL import Image
from skimage import measure,draw
import numpy as np
from segment.modules.semseg.deeplabv2 import DeepLabV2
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.pspnet import PSPNet
from segment.losses.cbl import CBL,Fast_CBL,Faster_CBL
from segment.losses.abl import ABL
import time
import cv2




num_classes = 3
device = 'cuda:0'
input = torch.randn(2,3,256,256).to(device)
label = torch.zeros(2,256,256).to(device)
label[0, 5:100] = 1.0
label[0, 100:200] = 1.0
label[0, 205:300] = 2.0
label[0, 300:400] = 2.0
cbl = CBL(num_classes)
fast_cbl = Fast_CBL(num_classes)
faster_cbl = Faster_CBL(num_classes)
abl = ABL()
# loss_dict = {'initcbl':cbl,'cbl':cbl,'fast_cbl':fast_cbl,'faster_cbl':faster_cbl,'abl':abl}
loss_dict = {'initcbl':cbl,'cbl':cbl,'cbl1':cbl,'cbl2':cbl,'abl':abl}

model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
model = model_zoo['deeplabv3plus']('resnet50', num_classes)
model.cuda()

for k,v in loss_dict.items():
    start_time_forward = time.time()
    outputs = model(input)
    end_time_forward = time.time()
    forward_time = (end_time_forward - start_time_forward) * 1000
    print("Forward Pass Time:", forward_time, "ms")
    start_time_loss = time.time()
    if k == 'cbl':
        loss = cbl(outputs,label,model.classifier.weight,model.classifier.bias)
    elif k == 'fast_cbl':
        loss = fast_cbl(outputs, label, model.classifier.weight, model.classifier.bias)
    elif k == 'faster_cbl':
        loss = faster_cbl(outputs, label, model.classifier.weight, model.classifier.bias)
    else:
        loss = abl(outputs['out'],label)
    end_time_loss = time.time()
    loss_time = (end_time_loss - start_time_loss) * 1000

    start_time_backward = time.time()
    loss.backward()
    end_time_backward = time.time()
    backward_time = (end_time_backward - start_time_backward) * 1000
    print("{} Loss Calculation Time:".format(k), loss_time, "ms")
    print("{} Backward Pass Time:".format(k), backward_time, "ms")
    print("{}_loss:{}".format(k,loss))
    print('=========================================================')
