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
# from utils.training_tricks import TTA
import utils.ttach as TTA
from utils.ttach.wrappers import SegmentationTTAWrapper


tta = True
#
num_classes = 3
ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/REFUGE/cropped_sup256x256/1200/noise/lightning_logs/version_17/ckpt/epoch=22-val_OC_dice=0.890492-val_OC_IoU=0.817475.ckpt'
log_path = 'experiments/preds'
model_zoo = {'deeplabv3plus': DeepLabV3Plus,'mydeeplabv3plusplus': My_DeepLabV3PlusPlus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
# model = model_zoo['deeplabv3plus']('resnet50', num_classes,attention='Criss_Attention_R2_V1',seghead_last=True)
model = SegFormer(num_classes=num_classes, phi='b4',attention='o1-fam-inj-skip')
sd = torch.load(ckpt_path,map_location='cpu')

if 'state_dict' in sd:
    # If 'state_dict' exists, use it directly
    sd = sd['state_dict']

new_state_dict = {}
for key, value in sd.items():
    if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
        if 'module.' + key in model.state_dict():
            # 模型在多GPU上训练并保存，加载权重时加上"module."前缀
            key = 'module.' + key
    key = key.replace('model.', '')
    new_state_dict[key] = value
model.load_state_dict(new_state_dict,strict=False)
model.to('cuda:0')
model.eval()

dataset = SupTrain(task='od_oc',
                    name='REFUGE/cropped_sup',
                    root='./data/fundus_datasets/od_oc/REFUGE/',
                    mode='test',
                    size=256
                             )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                  pin_memory=True, num_workers=8, drop_last=False)
tbar = tqdm(dataloader)

with torch.no_grad():
    for batch in tbar:
        img,mask,id = batch['img'],batch['mask'],batch['id']
        mask = mask.to('cuda:0')
        img = img.to('cuda:0')
        logits = model(img)['out']

        break

