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
from segment.modules.semseg.pspnet import PSPNet
from segment.dataloader.od_oc_dataset import SupTrain
from torch.utils.data import DataLoader

#
num_classes = 3
ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/Drishti-GS/cropped_sup512x512/res50Mydeeplabv3plusplus/random1_ODOC_backbone_pretrained_flip_rotateDCBDFCLoss/lightning_logs/version_0/ckpt/epoch=77-val_OC_dice=0.882072-val_OC_mIoU=0.868874.ckpt'
log_path = 'experiments/preds'
model_zoo = {'deeplabv3plus': DeepLabV3Plus,'mydeeplabv3plusplus': My_DeepLabV3PlusPlus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
model = model_zoo['mydeeplabv3plusplus']('resnet50', num_classes)

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
model.load_state_dict(new_state_dict)
model.to('cuda:0')
model.eval()

dataset = SupTrain(task='od_oc',
                    name='Drishti-GS/cropped_sup',
                    root='/data/fundus_datasets/od_oc/Drishti-GS/',
                    mode='test',
                    size=512
                             )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                  pin_memory=True, num_workers=8, drop_last=False)
tbar = tqdm(dataloader)
od_mIoU = JaccardIndex(num_classes=2, task='multiclass').to('cuda:0')
oc_mIoU = JaccardIndex(num_classes=2, task='multiclass').to('cuda:0')
od_Dice = Dice(num_classes=1,multiclass=False).to('cuda:0')
oc_Dice = Dice(num_classes=1,multiclass=False).to('cuda:0')
cmap = color_map('eye')
if not os.path.exists(log_path):
    os.mkdir(log_path)
# 创建csv文件
with open(os.path.join('experiments','preds_metrics.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'OD_mIoU','OD_Dice','OC_mIoU','OC_Dice'])  # 写入表头
    with torch.no_grad():
        for batch in tbar:
            img,mask,id = batch['img'],batch['mask'],batch['id']
            mask = mask.to('cuda:0')
            img = img.to('cuda:0')
            logits = model(img)['out']
            preds = nn.functional.softmax(logits, dim=1).argmax(1)

            od_preds = deepcopy(preds)
            od_mask = deepcopy(mask)
            od_preds[od_preds != 1] = 0
            od_mask[od_mask != 1] = 0

            oc_preds = deepcopy(preds)
            oc_mask = deepcopy(mask)
            oc_preds[oc_preds != 2] = 0
            oc_preds[oc_preds != 0] = 1
            oc_mask[oc_mask != 2] = 0
            oc_mask[oc_mask != 0] = 1

            od_cover_gt = od_mask + oc_mask
            od_cover_gt[od_cover_gt > 0] = 1
            od_cover_preds = od_preds + oc_preds
            od_cover_preds[od_cover_preds > 0] = 1

            # pred = Image.fromarray(preds.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
            # pred.putpalette(cmap)
            #
            # pred.save('%s/%s' % (log_path, os.path.basename(id[0].split(' ')[1])))
            # 写入csv
            writer.writerow([id[0],
                             round(od_mIoU(od_cover_gt,od_cover_preds).item()*100,2),
                             round(od_Dice(od_cover_gt,od_cover_preds).item()*100,2),
                             round(oc_mIoU(oc_mask, oc_preds).item()*100,2),
                             round(oc_Dice(oc_mask, oc_preds).item()*100,2)
                             ])
