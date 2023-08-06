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
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.pspnet import PSPNet
from segment.dataloader.od_oc_dataset import SupTrain
from torch.utils.data import DataLoader

#
num_classes = 3
ckpt_path = ''

model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
model = model_zoo['deeplabv3plus']('resnet50', num_classes)

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
model.cuda()
model.eval()

dataset = SupTrain(task='od_oc',
                            name='REFUGE/cropped_sup/select_all',
                            root='./data/fundus_datasets/od_oc/SEG/',
                            mode='val',
                            size=512
                             )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                  pin_memory=True, num_workers=8, drop_last=False)
tbar = tqdm(dataloader)
od_mIoU = JaccardIndex(num_classes=2, task='multiclass').to('cuda')
oc_mIoU = JaccardIndex(num_classes=2, task='multiclass').to('cuda')
od_Dice = Dice(num_classes=1,multiclass=False).to('cuda')
oc_Dice = Dice(num_classes=1,multiclass=False).to('cuda')
cmap = color_map('eye')

# 创建csv文件
with open(os.path.join('pseudo_label_metrics.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'OD_mIoU','OD_Dice','OC_mIoU','OC_Dice'])  # 写入表头
    with torch.no_grad():
        for batch in tbar:
            img,mask,id = batch['img'],batch['mask'],batch['id']
            img = img.cuda()
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

            # od_mIoU(od_cover_gt,od_cover_preds)
            # od_Dice(od_cover_gt,od_cover_preds)
            #
            # oc_mIoU(oc_mask,oc_preds)
            # oc_Dice(oc_mask,oc_preds)

            pred = Image.fromarray(preds.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            if not os.path.exists('experiments/preds'):
                os.mkdir('experiments/preds')
            pred.save('%s/%s' % ('experiments/preds', os.path.basename(id[0].split(' ')[1])))

            # 写入csv
            writer.writerow([id[0],
                             od_mIoU(od_cover_gt,od_cover_preds),
                             od_Dice(od_cover_gt,od_cover_preds),
                             oc_mIoU(oc_mask, oc_preds),
                             oc_Dice(oc_mask, oc_preds)
                             ])
