from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import os
from torchmetrics import JaccardIndex, Dice
import torch.nn as nn
from copy import deepcopy
from segment.util import count_params, meanIOU, color_map
from PIL import Image
from skimage import measure, draw
import numpy as np
from segment.modules.semseg.deeplabv2 import DeepLabV2
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus, My_DeepLabV3PlusPlus
from segment.modules.semseg.segformer import SegFormer
from segment.modules.semseg.pspnet import PSPNet
from segment.dataloader.od_oc_dataset import SupTrain
from torch.utils.data import DataLoader
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import skimage.transform


def polar_to_cartesian(polar_img):
    # 确保图像是 float 类型
    polar_img_float = polar_img.astype(np.float32)

    # 计算用于极坐标变换的值，使用图像的短边作为半径
    value = min(polar_img_float.shape[0], polar_img_float.shape[1]) / 2

    # 执行极坐标到笛卡尔坐标的逆变换
    cartesian_img_cv = cv2.linearPolar(polar_img_float, (polar_img_float.shape[1] / 2, polar_img_float.shape[0] / 2),
                                       value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    return cartesian_img_cv


#
num_classes = 3
ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/REFUGE/cropped_sup256x256/polar_400/noise/lightning_logs/version_2/ckpt/epoch=75-val_OC_dice=0.946621-val_OC_IoU=0.900515.ckpt'
log_path = 'experiments/preds'
model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'mydeeplabv3plusplus': My_DeepLabV3PlusPlus, 'pspnet': PSPNet,
             'deeplabv2': DeepLabV2}
# model = model_zoo['deeplabv3plus']('resnet50', num_classes,attention='Criss_Attention_R2_V1',seghead_last=True)
model = SegFormer(num_classes=num_classes, phi='b4', attention='o1')
sd = torch.load(ckpt_path, map_location='cpu')

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
model.load_state_dict(new_state_dict, strict=False)
model.to('cuda:0')
model.eval()

dataset = SupTrain(task='od_oc',
                   name='REFUGE/cropped_polared/400',
                   root='./data/fundus_datasets/od_oc/REFUGE/',
                   mode='test',
                   size=256
                   )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                        pin_memory=True, num_workers=8, drop_last=False)
tbar = tqdm(dataloader)
od_mIoU = JaccardIndex(num_classes=2, task='binary', average='micro').to('cuda:0')
oc_mIoU = JaccardIndex(num_classes=2, task='binary', average='micro').to('cuda:0')
od_Dice = Dice(num_classes=1, multiclass=False, average='samples').to('cuda:0')
oc_Dice = Dice(num_classes=1, multiclass=False, average='samples').to('cuda:0')
cmap = color_map('eye')
if not os.path.exists(log_path):
    os.mkdir(log_path)
# 创建csv文件
with open(os.path.join('experiments', 'preds_metrics.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'OD_mIoU', 'OD_Dice', 'OC_mIoU', 'OC_Dice'])  # 写入表头
    with torch.no_grad():
        for batch in tbar:
            img, mask, id = batch['img'], batch['mask'], batch['id']
            mask = mask.to('cuda:0')
            img = img.to('cuda:0')

            logits = model(img)['out']
            # preds = nn.functional.softmax(logits, dim=1).argmax(1)
            # -------------
            probs = nn.functional.softmax(logits, dim=1)
            threshold = 0.5
            thresholded_preds = (probs >= threshold).float()
            preds = torch.argmax(thresholded_preds, dim=1)
            # 转换为笛卡尔坐标
            center = (preds.shape[2] / 2, preds.shape[1] / 2)  # 假设 preds 的维度是 [channels, height, width]
            max_radius = np.sqrt(center[0] ** 2 + center[1] ** 2)

            preds = polar_to_cartesian(preds.squeeze(0).cpu().numpy())
            mask = polar_to_cartesian(mask.squeeze(0).cpu().numpy())
            preds = torch.tensor(preds).unsqueeze(0).to('cuda:0')
            mask = torch.tensor(mask).unsqueeze(0).to('cuda:0')
            # -------------

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

            pred = Image.fromarray(preds.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (log_path, os.path.basename(id[0].split(' ')[1])))
            # 写入csv
            writer.writerow([id[0],
                             round(od_mIoU(od_cover_gt, od_cover_preds).item() * 100, 2),
                             round(od_Dice(od_cover_gt, od_cover_preds).item() * 100, 2),
                             round(oc_mIoU(oc_mask, oc_preds).item() * 100, 2),
                             round(oc_Dice(oc_mask, oc_preds).item() * 100, 2)
                             ])
        writer.writerow(["avg",
                         round(od_mIoU.compute().item() * 100, 2),
                         round(od_Dice.compute().item() * 100, 2),
                         round(oc_mIoU.compute().item() * 100, 2),
                         round(oc_Dice.compute().item() * 100, 2)
                         ])
