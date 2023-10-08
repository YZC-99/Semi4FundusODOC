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

# 帮我修改并完善这个代码：
"""
输入进来的polar_img是一个灰度图像，且只有像素值0，1，2，
经过极坐标转化到笛卡尔坐标后的图片cartesian_img_cv
其中0是背景，1和2的区域形成了一个同心圆，2在里面
现在我希望对图像进行如下操作：
令整个图像中像素值大于0的区域为1，得到od_mask,令图像中像素值为1的区域全为0，像素值为2的区域为1，其余为0，得到oc_mask
选择od_mask图像中像素值为1的最大联通区域，并利用椭圆拟合生成最后的分割结果，如果拟合出来的椭圆有残缺，如果拟合后的图像，除了有椭圆以外的部分，则消除，然后被拟合出来的椭圆如果有残缺，则根据椭圆轮廓使用1来填充，得到pred_od
对oc_mask使用同样的方法，得到pred_oc，然后将pred_oc和pred_od相加，得到结果
"""


def polar_to_cartesian(polar_img):
    polar_img_float = polar_img.astype(np.float32)
    value = min(polar_img_float.shape[0], polar_img_float.shape[1]) / 2

    cartesian_img_cv = cv2.linearPolar(polar_img_float,
                                       (polar_img_float.shape[1] / 2, polar_img_float.shape[0] / 2),
                                       value,
                                       cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP).astype(np.uint8)

    od_mask = (cartesian_img_cv > 0).astype(np.uint8)
    oc_mask = np.where(cartesian_img_cv == 2, 1, 0).astype(np.uint8)

    pred_od = generate_ellipse_mask(od_mask)
    pred_oc = generate_ellipse_mask(oc_mask)

    final_result = pred_od + pred_oc
    return final_result


def generate_ellipse_mask(mask_img):
    # 选择最大的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area_img = np.zeros_like(mask_img)
    largest_area_img[labels == largest_label] = 1

    # 找到这个连通组件的轮廓
    contours, _ = cv2.findContours(largest_area_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对轮廓进行椭圆拟合
    ellipse = cv2.fitEllipse(contours[0])
    ellipse_img = np.zeros_like(largest_area_img)
    cv2.ellipse(ellipse_img, ellipse, 1, -1)  # 填充椭圆

    return ellipse_img


#
num_classes = 3
ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/REFUGE/cropped_sup512x512/flip_scale/lightning_logs/version_1/ckpt/epoch=20-val_OC_dice=0.936030-val_OC_IoU=0.885466.ckpt'
log_path = 'experiments/preds'
polar_log_path = 'experiments/polar_preds'
mask_path = 'experiments/masks'
polar_mask_path = 'experiments/polar_masks'
model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'mydeeplabv3plusplus': My_DeepLabV3PlusPlus, 'pspnet': PSPNet,
             'deeplabv2': DeepLabV2}
# model = model_zoo['deeplabv3plus']('resnet50', num_classes,attention='Criss_Attention_R2_V1',seghead_last=True)
model = SegFormer(num_classes=num_classes, phi='b2', attention='o1-fam-inj-skip')
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
                   name='REFUGE/cropped_polared/400/sample1',
                   root='./data/fundus_datasets/od_oc/REFUGE/',
                   mode='test',
                   size=512
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
if not os.path.exists(mask_path):
    os.mkdir(mask_path)
if not os.path.exists(polar_log_path):
    os.mkdir(polar_log_path)
if not os.path.exists(polar_mask_path):
    os.mkdir(polar_mask_path)

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

            polarpreds = Image.fromarray(preds.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
            polarpreds.putpalette(cmap)
            polarpreds.save('%s/%s' % (polar_log_path, os.path.basename(id[0].split(' ')[1])))

            polarmasks = Image.fromarray(mask.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
            polarmasks.putpalette(cmap)
            polarmasks.save('%s/%s' % (polar_mask_path, os.path.basename(id[0].split(' ')[1])))

            preds = polar_to_cartesian(preds.squeeze(0).detach().cpu().numpy())
            mask = polar_to_cartesian(mask.squeeze(0).detach().cpu().numpy())
            preds = torch.tensor(preds).unsqueeze(0).to('cuda:0').to(torch.int)
            mask = torch.tensor(mask).unsqueeze(0).to('cuda:0').to(torch.int)
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

            mask = Image.fromarray(mask.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
            mask.putpalette(cmap)
            mask.save('%s/%s' % (mask_path, os.path.basename(id[0].split(' ')[1])))
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
