from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import os
from torchmetrics import JaccardIndex,Dice

from copy import deepcopy
from segment.util import count_params, meanIOU, color_map
from PIL import Image
from skimage import measure,draw
import numpy as np
from segment.modules.semseg.deeplabv2 import DeepLabV2
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.pspnet import PSPNet
import cv2

def mask_to_boundary(mask,boundary_size = 3, dilation_ratio=0.005):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((boundary_size, boundary_size), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def label(dataloader, ckpt_path,cfg):
    print(">>>>>>>>>>>>>正在推理伪标签<<<<<<<<<<<<<<<")
    num_classes = cfg.MODEL.NUM_CLASSES

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo['deeplabv3plus']('resnet50',num_classes)

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
    tbar = tqdm(dataloader)
    metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)
    dice = Dice(num_classes=cfg.MODEL.NUM_CLASSES, average='macro')
    cmap = color_map(cfg.MODEL.dataset)

    # 创建csv文件
    if not os.path.exists(cfg.MODEL.logs_path):
       os.makedirs(cfg.MODEL.logs_path)
    with open(os.path.join(cfg.MODEL.logs_path,'pseudo_label_metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'IoU','Dice'])  # 写入表头
        with torch.no_grad():
            for batch in tbar:
                img,mask,id = batch['img'],batch['mask'],batch['id']
                img = img.cuda()
                pred = model(img)['out']
                pred = torch.argmax(pred, dim=1).cpu()
                pred_arr = pred.squeeze(0).numpy().astype(np.uint8)

                if cfg.MODEL.label_minus_boundary != 0:
                    od_pred = np.zeros_like(pred_arr)
                    od_pred[pred_arr > 0] = 1

                    oc_pred = np.zeros_like(pred_arr)
                    oc_pred[pred_arr == 2] = 1

                    od_pred_boundary = mask_to_boundary(od_pred, boundary_size=cfg.MODEL.label_minus_boundary)
                    oc_pred_boundary = mask_to_boundary(od_pred, boundary_size=cfg.MODEL.label_minus_boundary)
                    pred_arr = oc_pred + od_pred - od_pred_boundary - oc_pred_boundary

                metric.add_batch(pred.numpy(), mask.numpy())
                dice_score = dice(pred.cpu(), mask.cpu())
                mIOU = metric.evaluate()[-1]

                pred = Image.fromarray(pred_arr, mode='P')
                pred.putpalette(cmap)
                pred.save('%s/%s' % (cfg.MODEL.pseudo_mask_path, os.path.basename(id[0].split(' ')[1].replace('.jpg','.png'))))

                # 写入csv
                writer.writerow([id[0],mIOU,dice_score.item()])
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
