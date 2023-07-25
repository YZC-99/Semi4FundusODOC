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

                metric.add_batch(pred.numpy(), mask.numpy())
                dice_score = dice(pred.cpu(), mask.cpu())
                mIOU = metric.evaluate()[-1]

                pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
                pred.putpalette(cmap)
                pred.save('%s/%s' % (cfg.MODEL.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

                # 写入csv
                writer.writerow([id[0],mIOU,dice_score.item()])
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
