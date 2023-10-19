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
from segment.modules.semseg.trans_unet.vit_seg_modeling import VisionTransformer as ViT_seg
from segment.modules.semseg.trans_unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

tta = False
num_classes = 3
root_path = '/root/autodl-tmp'
results_root_path = 'results'
current_ex = 'Drishti-GS'
current_result_dir = os.path.join(results_root_path,current_ex)
if not os.path.exists(current_result_dir):
    os.makedirs(current_result_dir)

current_result_preds_dir = os.path.join(current_result_dir,'preds')
if not os.path.exists(current_result_preds_dir):
    os.makedirs(current_result_preds_dir)

folders_path = {i:os.path.join(root_path,i) for i in os.listdir(root_path)}
for k,v in folders_path.items():
    folders_path[k] = os.path.join(v,os.listdir(v)[0])
    ckpt_path = folders_path[k]
    sd = torch.load(ckpt_path, map_location='cpu')
    if k == 'proposed':
        model = SegFormer(num_classes=num_classes, phi='b4',attention='dec_transpose_FAMIFM_CBAM_CCA')
    elif k == 'Segformer':
        model = SegFormer(num_classes=num_classes, phi='b4', attention='org')
    elif k == 'Transunet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes
        model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes)

    model.load_state_dict(sd,strict=False)
    model.to('cuda:0')
    model.eval()

    dataset = SupTrain(task='od_oc',
                        name='{}/cropped_sup'.format(current_ex),
                        root='./data/fundus_datasets/od_oc/{}/'.format(current_ex),
                        mode='test',
                        size=256
                             )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                      pin_memory=True, num_workers=8, drop_last=False)
    tbar = tqdm(dataloader)
    od_mIoU = JaccardIndex(num_classes=2, task='binary',average='micro').to('cuda:0')
    oc_mIoU = JaccardIndex(num_classes=2, task='binary',average='micro').to('cuda:0')
    od_Dice = Dice(num_classes=1,multiclass=False,average='samples').to('cuda:0')
    oc_Dice = Dice(num_classes=1,multiclass=False,average='samples').to('cuda:0')
    cmap = color_map('eye')
    # 创建csv文件
    with open(os.path.join(current_result_dir,'preds_metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'OD_mIoU','OD_Dice','OC_mIoU','OC_Dice'])  # 写入表头
        with torch.no_grad():
            for batch in tbar:
                img,mask,id = batch['img'],batch['mask'],batch['id']
                mask = mask.to('cuda:0')
                img = img.to('cuda:0')
                logits = model(img)['out']

                # preds = nn.functional.softmax(logits, dim=1).argmax(1)
                #-------------
                probs = nn.functional.softmax(logits, dim=1)
                threshold = 0.5
                thresholded_preds = (probs >= threshold).float()
                preds = torch.argmax(thresholded_preds, dim=1)
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

                pred.save('%s/%s' % (current_result_preds_dir, os.path.basename(id[0].split(' ')[1])))
                # 写入csv
                writer.writerow([id[0],
                                 round(od_mIoU(od_cover_gt,od_cover_preds).item()*100,2),
                                 round(od_Dice(od_cover_gt,od_cover_preds).item()*100,2),
                                 round(oc_mIoU(oc_mask, oc_preds).item()*100,2),
                                 round(oc_Dice(oc_mask, oc_preds).item()*100,2)
                                 ])
            writer.writerow(["avg",
                             round(od_mIoU.compute().item()*100,2),
                             round(od_Dice.compute().item()*100,2),
                             round(oc_mIoU.compute().item()*100,2),
                             round(oc_Dice.compute().item()*100,2)
                             ])
