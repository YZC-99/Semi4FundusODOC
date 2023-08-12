from segment.modules.semibase import Base
from segment.dataloader.od_oc_dataset import SemiUabledTrain
from segment.modules.semseg.deeplabv3plus import DualDeepLabV3Plus, DeepLabV3Plus
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import math
from segment.util import color_map
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import cleanlab
from sklearn.manifold import TSNE
from cleanlab.pruning import get_noise_indices


def get_sd(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    if 'state_dict' in sd:
        # If 'state_dict' exists, use it directly
        sd = sd['state_dict']
    new_state_dict = {}
    for key, value in sd.items():
        # if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
        #     if 'module.' + key in model.state_dict():
        #         # 模型在多GPU上训练并保存，加载权重时加上"module."前缀
        #         key = 'module.' + key
        key = key.replace('model.', '')
        new_state_dict[key] = value
    return new_state_dict


num_classes = 3
pre_ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None/ckpt/epoch=166-val_OC_mIoU=0.879493.ckpt'
now_ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None/ckpt/epoch=185-val_OD_dice=0.962586.ckpt'

cmap = color_map('eye')
pre_model = DeepLabV3Plus('resnet50', num_classes)
pre_model.load_state_dict(get_sd(pre_ckpt_path))
pre_model.eval()
pre_model.to('cuda:0')

now_model = DeepLabV3Plus('resnet50', num_classes)
now_model.load_state_dict(get_sd(now_ckpt_path))
now_model.eval()
now_model.to('cuda:0')

dataset = SemiUabledTrain(task='od_oc',
                          name='SEG/semi/90',
                          root='./data/fundus_datasets/od_oc/SEG/',
                          mode='label',
                          size=512,
                          unlabeled_id_path='dataset/SEG/semi/90/random1/unlabeled.txt',
                          pseudo_mask_path='/root/autodl-tmp/Semi4FundusODOC/experiments/SEG/semi/50/ODOC_semi50/pseudo_masks',
                          aug=None)
dataloader = DataLoader(dataset, batch_size=1)
for idx, batch in enumerate(dataloader):
    x = batch['img'].to('cuda:0')
    mask = batch['mask'].to('cuda:0')

    pre_out = pre_model(x)['out']
    now_out = now_model(x)['out']

    #
    pre_soft_np = torch.softmax(pre_out, dim=1).cpu().detach().numpy()
    now_soft_np = torch.softmax(now_out, dim=1).argmax(1).cpu().detach().numpy()
    #     masks_np = now_soft_np
    masks_np = mask.detach().cpu().numpy()
    preds_softmax_np_accumulated = np.swapaxes(pre_soft_np, 1, 2)
    preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
    preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
    preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)

    masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
    noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                               prune_method='prune_by_noise_rate', n_jobs=1)
    confident_maps_np = np.squeeze(noise.reshape(-1, 512, 512).astype(np.uint8))
    print(np.unique(confident_maps_np))
    print(np.sum(confident_maps_np == 1))

    confident = Image.fromarray(confident_maps_np, mode='P')
    confident.putpalette(cmap)
    confident.save('./postprocess/confident_map/{}.png'.format(idx))
