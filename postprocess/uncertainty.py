
from segment.modules.semibase import Base
from segment.dataloader.od_oc_dataset import SemiUabledTrain
from segment.modules.semseg.deeplabv3plus import DualDeepLabV3Plus
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import math
from segment.util import  color_map
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import cleanlab
from sklearn.manifold import TSNE
from cleanlab.pruning import get_noise_indices

num_classes = 2
ckpt_path = '../temp/panaro.ckpt'

cmap = color_map('eye')

model = DualDeepLabV3Plus('resnet50',num_classes)

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


model.load_state_dict(new_state_dict)
model.eval()
model.to('cuda:0')


dataset = SemiUabledTrain(task='od_oc',
                         name='SEG/semi/50',
                         root='./data/fundus_datasets/od_oc/SEG/',
                         mode='pseudo',
                         size=512,
                         unlabeled_id_path='dataset/SEG/semi/50/random1/unlabeled.txt',
                         pseudo_mask_path='path',
                         aug=None)
dataloader = DataLoader(dataset,batch_size=1)
for batch in dataloader:
    x = batch['img'].to('cuda:0')
    mask = batch['mask'].to('cuda:0')

    out_od = model(x)['out1']

    # 计算uncertainty
    T = 4
    _, _, w, h = batch.shape
    volume_batch_r = batch.repeat(2, 1, 1, 1)
    stride = volume_batch_r.shape[0] // 2
    preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
    for i in range(T // 2):
        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
        with torch.no_grad():
            preds[2 * stride * i:2 * stride * (i + 1)] = model(ema_inputs)['out1']
    preds = F.softmax(preds, dim=1)
    preds = preds.reshape(T, stride, num_classes, w, h)
    preds = torch.mean(preds, dim=0)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
    uncertainty = uncertainty / math.log(2)  # normalize uncertainty, cuz ln2 is the max value

    #
    pred_soft_np = torch.softmax(out_od, dim=1).cpu().detach().numpy()
    masks_np = mask.detach().cpu().numpy()
    preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
    preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
    preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
    preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
    masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
    noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                               prune_method='prune_by_noise_rate', n_jobs=1)
    confident_maps_np = np.squeeze(noise.reshape(-1, 512, 512).astype(np.uint8))
    print(confident_maps_np.shape)
    break