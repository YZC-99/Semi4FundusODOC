from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision import transforms as T
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
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.pspnet import PSPNet
from segment.dataloader.od_oc_dataset import SupTrain
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")  # 读取图片并转换为RGB格式

        # 如果有相应的mask文件，可以在这里加载
        # mask_path = os.path.join(self.data_dir, mask_filenames[idx])
        # mask = Image.open(mask_path).convert("L")  # 读取mask并转换为灰度格式

        # 如果没有mask，你可以返回只包含image的数据
        return T.ToTensor()(image), self.image_paths[idx]


#
num_classes = 2
ckpt_path = '/root/autodl-tmp/Semi4FundusODOC/experiments/SEG/sup/random1_OD_sup/ckpt/val_mDice=0.976844-val_mIoU=0.912914-val_OD_dice_score=0.976844-val_OD_IoU=0.912914-val_OC_dice_score=0.000000-val_OC_IoU=0.000000.ckpt'
log_path = 'experiments/whole_preds'
model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
model = model_zoo['deeplabv3plus']('resnet50', num_classes)

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
model.load_state_dict(new_state_dict)
model.eval()

img_path = '/root/autodl-tmp/DDR1'

dataset = CustomDataset(img_path)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                        pin_memory=True, num_workers=1, drop_last=False)
cmap = color_map('eye')
for image, idx in dataloader:
    logits = model(image)['out']
    preds = nn.functional.softmax(logits, dim=1).argmax(1)
    pred = Image.fromarray(preds.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
    pred.putpalette(cmap)
    pred.save('whole.bmp')
    print(idx)
    break

# image_arr = np.array(Image.open('./007-0184-000.jpg'))
# # img = torch.tensor(image_arr,dtype=torch.float32).unsqueeze(dim=0)
# img = T.ToTensor()(image_arr)
# img = img.unsqueeze(dim=0)
# # img = img.permute(0,3,1,2)
# img = img
# # img = img.to(torch.cuda.FloatTensor)  # 将输入数据转换为相同的类型
# logits = model(img)['out']
# logits = model(img)['out']
# preds = nn.functional.softmax(logits, dim=1).argmax(1)

# pred = Image.fromarray(preds.squeeze(0).cpu().detach().numpy().astype(np.uint8), mode='P')
# pred.putpalette(cmap)

# pred.save('whole.bmp')

