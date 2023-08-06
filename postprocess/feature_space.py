from segment.modules.semibase import Base
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
from segment.util import  color_map
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

num_classes = 3
ckpt_path = '../temp/val_OD_dice=0.847466.ckpt'

cmap = color_map('eye')

model = DeepLabV3Plus('resnet50',num_classes)

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


image_path = './T0001.jpg'
mask_path = './T0001.bmp'
image_tensor = T.ToTensor()(Image.open(image_path)).unsqueeze(dim=0)
mask_tensor = T.ToTensor()(Image.open(mask_path))


output = model(image_tensor)
'''
logits的形状为1*3*512*512,其中3代表语义类别数，现在需要对logits进行t-sne降维.
降维的目的是想要展示语义分割网络最后一层输出的logits通过降维后的二维可视化，
比如当前是三个类别，那么我希望降维后就应该呈现出三个点，这三个点分别代表语义分割的三个类别，
请你写出python代码
'''
backbone_feat, logits = output['backbone_features'], output['out']

# Take the first sample in batch
logits_sample = logits[0].detach().numpy()
# 将logits的形状转换为(512*512, 3)
logits_reshaped = logits_sample.reshape(-1, 3)

# 使用t-SNE进行降维，降维后的结果为(512*512, 2)
tsne = TSNE(n_components=2)
logits_tsne = tsne.fit_transform(logits_reshaped)