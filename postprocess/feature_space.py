from segment.modules.semibase import Base
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment.util import  color_map
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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
backbone_feat, logits = output['backbone_features'], output['out']

b,c,h,w = logits.shape
feat = logits.reshape(b*h*w, c)
label = mask_tensor.reshape(b*h*w)

pca = PCA(n_components=2)
visual_feats = pca.fit_transform(feat.detach().numpy().astype(np.float32))
plt.scatter(visual_feats[:,0], visual_feats[:,1], c=label)
plt.legend({0,1,2})
# 1. 拟合模型
lr = LogisticRegression()
lr.fit(visual_feats, label)

# 2. 预测
boundary = lr.predict(visual_feats)
plt.contour(boundary)
plt.show()

# # pred = torch.argmax(logits, dim=1)
# pred = nn.functional.softmax(logits, dim=1).argmax(1)
# pred = pred.squeeze()
#
# preds_arr = pred.numpy().astype(np.uint8)
# pred = Image.fromarray(preds_arr, mode='P')
# pred.putpalette(cmap)
# pred.save('./T0001_preds.bmp')



# print(pred.shape)