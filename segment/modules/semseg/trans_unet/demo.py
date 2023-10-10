from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
import torch

vit_name = 'R50-ViT-B_16'
config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = 3

net = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes)
# net.load_from(weights=np.load(config_vit.pretrained_path))
input = torch.randn(2,3,256,256,dtype=torch.float32)
out = net(input)
print(out.shape)