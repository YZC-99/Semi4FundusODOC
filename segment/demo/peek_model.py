import torch
from segment.modules.semseg.deeplabv2 import DeepLabV2
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.pspnet import PSPNet
from segment.modules.semseg.unet import UNet

num_classes = 3

model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
res101model = model_zoo['deeplabv3plus']('resnet101', num_classes)
res50model = model_zoo['deeplabv3plus']('resnet50', num_classes)
res34model = model_zoo['deeplabv3plus']('resnet34', num_classes)
res18model = model_zoo['deeplabv3plus']('resnet18', num_classes)

unet = UNet(num_classes)

input = torch.randn(2,3,256,256)
res101output = res101model(input)
res50output = res50model(input)
res34output = res34model(input)
res18output = res18model(input)
unet_output = unet(input)

backbone_res101_feat = res101output['backbone_features'] # (2,2048,32,32)
backbone_res50_feat = res50output['backbone_features'] # (2,2048,32,32)
backbone_res34_feat = res34output['backbone_features'] # (2,512,8,8)
backbone_res18_feat = res18output['backbone_features'] # (2,512,8,8)
backbone_unet_feat = unet_output['backbone_features'] # (2,512,16,16)
unet_out = unet_output['out']
unet_final = unet_output['out_classifier']
print(backbone_res101_feat.size())
print(backbone_res50_feat.size())
print(backbone_res34_feat.size())
print(backbone_res18_feat.size())
print(backbone_unet_feat.size())
print(unet_out.size())
print(unet_final.size())