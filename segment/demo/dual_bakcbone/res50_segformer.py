from segment.modules.semseg.segformer import SegFormer
from segment.modules.backbone.resnet import resnet18,resnet34, resnet50, resnet101
import torch
segformer_model = SegFormer(num_classes=3, phi='b2', pretrained=False)
res_model = resnet34(pretrained=False)
# print(segformer_model)
print(res_model)
img= torch.randn(2,3,256,256)
res_out = res_model.base_forward(img)
segformer_backbone_out = segformer_model.backbone.forward(img)
print("x")