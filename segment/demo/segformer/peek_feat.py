from segment.modules.semseg.segformer import SegFormer
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import normalize, resize, to_pil_image
from segment.util import count_params, meanIOU, color_map
import visdom
import torchvision
import numpy as np
from segment.demo.segformer.utils import overlay_mask
import os
# 初始化Visdom客户端
# python -m visdom.server
vis = visdom.Visdom(log_to_filename='1')

# 定义权重透明叠加函数

cmap = color_map("c")
num_classes = 3
backbone = "b2"
ckpt_path = '../../../temp/segformer_subv1/epoch=67-val_OC_dice=0.901489-val_OC_mIoU=0.943023.ckpt'
model = SegFormer(num_classes=num_classes, phi=backbone,attention='subv1')

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
model.cuda()
model.eval()

img_path = './drishtiGS_001.png'
img = Image.open(img_path)
gt_path = './drishtiGS_001_gt.png'
gt = Image.open(gt_path)
trans = T.Compose([
    T.ToTensor(),
    T.Resize([256,256]),
    T.Normalize((0.0,), (1.0,)),
])
trans_gt = T.Compose([
    T.ToTensor(),
])
gt_t = trans_gt(gt)
input_t = trans(img)
input_t = input_t.unsqueeze(0).to("cuda")
# print(input_t.shape)
out = model(input_t)
features = out['out_features']
decodehead_out = out['decodehead_out']
preds = torch.nn.functional.interpolate(out['out'],size=(512,512))
preds = torch.nn.functional.softmax(preds,dim=1).argmax(1).squeeze().cpu().float()
decodehead_out.update({"_c1-_c2":decodehead_out["_c1"]-decodehead_out["_c2"]})
decodehead_out.update({"_c3-_c4":decodehead_out["_c3"]-decodehead_out["_c4"]})
vis.image(gt_t*120,opts={'title': 'ground truth'})
vis.image(preds*120,opts={'title': 'pred'})
#在通道维度上求和后，可视化
# for key,value in decodehead_out.items():
#     c = value.squeeze()
#     c = torch.sum(c,dim=0)
#     min_value = float(torch.min(c))
#
#     # 将单通道的热力图扩展为与RGB图像相同的通道数（3通道）
#     # c_rgb = torch.stack([c, c, c], dim=0)
#
#     # 调整热力图的大小以匹配原始图片
#     c_rgb_pil = to_pil_image(c.detach().squeeze(0).cpu().numpy(), mode='F')
#
#     # c_rgb_pil = torchvision.transforms.ToPILImage()(c_rgb.cpu())
#     # img_rgb_pil = img.resize(c_rgb_pil.size)
#
#     # 使用 overlay_mask 函数叠加图片
#     overlayed_image = overlay_mask(img, c_rgb_pil, alpha=0.5)
#
#     # 将叠加后的图像转为Tensor并可视化
#     overlayed_image_tensor = torchvision.transforms.ToTensor()(overlayed_image)
#     vis.image(overlayed_image_tensor, opts={'title': 'Visualization of c{}'.format(key), 'vmin': min_value})

# 在通道维度上逐个可视化从c1-c4
for key,value in decodehead_out.items():
    c = value.squeeze()
    channels_num = 0
    for chann in range(c.shape[0]):
        min_value = float(torch.min(c[chann, ...]))
        # 将单通道的热力图扩展为与RGB图像相同的通道数（3通道）
        # c_rgb = torch.stack([c, c, c], dim=0)

        # 调整热力图的大小以匹配原始图片
        c_rgb_pil = to_pil_image(c[chann, ...].detach().squeeze(0).cpu().numpy(), mode='F')

        # c_rgb_pil = torchvision.transforms.ToPILImage()(c_rgb.cpu())
        # img_rgb_pil = img.resize(c_rgb_pil.size)

        # 使用 overlay_mask 函数叠加图片
        overlayed_image = overlay_mask(img, c_rgb_pil, alpha=0.5)

        # 将叠加后的图像转为Tensor并可视化
        overlayed_image_tensor = torchvision.transforms.ToTensor()(overlayed_image)
        folder = "D:/Dev_projects/Semi4FundusODOC/temp/segformer_features_visualization/{}".format(key)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # vis.image(overlayed_image_tensor, opts={'title': 'Visualization of c{}'.format(key), 'vmin': min_value})
        overlayed_image.save("{}/c{}-{}.jpg".format(folder,key,chann))
        # vis.heatmap(c[chann, ...], opts={'title': 'Visualization of c{}-{}'.format(key,chann), 'vmin': min_value})
        channels_num += 1
        # if channels_num > 10:
        #     break
print("done")





#
# preds = torch.argmax(out['out'], dim=1).cpu()
#
# preds_arr = preds.squeeze().detach().cpu().numpy().astype(np.uint8)
# preds_img = Image.fromarray(preds_arr,mode='P')
# preds_img.putpalette(cmap)
# preds_img.save("preds.png")
# print(preds.shape)