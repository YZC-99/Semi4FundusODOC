from segment.modules.semseg.segformer import SegFormer
import torch
from PIL import Image
import torchvision.transforms as T
from segment.util import count_params, meanIOU, color_map
import visdom
import numpy as np
# 初始化Visdom客户端
vis = visdom.Visdom()

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
trans = T.Compose([
    T.ToTensor(),
    T.Resize([256,256]),
    T.Normalize((0.0,), (1.0,)),
])
input_t = trans(img)
input_t = input_t.unsqueeze(0).to("cuda")
# print(input_t.shape)
out = model(input_t)
features = out['out_features']
decodehead_out = out['decodehead_out']
decodehead_out.update({"_c1-_c2":decodehead_out["_c1"]-decodehead_out["_c2"]})
decodehead_out.update({"_c3-_c4":decodehead_out["_c3"]-decodehead_out["_c4"]})
# 可视化从c1-c4
for key,value in decodehead_out.items():
    c = value.squeeze()
    channels_num = 0
    for chann in range(c.shape[0]):
        min_value = float(torch.min(c[chann, ...]))
        vis.heatmap(c[chann, ...], opts={'title': 'Visualization of c{}-{}'.format(key,chann), 'vmin': min_value})
        channels_num += 1
        if channels_num > 10:
            break





#
# preds = torch.argmax(out['out'], dim=1).cpu()
#
# preds_arr = preds.squeeze().detach().cpu().numpy().astype(np.uint8)
# preds_img = Image.fromarray(preds_arr,mode='P')
# preds_img.putpalette(cmap)
# preds_img.save("preds.png")
# print(preds.shape)