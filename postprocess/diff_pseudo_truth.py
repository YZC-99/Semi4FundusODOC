from PIL import Image
import os
import numpy as np
from torch.nn import CrossEntropyLoss


def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])  # 红色
    cmap[2] = np.array([0, 255, 0])  # 绿色
    cmap[-1] = np.array([0, 0, 255])  # 蓝色
    cmap[-2] = np.array([255, 0, 255])

    return cmap


cmap = color_map()

root = '/root/autodl-tmp/Semi4FundusODOC/data/fundus_datasets/od_oc/Drishti-GS'
unlabeled_path = '../dataset/Drishti-GS/cropped_sup/random1/test.txt'
pseudo_path_root = '/root/autodl-tmp/Semi4FundusODOC/experiments/preds0'

with open(unlabeled_path, 'r') as f:
    truth = f.read().splitlines()
ground_truth = [os.path.join(root, i.split(' ')[1]) for i in truth]

pseudo_masks = [os.path.join(pseudo_path_root, i) for i in os.listdir(pseudo_path_root) if
                i.endswith('.png') or i.endswith('.bmp')]
size = 512
for name in truth:
    name = name.split(' ')[1]
    gt_path = os.path.join(root, name)
    pseudo_path = os.path.join(pseudo_path_root, os.path.basename(name))

    gt = Image.open(gt_path)
    pseudo = Image.open(pseudo_path)
    gt = gt.resize((size, size), Image.NEAREST)
    pseudo = pseudo.resize((size, size), Image.NEAREST)

    gt_arr = np.array(gt)
    pseudo_arr = np.array(pseudo)

    fusion = Image.fromarray(gt_arr - pseudo_arr, mode='P')
    fusion.putpalette(cmap)

    merged_width = size * 3
    merged_height = size  # 一列有三张图片，所以高度乘以3
    merged_image = Image.new("RGB", (merged_width, merged_height))

    # 将 gt、pseudo 和 fusion 图片按顺序粘贴到新的图像上
    merged_image.paste(gt, (0, 0))
    merged_image.paste(pseudo, (size, 0))
    merged_image.paste(fusion, (size * 2, 0))

    #     fusion.save(os.path.join('./diff_pseudo_truth',os.path.basename(name)))
    '''
    需要详细说明：
    内圈如果出现红色，则该部分代表真实标签为OC，而被预测成为了OD，即OC的假阴
    内圈如果出现蓝色，则该部分代表真实标签为OD，而被预测成为了OC，即为OC的假阳

    外圈出现红色，则代表该部分真实标签为OD，而被预测成为了背景，即为OD的假阴
    外圈出现了蓝色，则代表该部分真实标签为背景，而被预测成为了OD，即为OD的假阳
    '''
    merged_image.save(os.path.join('./diff_pseudo_truth', os.path.basename(name)))
