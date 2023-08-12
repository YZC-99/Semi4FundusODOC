from PIL import Image
import os
import numpy as np
from torch.nn import CrossEntropyLoss
import cv2

def mask_to_boundary(mask,boundary_size = 3, dilation_ratio=0.005):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((boundary_size, boundary_size), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])  # 红色
    cmap[2] = np.array([0, 255, 0])  # 绿色
    cmap[-1] = np.array([0, 0, 255])  # 蓝色
    cmap[-2] = np.array([255, 0, 255])

    return cmap


cmap = color_map()

root = '/root/autodl-tmp/Semi4FundusODOC/data/fundus_datasets/od_oc/SEG'
unlabeled_path = '../dataset/SEG/cropped_semi/random1/90/unlabeled.txt'
pseudo_path_root = '/root/autodl-tmp/Semi4FundusODOC/experiments/SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None/pseudo_masks'

with open(unlabeled_path, 'r') as f:
    truth = f.read().splitlines()
ground_truth = [os.path.join(root, i.split(' ')[1]) for i in truth]

pseudo_masks = [os.path.join(pseudo_path_root, i) for i in os.listdir(pseudo_path_root) if
                i.endswith('.png') or i.endswith('.bmp')]
size = 512
for name in truth:
    name = name.split(' ')[1]
    pseudo_path = os.path.join(pseudo_path_root, os.path.basename(name))

    pseudo = Image.open(pseudo_path)
    pseudo = pseudo.resize((size, size), Image.NEAREST)

    pseudo_arr = np.array(pseudo)

    od_pseudo_arr = np.zeros_like(pseudo_arr)
    od_pseudo_arr[pseudo_arr > 0] = 1
    oc_pseudo_arr = np.zeros_like(pseudo_arr)
    oc_pseudo_arr[pseudo_arr == 2] = 1

    od_boundary_mask_arr = mask_to_boundary(od_pseudo_arr, boundary_size=2)
    oc_boundary_mask_arr = mask_to_boundary(oc_pseudo_arr, boundary_size=2)

    od_boundary_mask = Image.fromarray(od_boundary_mask_arr,mode='P')
    oc_boundary_mask = Image.fromarray(oc_boundary_mask_arr,mode='P')

    od_boundary_mask.putpalette(cmap)
    oc_boundary_mask.putpalette(cmap)

    od_boundary_mask.save(os.path.join('./od_boundary',os.path.basename(name)))
    oc_boundary_mask.save(os.path.join('./oc_boundary',os.path.basename(name)))


