import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageEnhance
import random
from segment.util import count_params, meanIOU, color_map
import cv2
from scipy.ndimage import binary_dilation
import torch
from torchvision import transforms
from segment.dataloader.boundary_utils import class2one_hot,one_hot2dist

cmap = color_map('eye')


'''
下面这个代码是随机的掩盖掉一个矩形，但我现在的需求是：
根据传入进来的mask，mask有三个类别，0，1，2，
第一：首先现在我想先计算mask中像素值为2的10个像素宽度的边缘，将边缘像素位置定义为B
第二：将B掩盖掉
请帮我完成这个代码
'''
def cutout_org(img, mask, p=1.0, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask


import cv2
import numpy as np

def cutout(img, mask, p=1.0, value_min=0, value_max=255, pixel_level=True):
    if np.random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        # 找到mask中像素值为2的部分
        mask_2 = (mask == 2).astype(np.uint8)

        # 向内腐蚀50个像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
        mask_2_eroded = cv2.erode(mask_2, kernel)

        # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
        mask_edge = mask_2 - mask_2_eroded
        edge_y, edge_x = np.where(mask_edge > 0)

        # 将边缘掩盖掉
        for y, x in zip(edge_y, edge_x):
            if pixel_level:
                value = np.random.uniform(value_min, value_max, img[y, x].shape)
            else:
                value = np.random.uniform(value_min, value_max)
            img[y, x] = value
            mask[y, x] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
img = Image.open('./drishtiGS_002.png')
mask = Image.open('./drishtiGS_002_gt.png')
# out_img,out_mask = cutout(img,mask)
out_img,out_mask = cutout(img=img,mask=mask,p=1.0)
out_img.save('out_img.png')
out_mask.putpalette(cmap)
out_mask.save('out_mask.png')
