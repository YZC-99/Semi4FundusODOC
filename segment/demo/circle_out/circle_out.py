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

"circle_out"
'''
mask中只有三个类别，0，1，2
其中类别1和类别2是一个同心圆的关系，类别2在中间
现在掩盖的方式是根据mask中像素值为2的区域的边缘，然后根据像素值为2区域的边缘进行掩盖，在mask上被掩盖的部分被赋值为0，而img上被掩盖的部分是一个随机噪声
现在需要改进这个方法：
1、img上被掩盖的部分不用随机噪声填充，而是根据mask中类别为1的区域在img中对应的位置采样出像素值来填充，从而被掩盖的部分是利用img对应mask像素值为1的部分的像素值填充的
2、mask被掩盖的部分使用像素值为1来填充
方法命名为circle_out_v2
'''
def circle_out_v1(img, mask, p=1.0, value_min=0, value_max=255, pixel_level=True):
    if np.random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        # 找到mask中像素值为2的部分
        mask_2 = (mask == 2).astype(np.uint8)

        # 向内腐蚀50个像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
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


def circle_out_v2(img, mask, p=1.0):
    if np.random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        # 找到mask中像素值为2的部分
        mask_2 = (mask == 2).astype(np.uint8)

        # 向内腐蚀50个像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask_2_eroded = cv2.erode(mask_2, kernel)

        # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
        mask_edge = mask_2 - mask_2_eroded
        edge_y, edge_x = np.where(mask_edge > 0)

        # 找到mask中像素值为1的部分
        mask_1 = (mask == 1).astype(np.uint8)

        # 创建一个蒙版来保存类别1区域的像素值
        mask_1_values = img * np.expand_dims(mask_1, axis=-1)

        # 计算类别1区域的像素值的平均值
        mask_1_mean = np.sum(mask_1_values, axis=(0, 1)) / np.sum(mask_1)

        # 将边缘掩盖掉
        for y, x in zip(edge_y, edge_x):
            img[y, x] = mask_1_mean  # 使用类别1区域的像素值的平均值填充
            mask[y, x] = 1  # 使用像素值为1填充mask的被掩盖部分

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask

import cv2
import numpy as np
from PIL import Image

def circle_out_v3(img, mask, p=1.0):
    if np.random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        # 找到mask中像素值为2的部分
        mask_2 = (mask == 2).astype(np.uint8)

        # 向内腐蚀50个像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask_2_eroded = cv2.erode(mask_2, kernel)

        # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
        mask_edge = mask_2 - mask_2_eroded
        edge_y, edge_x = np.where(mask_edge > 0)

        # 找到mask中像素值为1的部分
        mask_1 = (mask == 1).astype(np.uint8)
        mask_1_y, mask_1_x = np.where(mask_1 > 0)

        # 如果mask中类别为1的区域不为空，则从中随机采样像素值
        if len(mask_1_y) > 0:
            # 将边缘掩盖掉
            for y, x in zip(edge_y, edge_x):
                # 随机选择mask中类别为1的一个像素点
                random_idx = np.random.randint(len(mask_1_y))
                random_y, random_x = mask_1_y[random_idx], mask_1_x[random_idx]
                # 使用随机选择的像素值填充img上的被掩盖部分
                img[y, x] = img[random_y, random_x]
                # 使用像素值为1填充mask的被掩盖部分
                mask[y, x] = 1

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask


import cv2
import numpy as np
from PIL import Image


def circle_out_v4(img, mask, p=1.0):
    if np.random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        # 找到mask中像素值为2的部分
        mask_2 = (mask == 2).astype(np.uint8)

        # 向内腐蚀50个像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        mask_2_eroded = cv2.erode(mask_2, kernel)

        # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
        mask_edge = mask_2 - mask_2_eroded
        edge_y, edge_x = np.where(mask_edge > 0)

        # 创建x和y的坐标网格
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # 创建一个从边缘区域指向图像中心的向量场
        center_x, center_y = w // 2, h // 2
        vectors_x = center_x - x
        vectors_y = center_y - y

        # 使用向量场创建一个坐标映射
        map_x = (x + vectors_x * mask_edge).astype(np.float32)
        map_y = (y + vectors_y * mask_edge).astype(np.float32)

        # 使用cv2.remap进行插值
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 将mask的被掩盖部分填充为1
        mask[edge_y, edge_x] = 1

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask


img = Image.open('./drishtiGS_002.png')
mask = Image.open('./drishtiGS_002_gt.png')
# out_img,out_mask = cutout(img,mask)
out_img,out_mask = circle_out_v4(img=img,mask=mask,p=1.0)
out_img.save('out_img.png')
out_mask.putpalette(cmap)
out_mask.save('out_mask.png')
