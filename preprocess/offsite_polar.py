import numpy as np
from PIL import Image, ImageDraw
import os
import sys

sys.path.append('..')
from segment.dataloader.transform import crop, hflip, vflip, normalize, resize, blur, cutout, dist_transform, \
    random_scale_and_crop
from segment.dataloader.transform import random_rotate, random_translate, add_salt_pepper_noise, random_scale, \
    color_distortion
from segment.util import count_params, meanIOU, color_map
import cv2

import numpy as np
from PIL import Image


def cartesian_to_polar(img, mask):
    # 将PIL图像转换为NumPy数组
    img_np = np.array(img.convert('RGB'))
    mask_np = np.array(mask)  # 假设mask是灰度图

    # 确保图像是float类型
    img_float = img_np.astype(np.float32)
    mask_float = mask_np.astype(np.float32)
    print(np.unique(mask_float))

    # 计算用于极坐标变换的值
    value = np.sqrt(((img_float.shape[0] / 2.0) ** 2.0) + ((img_float.shape[1] / 2.0) ** 2.0))

    # 执行极坐标变换
    polar_img_cv = cv2.linearPolar(img_float, (img_float.shape[1] / 2, img_float.shape[0] / 2), value,
                                   cv2.WARP_FILL_OUTLIERS)
    polar_mask_cv = cv2.linearPolar(mask_float, (mask_float.shape[1] / 2, mask_float.shape[0] / 2), value,
                                    cv2.WARP_FILL_OUTLIERS)

    # 将极坐标图像的数据类型转换为uint8
    polar_img_cv = polar_img_cv.astype(np.uint8)
    polar_mask_cv = polar_mask_cv.astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    polar_img = Image.fromarray(polar_img_cv)

    polar_mask = Image.fromarray(polar_mask_cv, mode="P")

    return polar_img, polar_mask


def write_save(img, mask, name, cf):
    cmap = color_map('fundus')
    mask.putpalette(cmap)
    img_path_name = cropped_img_path.replace(".png", "{}.png".format(name))
    mask_path_name = cropped_mask_path.replace(".png", "{}.png".format(name))
    img.save(img_path_name)
    mask.save(mask_path_name)
    cf.write(img_path_name + ' ' + mask_path_name + '\n')


root = '/root/autodl-tmp/data/REFUGE'
whole_path = '/root/autodl-tmp/Semi4FundusODOC/dataset/REFUGE/all_cropped.txt'
whole_cropped_path = '/root/autodl-tmp/Semi4FundusODOC/dataset/REFUGE/polared.txt'
with open(whole_path, 'r') as f:
    whole_ids = f.read().splitlines()
with open(whole_cropped_path, 'a') as cf:
    for id in whole_ids:
        img_path = os.path.join(root, id.split(' ')[0])
        mask_path = os.path.join(root, id.split(' ')[1])
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        mask_path = mask_path.replace('my_gts', 'my_gts_polared')
        if 'images' in img_path:
            img_path = img_path.replace('images', 'images_polared')
        if 'Images' in img_path:
            img_path = img_path.replace('Images', 'images_auged')
        if '650image' in img_path:
            img_path = img_path.replace('650image', 'images_auged')
        if 'MESSIDOR' in img_path:
            img_path = img_path.replace('MESSIDOR', 'MESSIDOR_auged')
        if 'Magrabia' in img_path:
            img_path = img_path.replace('Magrabia', 'Magrabia_auged')
        if 'BinRushed' in img_path:
            img_path = img_path.replace('BinRushed', 'BinRushed_auged')
        if 'imgs' in img_path:
            img_path = img_path.replace('imgs', 'images_auged')
        image_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)
        img_root = img_path.split(image_name)[0]
        mask_root = mask_path.split(mask_name)[0]
        if not os.path.exists(img_root):
            os.makedirs(img_root)
        if not os.path.exists(mask_root):
            os.makedirs(mask_root)
        mask_name = os.path.basename(mask_path)
        cropped_img_path = os.path.join(img_root, image_name)
        cropped_mask_path = os.path.join(mask_root, mask_name)
        polared_img, polared_mask = cartesian_to_polar(img, mask)
        # 原始图片
        write_save(polared_img, polared_mask, '', cf)

print('done')
