import numpy as np
from PIL import Image
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

def cartesian_to_polar(img):
    # Convert PIL image to numpy array
    data = np.array(img)

    # Calculate the center of the image
    center_x = data.shape[1] / 2
    center_y = data.shape[0] / 2

    # Create an empty array for the polar image
    polar_img = np.zeros_like(data)

    # Convert each pixel to polar coordinates
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            theta = np.arctan2(y - center_y, x - center_x)
            polar_x = int(r * np.cos(theta) + center_x)
            polar_y = int(r * np.sin(theta) + center_y)
            polar_img[y, x] = data[polar_y, polar_x]

    # Convert the numpy array back to a PIL image
    return Image.fromarray(polar_img)


def write_save(img,mask,name,cf):
    mask.putpalette(cmap)
    img_path_name = cropped_img_path.replace(".png", "{}.png".format(name))
    mask_path_name = cropped_mask_path.replace(".png", "{}.png".format(name))
    img.save(img_path_name)
    mask.save(mask_path_name)
    cf.write(img_path_name + ' ' + mask_path_name + '\n')

cmap = color_map('fundus')
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
        polared_img = cartesian_to_polar(img)
        polared_mask = cartesian_to_polar(mask)
        # 原始图片
        write_save(polared_img,polared_mask,'',cf)



print('done')
