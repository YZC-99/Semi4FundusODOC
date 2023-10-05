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
        # 原始图片
        write_save(img,mask,'',cf)
        # flip
        img_fliped, mask_fliped = hflip(img, mask, p=1.0)
        img_fliped, mask_fliped = vflip(img_fliped, mask_fliped, p=1.0)
        write_save(img_fliped,mask_fliped,'fliped',cf)
        base_combinations = [
         (img, mask, ""),
         (img_fliped, mask_fliped, "fliped_"),
        ]

        # rotate
        rotated_combinations = []
        for img_name, msk_name, prefix in base_combinations:
         img_rotated, mask_rotated = random_rotate(img_name, msk_name, p=1.0)
         write_save(img_rotated, mask_rotated, prefix + 'rotated', cf)
         rotated_combinations.append((img_rotated, mask_rotated, prefix + "rotated_"))

        # translate
        translated_combinations = []
        all_combinations = base_combinations + rotated_combinations
        for img_name, msk_name, prefix in all_combinations:
         img_translated, mask_translated = random_translate(img_name, msk_name, p=1.0)
         write_save(img_translated, mask_translated, prefix + 'translated', cf)
         translated_combinations.append((img_translated, mask_translated, prefix + "translated_"))

        # random_scale_and_crop
        scaled_cropped_combinations = []
        all_combinations += translated_combinations
        for img_name, msk_name, prefix in all_combinations:
         img_scaled_cropped, mask_scaled_cropped = random_scale_and_crop(img_name, msk_name, p=1.0)
         write_save(img_scaled_cropped, mask_scaled_cropped, prefix + 'scaled_cropped', cf)
         scaled_cropped_combinations.append((img_scaled_cropped, mask_scaled_cropped, prefix + "scaled_cropped_"))

        # noise
        all_combinations += scaled_cropped_combinations
        for img_name, msk_name, prefix in all_combinations:
         img_noised, mask_noised = add_salt_pepper_noise(img_name, msk_name, p=1.0)
         write_save(img_noised, mask_noised, prefix + 'noised', cf)

        # # flip
        # img_fliped, mask_fliped = hflip(img, mask, p=1.0)
        # img_fliped, mask_fliped = vflip(img_fliped, mask_fliped, p=1.0)
        # write_save(img_fliped,mask_fliped,'fliped',cf)
        #
        # # rotate
        # img_rotated, mask_rotated = random_rotate(img, mask, p=1.0)
        # write_save(img_rotated, mask_rotated, 'rotated', cf)
        # img_fliped_rotated, mask_fliped_rotated = random_rotate(img_fliped, mask_fliped, p=1.0)
        # write_save(img_fliped_rotated, mask_fliped_rotated, 'fliped_rotated', cf)
        #
        # # translate
        # for img_name, msk_name, suffix in [
        #  (img, mask, 'translated'),
        #  (img_fliped, mask_fliped, 'fliped_translated'),
        #  (img_rotated, mask_rotated, 'rotated_translated'),
        #  (img_fliped_rotated, mask_fliped_rotated, 'fliped_rotated_translated')
        # ]:
        #  img_translated, mask_translated = random_translate(img_name, msk_name, p=1.0)
        #  write_save(img_translated, mask_translated, suffix, cf)
        #
        # # random_scale_and_crop
        # for img_name, msk_name, suffix in [
        #  (img, mask, 'scaled_cropped'),
        #  (img_fliped, mask_fliped, 'fliped_scaled_cropped'),
        #  (img_rotated, mask_rotated, 'rotated_scaled_cropped'),
        #  (img_fliped_rotated, mask_fliped_rotated, 'fliped_rotated_scaled_cropped')
        # ]:
        #  img_scaled_cropped, mask_scaled_cropped = random_scale_and_crop(img_name, msk_name, p=1.0)
        #  write_save(img_scaled_cropped, mask_scaled_cropped, suffix, cf)
        #
        # # noise
        # for img_name, msk_name, suffix in [
        #  (img, mask, 'noised'),
        #  (img_fliped, mask_fliped, 'fliped_noised'),
        #  (img_rotated, mask_rotated, 'rotated_noised'),
        #  (img_fliped_rotated, mask_fliped_rotated, 'fliped_rotated_noised')
        # ]:
        #  img_noised, mask_noised = add_salt_pepper_noise(img_name, msk_name, p=1.0)  # Assuming you have an 'add_noise' function
        #  write_save(img_noised, mask_noised, suffix, cf)


print('done')
