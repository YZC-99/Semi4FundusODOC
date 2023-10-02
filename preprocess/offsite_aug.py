import numpy as np
from PIL import Image
import os
from segment.dataloader.transform import crop, hflip,vflip, normalize, resize, blur, cutout,dist_transform,random_scale_and_crop
from segment.dataloader.transform import random_rotate,random_translate,add_salt_pepper_noise,random_scale,color_distortion
import cv2
from segment.util import count_params, meanIOU, color_map
cmap = color_map('fundus')
root = '/root/autodl-tmp/data/Drishti-GS'
whole_path = '/root/autodl-tmp/Semi4FundusODOC/dataset/Drishti-GS/SEG.txt'
whole_cropped_path = '/root/autodl-tmp/Semi4FundusODOC/dataset/Drishti-GS/all_cropped.txt'
with open(whole_path, 'r') as f:
    whole_ids = f.read().splitlines()
with open(whole_cropped_path,'a') as cf:
    for id in whole_ids:
        img_path = os.path.join(root, id.split(' ')[0])
        mask_path = os.path.join(root, id.split(' ')[1])
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        mask_path = mask_path.replace('my_gts','my_gts_cropped')
        if 'images' in img_path:
            img_path = img_path.replace('images','images_cropped')
        if 'Images' in img_path:
            img_path = img_path.replace('Images','images_cropped')
        if '650image' in img_path:
            img_path = img_path.replace('650image','images_cropped')
        if 'MESSIDOR' in img_path:
            img_path = img_path.replace('MESSIDOR','MESSIDOR_cropped')
        if 'Magrabia' in img_path:
            img_path = img_path.replace('Magrabia','Magrabia_cropped')
        if 'BinRushed' in img_path:
            img_path = img_path.replace('BinRushed','BinRushed_cropped')
        if 'imgs' in img_path:
            img_path = img_path.replace('imgs','images_cropped')
        image_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)
        img_root = img_path.split(image_name)[0]
        mask_root = mask_path.split(mask_name)[0]
        if not os.path.exists(img_root):
            os.makedirs(img_root)
        if not os.path.exists(mask_root):
            os.makedirs(mask_root)
        mask_name = os.path.basename(mask_path)
        cropped_img_path = os.path.join(img_root,image_name)
        cropped_mask_path = os.path.join(mask_root,mask_name)
        cf.write(cropped_img_path+' '+cropped_mask_path+'\n')
        # flip
        img,mask = hflip(img,mask,p=1.0)
        img,mask = vflip(img,mask,p=1.0)
        # 保存
        img.save(cropped_img_path)
print('done')
