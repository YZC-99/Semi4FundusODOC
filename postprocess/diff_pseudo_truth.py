from PIL import Image
import os
import numpy as np

def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255,0, 0])
    cmap[2] = np.array([0, 255, 0])
    cmap[3] = np.array([0, 0, 255])
    cmap[4] = np.array([255, 0, 255])

    return cmap
cmap = color_map()

root = '..//data/fundus_datasets/od_oc/SEG'
unlabeled_path = '../dataset/SEG/cropped_semi/random1/90/unlabeled.txt'
pseudo_path_root = '/roo/autodl-tmp/Semi4FundusODOC/experiments/SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None/pseudo_masks'

with open(unlabeled_path,'r') as f:
    truth = f.read().splitlines()
ground_truth = [os.path.join(root,i.split(' ')[1]) for i in truth]

pseudo_masks = [os.path.join(pseudo_path_root,i)  for i in os.listdir(pseudo_path_root) if i.endswith('.png') or i.endswith('.bmp')]

for name in truth:
    gt_path = os.path.join(root,name.split(' ')[1])
    pseudo_path = os.path.join(pseudo_path_root,name)

    gt = np.array(Image.open(gt_path))
    pseudo = np.array(Image.open(pseudo_path))

    fusion = Image.fromarray(gt+pseudo,mode='P')
    fusion.putpalette(cmap)
    fusion.save(os.path.join('./diff_pseudo_truth',name))
