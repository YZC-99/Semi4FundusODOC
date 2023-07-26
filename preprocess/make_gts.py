import scipy.io
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter,ImageChops,ImageDraw
import numpy as np
import os

def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])
    else:
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([255,0, 0])
        cmap[2] = np.array([0, 255, 0])
        cmap[3] = np.array([0, 0, 255])

    return cmap

cmap = color_map('fundus')
def Drishti_GS1():
    root = "/root/autodl-tmp/data/Drishti-GS1/Drishti-GS1_files/"
    my_gts_path = root + '/my_gts'
    if not os.path.exists:
        os.mkdir(my_gts_path)
    training_gt_path = os.path.join(root,'Training/GT')
    test_gt_path = os.path.join(root,'Test/Test_GT')
    test_gt_list = [test_gt_path + '/' + i + '/SoftMap' for i in os.listdir(test_gt_path)]
    training_gt_list = [training_gt_path + '/' + j + '/SoftMap' for j in os.listdir(training_gt_path)]
    training_gt_list.extend(test_gt_list)
    gt_list = training_gt_list

    gt_pair_list = []
    for item in gt_list:
        pair_path = [os.path.join(item,i) for i in os.listdir(item)]
        gt_pair_list.append(pair_path)

    for item in gt_pair_list:
        mask_arr_result = 0
        for mask_path in item:
            mask = Image.open(mask_path)
            mask_arr = (np.array(mask) / 255).astype(np.uint8)
            mask_arr_result += mask_arr
        mask_result = Image.fromarray(mask_arr_result,mode='P')
        mask_result.putpalette(cmap)
        # mask_result.save('test.png')
        file_name = item[0].split('/')[-1].replace('cupsegSoftmap','gt')
        mask_result.save(os.path.join(my_gts_path + file_name))
    # 最大值是255
Drishti_GS1()

