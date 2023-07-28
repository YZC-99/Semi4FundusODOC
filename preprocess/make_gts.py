import scipy.io
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops, ImageDraw
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
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
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
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])
    else:
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([255, 0, 0])
        cmap[2] = np.array([0, 255, 0])
        cmap[3] = np.array([0, 0, 255])

    return cmap


cmap = color_map('fundus')


def Drishti_GS1():
    root = "/root/autodl-tmp/data/Drishti-GS1/Drishti-GS1_files/"

    my_gts_path = os.path.join(root, 'my_gts')
    if not os.path.exists(my_gts_path):
        os.mkdir(my_gts_path)
    training_gt_path = os.path.join(root, 'Training/GT')
    test_gt_path = os.path.join(root, 'Test/Test_GT')
    test_gt_list = [test_gt_path + '/' + i + '/SoftMap' for i in os.listdir(test_gt_path) if not i.startswith(".")]
    training_gt_list = [training_gt_path + '/' + j + '/SoftMap' for j in os.listdir(training_gt_path) if
                        not j.startswith(".")]
    training_gt_list.extend(test_gt_list)
    gt_list = training_gt_list

    gt_pair_list = []
    for item in gt_list:
        pair_path = [os.path.join(item, i) for i in os.listdir(item) if not i.startswith(".")]
        gt_pair_list.append(pair_path)

    for item in gt_pair_list:
        mask_arr_result = 0
        for mask_path in item:
            mask = Image.open(mask_path)
            mask_arr = (np.array(mask) / 255).astype(np.uint8)
            mask_arr_result += mask_arr
        mask_result = Image.fromarray(mask_arr_result, mode='P')
        mask_result.putpalette(cmap)
        # mask_result.save('test.png')
        file_name = item[0].split('/')[-1].replace('cupsegSoftmap', '').replace('ODsegSoftmap', '').replace('_.png',
                                                                                                            '.png')
        mask_result.save(os.path.join(my_gts_path, file_name))
    # 最大值是255


def RIM_ONE():
    root = "/root/autodl-tmp/data/RIM-ONE/"

    my_gts_path = os.path.join(root, 'my_gts')
    imgs_path = os.path.join(root, 'imgs')
    if not os.path.exists(my_gts_path):
        os.mkdir(my_gts_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)

    glau_img_path = os.path.join(root, 'Glaucoma and suspects/Stereo Images')
    health_img_path = os.path.join(root, 'Healthy/Stereo Images')

    glau_gt_path = os.path.join(root, 'Glaucoma and suspects/Average_masks')
    health_gt_path = os.path.join(root, 'Healthy/Average_masks')

    glau_img_list = [glau_img_path + '/' + i for i in os.listdir(glau_img_path) if
                     not i.startswith(".") and i.endswith('.jpg')]
    health_img_list = [health_img_path + '/' + i for i in os.listdir(health_img_path) if
                       not i.startswith(".") and i.endswith('.jpg')]
    health_img_list.extend(glau_img_list)
    img_list = health_img_list

    glau_gt_list = [glau_gt_path + '/' + i for i in os.listdir(glau_gt_path) if
                    not i.startswith(".") and i.endswith('.png') and 'Cup-Avg' in i]
    health_gt_list = [health_gt_path + '/' + i for i in os.listdir(health_gt_path) if
                      not i.startswith(".") and i.endswith('.png') and 'Cup-Avg' in i]
    health_gt_list.extend(glau_gt_list)
    gt_list = health_gt_list

    for img_path in img_list:
        img_arr = np.array(Image.open(img_path))
        img_arr = img_arr[:, :2144 // 2, :]
        img = Image.fromarray(img_arr)
        img_file_name = img_path.split('/')[-1]
        img.save(os.path.join(imgs_path, img_file_name))

    for mask_path in gt_list:
        mask_oc_arr = np.array(Image.open(mask_path))
        mask_od_arr = np.array(Image.open(mask_path.replace('Cup', 'Disc')))
        mask_oc_arr = (mask_oc_arr / 255).astype(np.uint8)
        mask_od_arr = (mask_od_arr / 255).astype(np.uint8)
        mask_arr_final = mask_oc_arr + mask_od_arr
        mask_arr_final = mask_arr_final[:, :2144 // 2]
        mask_final = Image.fromarray(mask_arr_final, mode='P')
        mask_final.putpalette(cmap)
        gt_file_name = mask_path.split('/')[-1].replace('-Cup-Avg.png', '.png')
        mask_final.save(os.path.join(my_gts_path, gt_file_name))


def RIGA():
    root = "/root/autodl-tmp/data/RIGA/RIGA_masks/DiscCups"

    MESSIDOR_root = root + "/MESSIDOR/hards"
    MESSIDOR_path = [os.path.join(MESSIDOR_root, i) for i in os.listdir(MESSIDOR_root) if
                     not i.startswith(".")]

    Mag_root_1 = root + "/Magrabia/MagrabiFemale/hards"
    Mag_root_2 = root + "/Magrabia/MagrabiaMale/hards"
    Mag_path = [os.path.join(Mag_root_1, i) for i in os.listdir(Mag_root_1) if
                not i.startswith(".")]
    Mag_path.extend([os.path.join(Mag_root_2, i) for i in os.listdir(Mag_root_2) if
                     not i.startswith(".")])

    Bin_root1 = root + "/BinRushed/BinRushed1-Corrected/hards"
    Bin_root2 = root + "/BinRushed/BinRushed2/hards"
    Bin_root3 = root + "/BinRushed/BinRushed3/hards"
    Bin_root4 = root + "/BinRushed/BinRushed4/hards"
    Bin_path = [os.path.join(Bin_root1, i) for i in os.listdir(Bin_root1) if
                not i.startswith(".")]
    Mag_path.extend([os.path.join(Bin_root2, i) for i in os.listdir(Bin_root2) if
                     not i.startswith(".")])
    Mag_path.extend([os.path.join(Bin_root3, i) for i in os.listdir(Bin_root3) if
                     not i.startswith(".")])
    Mag_path.extend([os.path.join(Bin_root4, i) for i in os.listdir(Bin_root4) if
                     not i.startswith(".")])
    MESSIDOR_path.extend(Mag_path)
    MESSIDOR_path.extend(Bin_path)

    path_list = MESSIDOR_path

    my_gts_path = os.path.join('/root/autodl-tmp/data/RIGA', 'my_gts')
    if not os.path.exists(my_gts_path):
        os.mkdir(my_gts_path)

    for mask_path in path_list:
        mask = Image.open(mask_path)
        mask_arr = (np.array(mask)).astype(np.uint8)
        mask_arr[mask_arr == 255] = 1
        mask_arr[mask_arr == 128] = 2

        mask_final = Image.fromarray(mask_arr, mode='P')
        mask_final.putpalette(cmap)
        file_name = mask_path.replace('/root/autodl-tmp/data/RIGA/RIGA_masks/DiscCups/', '').replace('hards/', '')
        mask_final.save(os.path.join(my_gts_path, file_name))


def ORIGA():
    import scipy.io
    path = '/root/autodl-tmp/data/ORIGA/650mask'
    mask_list = [os.path.join(path,i) for i in os.listdir(path)]
    for mask_path in mask_list:
        mask_arr = scipy.io.loadmat(mask_path)['maskFull']
        mask_arr = mask_arr.astype(np.uint8)
        mask_result = Image.fromarray(mask_arr, mode='P')
        mask_result.putpalette(cmap)

        file_name = mask_path.replace('650mask','my_gts').replace('.mat','.png')
        mask_result.save(file_name)
ORIGA()

