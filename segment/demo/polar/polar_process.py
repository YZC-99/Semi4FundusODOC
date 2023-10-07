
import cv2

import numpy as np
from PIL import Image
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

cmap = color_map('eye')

def cartesian_to_polar(img, mask):
    # 将PIL图像转换为NumPy数组
    img_np = np.array(img.convert('RGB'))
    mask_np = np.array(mask)  # 假设mask是灰度图

    # 确保图像是float类型
    img_float = img_np.astype(np.float32)
    mask_float = mask_np.astype(np.float32)
    print(np.unique(mask_float))

    # 计算用于极坐标变换的值，使用图像的短边作为半径
    value = min(img_float.shape[0], img_float.shape[1]) / 2

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


def polar_to_cartesian(polar_img, polar_mask):
    # 将PIL图像转换为NumPy数组
    polar_img_np = np.array(polar_img.convert('RGB'))
    polar_mask_np = np.array(polar_mask)  # 假设mask是灰度图

    # 确保图像是float类型
    polar_img_float = polar_img_np.astype(np.float32)
    polar_mask_float = polar_mask_np.astype(np.float32)

    # 计算用于笛卡尔坐标变换的值，使用图像的短边作为半径
    value = min(polar_img_float.shape[0], polar_img_float.shape[1]) / 2

    # 执行笛卡尔坐标变换
    cartesian_img_cv = cv2.linearPolar(polar_img_float, (polar_img_float.shape[1] / 2, polar_img_float.shape[0] / 2),
                                       value,
                                       cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    cartesian_mask_cv = cv2.linearPolar(polar_mask_float,
                                        (polar_mask_float.shape[1] / 2, polar_mask_float.shape[0] / 2), value,
                                        cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    # 将笛卡尔坐标图像的数据类型转换为uint8
    cartesian_img_cv = cartesian_img_cv.astype(np.uint8)
    cartesian_mask_cv = cartesian_mask_cv.astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    cartesian_img = Image.fromarray(cartesian_img_cv)
    cartesian_mask = Image.fromarray(cartesian_mask_cv, mode="P")

    return cartesian_img, cartesian_mask


img_path = './cropped_img.png'
mask_path = './cropped_mask.png'
img = Image.open(img_path)
mask = Image.open(mask_path)
polared_img,polared_mask = cartesian_to_polar(img,mask)
car_from_polar_img,car_from_polar_mask = polar_to_cartesian(polared_img,polared_mask)
polared_mask.putpalette(cmap)
car_from_polar_mask.putpalette(cmap)
polared_img.save('polared_img.png')
polared_mask.save('polared_mask.png')
car_from_polar_img.save('car_from_polar_img.png')
car_from_polar_mask.save('car_from_polar_mask.png')

