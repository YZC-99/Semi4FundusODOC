import cv2
import numpy as np
cmap = {
    0: np.array([217, 217, 217]),
    1: np.array([248, 186, 125]),
    2: np.array([142, 226, 252])
}

# Load the image and mask
img_path = 'G-24-L.jpg'
mask_path= 'G-24-L-removeLGAM.png'
img = cv2.imread(img_path)
#将mask按照这个颜色映射来转化,即将一个rgb转为灰度图
'''
    cmap[0] = np.array([217, 217, 217])
    cmap[1] = np.array([125,186, 248])
    cmap[2] = np.array([252, 226, 142])
'''
mask = cv2.imread(mask_path)
# 创建一个空的灰度图像（与原始mask尺寸相同）
gray_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

# 遍历每个像素，将其值设置为颜色映射中的索引值
for key, value in cmap.items():
    indices = np.where(np.all(mask == value, axis=-1))
    gray_mask[indices] = key
mask = gray_mask
# 获取img的尺寸
height, width = img.shape[:2]
# 调整mask的尺寸以使其与img匹配
mask = cv2.resize(mask, (width, height))


OD_mask = np.zeros_like(mask)
OC_mask = np.zeros_like(mask)
# Create OD_mask with all non-zero values set to 1
OD_mask[mask > 0] = 1
# OC_mask[mask > 76] = 1
OC_mask[mask > 1] = 1



# Find contours in OD_mask and OC_mask
OD_contours, _ = cv2.findContours(OD_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
OC_contours, _ = cv2.findContours(OC_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(img, OD_contours, -1, (255, 0,0 ), 2)  # green color for OD contours
cv2.drawContours(img, OC_contours, -1, (0, 100, 0), 2)  # red color for OC contours

# Save the resulting image
cv2.imwrite(mask_path.replace('.','covered.'), img)