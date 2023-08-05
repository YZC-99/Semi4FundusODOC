from skimage import feature
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = 'cropped_img.png'
mask_path = 'cropped_mask.png'

# 读取图像
img = cv2.imread(img_path)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask[mask == 0] = 50

# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用LBP方法
radius = 3
n_points = 8 * radius
lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")

# 将mask与img_lbp进行叠加
combined = cv2.addWeighted(lbp.astype('uint8'), 0.7, mask, 0.3, 0)

# 将图像转换为灰度图
gray = cv2.cvtColor(lbp, cv2.COLOR_BGR2GRAY)
# 使用二值化处理，将灰度图转换为二值图像
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 寻找二值图像中的轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 展示img，img_lbp,mask与img_lbp按照0.5倍叠加的图
plt.figure(figsize=(10,10))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221), plt.imshow(img_rgb), plt.title('Original Image')
plt.subplot(222), plt.imshow(lbp, 'gray'), plt.title('LBP Image')
plt.subplot(223), plt.imshow(mask, 'gray'), plt.title('Mask Image')
plt.subplot(224), plt.imshow(combined, 'gray'), plt.title('Combined Image')
plt.show()