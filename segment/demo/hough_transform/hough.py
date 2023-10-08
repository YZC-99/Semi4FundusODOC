import cv2
import numpy as np

# 读取图片并转换为灰度图
img = cv2.imread('../images/drishtiGS_002.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊减少噪音并提高圆形检测的准确性
gray = cv2.medianBlur(gray, 5)

# 使用Hough变换检测圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=60, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # 画圆心
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # 画圆
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 2)

# 保存处理后的图片
cv2.imwrite('./hough_circles.png', img)
