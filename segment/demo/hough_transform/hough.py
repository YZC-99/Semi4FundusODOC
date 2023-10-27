import cv2
import numpy as np


"""
希望对img进行Hough变化然后获取最大的圆记为C1，再将最大的圆形区域c1裁剪处理得到stage1_img,
再在stage1_img中继续寻找圆形，然后将在stage1_img中找到的圆形轮廓c2
最后将C1，C2绘制在img中
"""


# 读取图片并转换为灰度图
img = cv2.imread('../images/g0022.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊减少噪音并提高圆形检测的准确性
gray = cv2.medianBlur(gray, 5)

# 使用Hough变换检测圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=20,
                           param1=40, param2=60, minRadius=30, maxRadius=0)

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
