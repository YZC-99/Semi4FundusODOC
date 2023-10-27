import cv2
import numpy as np
import os
"""
我希望在下面的代码的基础上增加一个功能：
第一阶段检测出来的最大圆形区域集为C1，第二阶段检测出来的最大区域记为C2
现在不仅要将C1和C2的轮廓绘制到原图中并进行保存，现在还需要执行的是：
创建一个全为0的图片记为mask，该图片与img的大小一模一样，随后，原图被中C1围住的区域在mask中对应的位置将像素改为255，
原图被中C1围住的区域在mask中对应的位置将像素改为150.然后保存该图片
"""



def find_largest_circle(circles):
    max_radius = 0
    largest_circle = None
    for i in circles[0, :]:
        if i[2] > max_radius:
            max_radius = i[2]
            largest_circle = i
    return largest_circle

def find_nearest_circle_to_center(circles, center):
    min_distance = float('inf')
    nearest_circle = None
    for i in circles[0, :]:
        distance = np.sqrt((i[0] - center[0]) ** 2 + (i[1] - center[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_circle = i
    return nearest_circle


def hough_transform_mask(img_path):
    # 读取图片并转换为灰度图
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用中值模糊减少噪音并提高圆形检测的准确性
    gray = cv2.medianBlur(gray, 5)

    # 使用Hough变换检测圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=10,
                               param1=30, param2=40, minRadius=40, maxRadius=0)

    if circles is None:
        return
    # 找到最大的圆C1
    largest_circle = find_largest_circle(circles)

    # 在检测到最大的圆C1和C2后，创建一个新的全零图像
    mask = np.zeros_like(img)

    if largest_circle is not None:
        center, radius = (int(largest_circle[0]), int(largest_circle[1])), int(
            largest_circle[2])  # Ensure coordinates and radius are integers
        x, y = max(int(center[0] - radius), 0), max(int(center[1] - radius), 0)
        w, h = int(radius * 2), int(radius * 2)
        stage1_img = gray[y:y + h, x:x + w]

        # 在stage1_img中继续寻找圆
        circles_stage1 = cv2.HoughCircles(stage1_img, cv2.HOUGH_GRADIENT, 1, minDist=10,
                                          param1=30, param2=60, minRadius=0, maxRadius=0)
        if circles_stage1 is None:
            return
        # 找到在stage1_img中的最大圆C2
        largest_circle_stage1 = find_nearest_circle_to_center(circles_stage1, (x + radius, y + radius))
        # 在mask中标记由圆C1围住的区域

        cv2.circle(mask, center, radius, (255, 255, 255), -1)  # -1表示填充圆

        if largest_circle_stage1 is not None:
            center_stage1 = (
            int(largest_circle_stage1[0] + x), int(largest_circle_stage1[1] + y))  # Ensure center coordinates are integers
            radius_stage1 = int(largest_circle_stage1[2])  # Ensure radius is an integer
            cv2.circle(img, center_stage1, 1, (0, 100, 100), 3)
            cv2.circle(img, center_stage1, radius_stage1, (255, 0, 255), 2)
            # 在mask中标记由圆C2围住的区域
            cv2.circle(mask, center_stage1, radius_stage1, (150, 150, 150), -1)  # -1表示填充圆

        else:
            return
        # 将最大的圆C1绘制回原图
        center = (int(center[0]), int(center[1]))  # Ensure center coordinates are integers
        radius = int(radius)  # Ensure radius is an integer
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        cv2.circle(img, center, radius, (255, 0, 255), 2)

    else:
        return
    # 保存处理后的图片
    cv2.imwrite(img_path.replace('imgs','Houghed_imgs'), img)
    cv2.imwrite(img_path.replace('imgs','Hough_masks'), mask)

dataset_path = './Drishti-GS'
for root,dir,files in os.walk(dataset_path):
    if 'imgs' in root:
        files_path = [os.path.join(root,i) for i in files]
        for img in files_path:
            hough_transform_mask(img)





