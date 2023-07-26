import scipy.io
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter,ImageChops,ImageDraw
import numpy as np
from segment.util import color_map
import os
cmap = color_map('fundus')
def Drishti_GS1():
    root = 'F:/DL-Data/eyes/glaucoma_OD_OC/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files'
    training_gt_path = os.path.join(root,'Training/GT')
    test_gt_path = os.path.join(root,'Test/GT')
    print(os.listdir(test_gt_path))


    # 最大值是255
Drishti_GS1()

