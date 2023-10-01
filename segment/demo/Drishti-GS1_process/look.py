import scipy.io
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops, ImageDraw
import numpy as np
import os

gt_path = 'E:/Deep_Learning_DATABASE/fundus_images/Drishti-GS1_files/Drishti-GS1_files/Test/Test_GT/drishtiGS_096/SoftMap/drishtiGS_096_cupsegSoftmap.png'

gt_arr = np.array(Image.open(gt_path))
print(np.unique(gt_arr))