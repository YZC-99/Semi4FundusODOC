import numpy as np
from PIL import Image

img_path = './g0001.jpg'
mask_path = './g0001.bmp'

mask_arr = np.array(Image.open(mask_path))
img_arr = np.array(Image.open(img_path))

print('')