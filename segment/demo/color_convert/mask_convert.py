import numpy as np
from PIL import Image



def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([217, 217, 217])
    cmap[1] = np.array([125,186, 248])
    cmap[2] = np.array([252, 226, 142])
    cmap[3] = np.array([0, 0, 255])
    return cmap
cmap = color_map()

img_path = './drishtiGS_052.png'
img = Image.open(img_path)
img.putpalette(cmap)
img.save(img_path.replace('.bmp','.png'))
