from PIL import Image
import numpy as np

mask_path = './image_0.png'
mask = Image.open(mask_path)
arr = np.array(mask)
print(np.unique(arr))