import os
from PIL import Image
import numpy as np

path = 'dataset/SEG/sup/random1/training.txt'


od_ratio = 0.0
oc_ratio = 0.0
count = 0
with open(path,'r') as f:
    ids = f.read().splitlines()
    for i in ids:
        count += 0
        mask_path = i.split(' ')[-1]
        mask_arr = np.array(Image.open(mask_path))
        w,h = mask_arr.shape()
        od_count = mask_arr.count(1)
        oc_count = mask_arr.count(2)
        od_ratio += od_count/(w*h)
        oc_ratio += oc_count/(w*h)

print('od_ratio:{}'.format(od_ratio))
print('oc_ratio:{}'.format(oc_ratio))
