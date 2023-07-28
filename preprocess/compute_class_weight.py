import os
from PIL import Image
import numpy as np

root = '../data/fundus_datasets/od_oc/SEG/'
path = '../dataset/SEG/sup/random1/training.txt'

od_ratio = 0.0
oc_ratio = 0.0
count = 0
with open(path, 'r') as f:
    ids = f.read().splitlines()
    for i in ids:
        count += 0
        mask_path = i.split(' ')[-1]
        mask_arr = np.array(Image.open(os.path.join(root, mask_path)))

        counts = np.bincount(mask_arr.flatten())
        now_od_ratio = counts[1] / sum(counts)
        now_oc_ratio = counts[2] / sum(counts)

        od_ratio += now_od_ratio
        oc_ratio += now_oc_ratio

print('od_ratio:{}'.format(od_ratio / count))
print('oc_ratio:{}'.format(oc_ratio / count))
