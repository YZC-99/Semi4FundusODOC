import os
from PIL import Image
import numpy as np

def compute_class_weight():
    root = '../data/fundus_datasets/od_oc/SEG/'
    path = '../dataset/SEG/cropped_sup/random1/training.txt'

    od_ratio = 0.0
    oc_ratio = 0.0
    count = 0
    with open(path, 'r') as f:
        ids = f.read().splitlines()
        for i in ids:
            count += 1
            mask_path = i.split(' ')[-1]
            mask_arr = np.array(Image.open(os.path.join(root, mask_path)))
            shape = mask_arr.shape
            counts = np.bincount(mask_arr.flatten())
            now_od_ratio = counts[1] / sum(counts)
            now_oc_ratio = counts[2] / sum(counts)

            od_ratio += now_od_ratio
            oc_ratio += now_oc_ratio

    print('od_ratio:{}'.format(od_ratio / count))
    print('oc_ratio:{}'.format(oc_ratio / count))
    print('od_counts:{}'.format(od_ratio*shape[0]*shape[1]) / count)
    print('oc_counts:{}'.format(oc_ratio*shape[0]*shape[1]) / count)

def test_crw():
    num_classes = 3
    class_weight = '../dataset/SEG/whole_class_weight.txt'
    txt_info = open(class_weight, "r").readlines()
    data_info = dict()
    for idx in range(num_classes):
        item = txt_info[idx + 1]
        data = item.strip().split("\t")
        key = data[-1].split(",")[0]
        # assert result[0] == key, "key:{}, result key:{}".format(key,result[0])

        data_info[key] = {
            "idx": int(data[0]),
            "ratio": float(data[1]),
            "train": int(data[2]),
            "val": int(data[3]),
        }

    ratio = [item['ratio'] for item in data_info.values()]
    print(ratio)
# test_crw()
compute_class_weight()