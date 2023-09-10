import os
import numpy as np
from PIL import Image

def compute_mean_std():
    root = '../data/fundus_datasets/od_oc/Drishti-GS/'
    path = '../dataset/Drishti-GS/cropped_sup/training.txt'

    num_images = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    with open(path, 'r') as f:
        ids = f.read().splitlines()
        for i in ids:
            img_path = i.split(' ')[0]
            img = Image.open(os.path.join(root, img_path))
            img_arr = np.array(img)
            channel_sum += np.sum(img_arr, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(img_arr), axis=(0, 1))
            num_images += 1

    mean = channel_sum / (num_images * img_arr.shape[0] * img_arr.shape[1])
    std = np.sqrt((channel_sum_squared / (num_images * img_arr.shape[0] * img_arr.shape[1])) - np.square(mean))

    return mean, std

mean, std = compute_mean_std()
print("Mean:", mean)
print("Std:", std)
