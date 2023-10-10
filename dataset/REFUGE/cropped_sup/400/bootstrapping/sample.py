from sklearn.utils import resample
import numpy as np
from sklearn.utils import resample

g_path = '../g.txt'
n_path = '../n.txt'
# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# 从文件中加载数据
g_data = read_file(g_path)
n_data = read_file(n_path)

# 对g_data进行bootstrap采样
g_bootstrapped = resample(g_data, replace=True, n_samples=len(g_data))

# 对n_data进行bootstrap采样
n_bootstrapped = resample(n_data, replace=True, n_samples=len(n_data))

# 如果需要，您可以将bootstrap样本保存到新的文件中
with open('bootstrapped_g.txt', 'w') as file:
    for line in g_bootstrapped:
        file.write(line + '\n')

with open('bootstrapped_n.txt', 'w') as file:
    for line in n_bootstrapped:
        file.write(line + '\n')


