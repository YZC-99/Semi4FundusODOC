import random

# 读取txt文件内容
with open('SEG.txt', 'r') as f:
    data = f.readlines()

# 随机打乱样本顺序
random.shuffle(data)

# 计算划分的索引位置
total_samples = len(data)
train_end = int(0.6 * total_samples)
val_end = int(0.7 * total_samples)

# 划分为训练集、验证集和测试集
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# 将划分结果保存到文件
with open('sup/random1/random_SEG.txt', 'w') as f:
    f.writelines(data)

with open('sup/random1/training.txt', 'w') as f:
    f.writelines(train_data)

with open('sup/random1/val.txt', 'w') as f:
    f.writelines(val_data)

with open('sup/random1/test.txt', 'w') as f:
    f.writelines(test_data)