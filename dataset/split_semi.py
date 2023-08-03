import os

# 读取文件中的数据
with open('SEG/cropped_semi/random1/training.txt', 'r') as f:
    data = f.readlines()

# 数据总量
total_data = len(data)

# 标记数据的占比从10%递增到90%
for i in range(1, 10):
    # 创建文件夹
    path = os.path.join('SEG/cropped_semi/random1',str(i * 10))
    if not os.path.exists(str(i * 10)):
        os.makedirs(path)

    # 计算标记数据和未标记数据的数量
    labeled_data_count = int(total_data * (i * 10 / 100))
    unlabeled_data_count = total_data - labeled_data_count

    # 将标记数据写入labeled.txt
    with open(os.path.join(path, 'labeled.txt'), 'w') as f:
        for j in range(labeled_data_count):
            f.write(data[j])

    # 将未标记数据写入unlabeled.txt
    with open(os.path.join(path, 'unlabeled.txt'), 'w') as f:
        for j in range(labeled_data_count, total_data):
            f.write(data[j])