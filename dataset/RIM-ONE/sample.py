#下列文件中的记录都是一行一条记录，其中共有159条
# 我希望你帮我抽取60条记录出来，形成10个文件夹，分别为sample1-sample10，文件夹下有三个txt
# 分别为training.txt，val.txt,test.txt，如果被抽样到的就复制进入val.txt和test.txt(val和test的记录一样),其余的进入training.txt
# 整个操作我不希望你修改原来的文件
all_path = './all_cropped.txt'

import os
import random

# 读取原始文件的所有记录
with open(all_path, 'r') as file:
    all_records = file.readlines()


# 确保文件夹存在的函数
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 抽取和分配记录的函数
def extract_and_save_samples(index):
    # 随机抽取60条记录
    selected_records = random.sample(all_records, 60)

    # 计算未被选中的记录
    not_selected = [record for record in all_records if record not in selected_records]

    # 创建对应的文件夹
    sample_dir = f'sample{index}'
    ensure_dir(sample_dir)

    # 将记录保存到对应的txt文件
    with open(os.path.join(sample_dir, 'val.txt'), 'w') as file:
        file.writelines(selected_records)

    with open(os.path.join(sample_dir, 'test.txt'), 'w') as file:
        file.writelines(selected_records)

    with open(os.path.join(sample_dir, 'training.txt'), 'w') as file:
        file.writelines(not_selected)


# 执行10次抽取和保存操作
for i in range(1, 11):
    extract_and_save_samples(i)
