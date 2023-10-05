#下列文件中的记录都是一行一条记录，其中g,txt有40条，n.txt有360条
# 我希望你帮我按照占比指定的0.25比例完成分层抽样，形成10个文件夹，分别为sample1-sample10，文件夹下有三个txt
# 分别为training.txt，val.txt,test.txt，如果被抽样到的就复制进入val.txt和test.txt(val和test的记录一样),其余的进入training.txt
# 整个操作我不希望你修改原来的文件
import os
import random

# 分层抽样函数
def stratified_sampling(records, ratio):
    sample_size = round(len(records) * ratio)
    sampled_records = random.sample(records, sample_size)
    remaining_records = [r for r in records if r not in sampled_records]
    return sampled_records, remaining_records

# 创建样本文件夹和文件的函数
def create_sample_folders(base_path, g_records, n_records, ratio=0.25):
    for i in range(1, 11):  # 创建10个样本文件夹
        folder_path = os.path.join(base_path, f'sample{i}')
        os.makedirs(folder_path, exist_ok=True)

        # 对 g.txt 和 n.txt 的记录进行分层抽样
        sampled, remaining = stratified_sampling(g_records + n_records, ratio)

        # 创建并写入 training.txt, val.txt, 和 test.txt 文件
        with open(os.path.join(folder_path, 'training.txt'), 'w') as f:
            f.writelines(remaining)
        with open(os.path.join(folder_path, 'val.txt'), 'w') as f:
            f.writelines(sampled)
        with open(os.path.join(folder_path, 'test.txt'), 'w') as f:
            f.writelines(sampled)

def main():
    g_path = './400/g.txt'
    n_path = './400/n.txt'

    # 读取原始文件的记录
    with open(g_path, 'r') as f:
        g_records = f.readlines()
    with open(n_path, 'r') as f:
        n_records = f.readlines()

    # 创建样本文件夹和文件
    base_path = './'
    create_sample_folders(base_path, g_records, n_records)

if __name__ == '__main__':
    main()
