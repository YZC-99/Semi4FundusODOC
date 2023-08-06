
'''
读取train.list和test.list,
他们里面的内容举例如下：
RIGA_BinRushed4_image40.h5
refuge_61_crop.h5
RIGA_MESSIDOR_image188.h5

现在需要读取他们并且单独提取出包含refuge的内容，形成一个新的refuge.list并保存
'''
# 读取tran.list和test.list文件
train_list_file = './SEG_org/train.list'
test_list_file = './SEG_org/test.list'

with open(train_list_file, 'r') as train_file:
    trian_lines = train_file.readlines()

with open(test_list_file, 'r') as test_file:
    test_lines = test_file.readlines()

# 提取包含refuge关键字的行
refuge_list = [line.strip() for line in trian_lines + test_lines if 'refuge' in line]
refuge_list.sort()
# 将结果保存到新的refuge.list文件
refuge_list_file = './SEG_org/refuge.list'
with open(refuge_list_file, 'w') as refuge_file:
    refuge_file.write('\n'.join(refuge_list))
