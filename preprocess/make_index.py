# 读取路径REFUGE下的images和ground_truths
# 然后形成一个index.txt，其内容格式如下
# images/V0301.jpg ground_truths/g0001.bmp
import os

root = '../data/fundus_datasets/od_oc/WACV/REFUGE_cross_new/'


def confirm_length(image_files, ground_truth_files):
    print("length of img:{}".format(len(image_files)))
    print("length of gt:{}".format(len(ground_truth_files)))
    if len(image_files) != len(ground_truth_files):
        print("Error: Number of image files and ground truth files mismatch!")
        exit(1)


def RIM_ONE(root):
    data_path = root + 'RIM-ONE'  # 数据路径

    images_dir = os.path.join(data_path, 'imgs')  # 图像文件夹路径

    save_path = data_path + '/index.txt'
    ground_truths_dir = os.path.join(data_path, 'my_gts')  # ground_truths文件夹路径
    # 获取images文件夹下的文件列表
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.jpg')]
    # 获取ground_truths文件夹下对应的文件列表
    ground_truth_files = [os.path.join(ground_truths_dir, file) for file in os.listdir(ground_truths_dir) if
                          file.endswith('.png')]
    # 确保两个文件列表长度相同
    confirm_length(image_files, ground_truth_files)
    #     创建index.txt文件并写入内容

    with open(save_path, 'w') as f:
        for image_file in image_files:
            image_path = image_file.replace(root, '')
            ground_truth_path = image_path.replace('imgs', 'my_gts').replace('jpg', 'png')
            f.write(f"{image_path} {ground_truth_path}\n")


def Drishti_GS1(root):
    data_path = root + 'Drishti-GS1/' + 'Drishti-GS1_files/'  # 数据路径
    save_path = data_path + 'index.txt'

    training_images_dir = data_path + 'Training/Images'
    test_images_dir = data_path + 'Test/Images'
    images_files = [os.path.join(training_images_dir, i) for i in os.listdir(training_images_dir) if i.endswith('.png')]
    images_files.extend([os.path.join(test_images_dir, j) for j in os.listdir(test_images_dir) if j.endswith('.png')])
    gt_dir = data_path + 'my_gts'
    gt_files = [os.path.join(gt_dir, file) for file in os.listdir(gt_dir) if file.endswith('.png')]

    confirm_length(images_files, gt_files)
    with open(save_path, 'w') as f:
        for image_file in images_files:
            image_path = image_file.replace(root, '')
            ground_truth_path = image_path.replace('Images', 'my_gts').replace("Training/", "").replace("Test/", "")
            f.write(f"{image_path} {ground_truth_path}\n")


def RIGA(root):
    data_path = root + 'RIGA/'  # 数据路径
    save_path = data_path + 'index.txt'

    dir_list = []

    MESSIDOR_images_dir = data_path + 'MESSIDOR'

    MagrabiFemale_images_dir = data_path + 'Magrabia/MagrabiFemale'
    MagrabiaMale_images_dir = data_path + 'Magrabia/MagrabiaMale'

    BinRushed1_images_dir = data_path + 'BinRushed/BinRushed1-Corrected'
    BinRushed2_images_dir = data_path + 'BinRushed/BinRushed2'
    BinRushed3_images_dir = data_path + 'BinRushed/BinRushed3'
    BinRushed4_images_dir = data_path + 'BinRushed/BinRushed4'

    dir_list.append(MESSIDOR_images_dir)
    dir_list.append(MagrabiFemale_images_dir)
    dir_list.append(MagrabiaMale_images_dir)
    dir_list.append(BinRushed1_images_dir)
    dir_list.append(BinRushed2_images_dir)
    dir_list.append(BinRushed3_images_dir)
    dir_list.append(BinRushed4_images_dir)

    images_files = []
    for dir_item in dir_list:
        images_files.extend([os.path.join(dir_item, i) for i in os.listdir(dir_item) if
                             (i.endswith('.tif') or i.endswith('.jpg')) and 'prime' in i])

    with open(save_path, 'w') as f:
        for image_file in images_files:
            image_path = image_file.replace(root, '')
            ground_truth_path = image_path.replace('RIGA', 'RIGA/my_gts').replace('.tif', '.png').replace('.jpg',
                                                                                                          '.png')
            f.write(f"{image_path} {ground_truth_path}\n")


def ACRIMA(root):
    data_path = '/root/autodl-tmp/data/ACRIMA'  # 数据路径

    images_dir = data_path  # 图像文件夹路径

    save_path = '/root/autodl-tmp/data/ACRIMA_index.txt'

    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.jpg')]

    with open(save_path, 'w') as f:
        for image_file in image_files:
            image_path = image_file.replace(root, '')
            f.write(f"{image_path} {image_path}\n")


def G1020():
    data_path = '/root/autodl-tmp/data/G1020'  # 数据路径

    images_dir = data_path  # 图像文件夹路径

    save_path = '/root/autodl-tmp/data/G1020_index.txt'

    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.jpg')]

    with open(save_path, 'w') as f:
        for image_file in image_files:
            image_path = image_file.replace('/root/autodl-tmp/data/', '')
            f.write(f"{image_path} {image_path}\n")

def DDR_Cropped():
    root = '/root/autodl-tmp/data/'
    data_path = os.path.join(root,'DDR_Cropped')  # 数据路径

    train_dir = os.path.join(data_path,'cropped_train')  # 图像文件夹路径
    val_dir = os.path.join(data_path,'cropped_valid')  # 图像文件夹路径

    save_path = os.path.join(data_path,'all_index.txt')

    image_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.jpg')]
    image_files2 = [os.path.join(val_dir, file) for file in os.listdir(val_dir) if file.endswith('.jpg')]
    image_files.extend(image_files2)
    with open(save_path, 'w') as f:
        for image_file in image_files:
            image_path = image_file.replace(root, '')
            f.write(f"{image_path} {image_path.replace('.jpg','.png')}\n")
DDR_Cropped()