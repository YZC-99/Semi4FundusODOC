import os
import csv

# path = 'experiments/SEG/cropped_sup'
#
# csv_path = os.path.join(path, 'statistic.csv')
# with open(csv_path, 'w', newline='') as csvfile:
#     w = csv.writer(csvfile)
#     # 写入列头
#     w.writerow(['experiment', 'OD_dice', 'OD_mIoU', 'OC_dice', 'OC_mIoU'])
#     for root, dirs, file in os.walk(path):
#         if 'ckpt' in root:
#             experiment = root.split(path)[-1].replace('ckpt', '')
#             results_list = [i.split('-')[0].replace('val_', '').split('=') for i in file]
#             results_dict = {item[0]: round(float(item[1].replace('.ckpt','')) * 100, 2) for item in results_list}
#             w.writerow([experiment, results_dict['OD_dice'],
#                         results_dict['OD_mIoU'],
#                         results_dict['OC_dice'],
#                         results_dict['OC_mIoU']])

import os
import csv

path = 'experiments/RIM-ONE/cropped_sup256x256'
csv_path = os.path.join(path, 'statistic.csv')
with open(csv_path, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    # 写入列头
    w.writerow(['experiment','OD_dice', 'OD_mIoU', 'OC_dice', 'OC_mIoU'])
    for root, dirs, file in os.walk(path):
        if 'ckpt' in root:
            file = [ i for i in file if 'valloss' not in i]
            file = [ i for i in file if 'last' not in i]
            data = [i.replace("val_","").split('-')[1:] for i in file]
            result = {}
            for sublist in data:
                for item in sublist:
                    key, value = item.split('=')
                    key = key.strip()  # 去除键的前后空格
                    value = value.replace(".ckpt","")  # 去除文件扩展名
                    result[key] = value
            w.writerow([root.replace(path,"").replace("/lightning_logs/","").replace("/ckpt",""),
                        round(float(result['OD_dice']) * 100,2),
                        round(float(result['OD_mIoU']) * 100,2),
                        round(float(result['OC_dice']) * 100,2),
                        round(float(result['OC_mIoU']) * 100,2)
                       ]
                        )




