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

path = 'experiments/REFUGE/cropped_sup512x512'
csv_path = os.path.join(path, 'statistic.csv')
with open(csv_path, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    # 写入列头
    w.writerow(['experiment','epoch', 'OD_dice', 'OD_mIoU', 'OC_dice', 'OC_mIoU'])
    for root, dirs, file in os.walk(path):
        if 'ckpt' in root:
            experiment = file[-1].replace('.ckpt', '').replace('val_','')
            results_list = experiment.split('-')
            results_dict = {i.split('=')[0]:round(float(i.split('=')[1])*100,2) for i in results_list}
            print(results_dict)
            w.writerow([root,
                        results_dict['epoch'] / 100,
                        results_dict['OD_dice'],
                        results_dict['OD_mIoU'],
                        results_dict['OC_dice'],
                        results_dict['OC_mIoU']])




