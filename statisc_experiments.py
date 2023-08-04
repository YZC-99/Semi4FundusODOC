import os
import csv

path = 'experiments/SEG/cropped_sup'

csv_path = os.path.join(path, 'statistic.csv')
with open(csv_path, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    # 写入列头
    w.writerow(['experiment', 'OD_dice', 'OD_mIoU', 'OC_dice', 'OC_mIoU'])
    for root, dirs, file in os.walk(path):
        if 'ckpt' in root:
            experiment = root.split(path)[-1].replace('ckpt', '')
            results_list = [i.split('-')[0].replace('val_', '').split('=') for i in file]
            results_dict = {item[0]: round(float(item[1].replace('.ckpt','')) * 100, 2) for item in results_list}
            w.writerow([experiment, results_dict['OD_dice'],
                        results_dict['OD_mIoU'],
                        results_dict['OC_dice'],
                        results_dict['OC_mIoU']])


