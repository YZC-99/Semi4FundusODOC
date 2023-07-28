import os
import csv

path = 'experiments/SEG/semi'

csv_path = os.path.join(path, 'statistic.csv')
with open(csv_path, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    # 写入列头
    w.writerow(['experiment', 'OD_dice_score', 'OD_IoU', 'OC_dice_score', 'OC_IoU', 'mDice', 'mIoU'])
    for root, dirs, file in os.walk(path):
        if 'ckpt' in root:
            experiment = root.split(path)[-1].replace('ckpt', '')
            results_list = [i.split('-')[0].replace('val_', '').split('=') for i in file]
            results_dict = {item[0]: round(float(item[1]) * 100, 2) for item in results_list}
            w.writerow([experiment, results_dict['OD_dice_score'],
                        results_dict['OD_IoU'],
                        results_dict['OC_dice_score'],
                        results_dict['OC_IoU'],
                        results_dict['mDice'],
                        results_dict['mIoU']])


