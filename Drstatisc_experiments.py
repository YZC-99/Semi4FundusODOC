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
from utils.general import get_config_from_file

path = 'experiments/Drishti-GS/cropped_sup256x256'
csv_path = os.path.join(path, 'statistic.csv')
with open(csv_path, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    # 写入列头
    w.writerow(['experiment','bb','OD_epoch','OD_dice', 'OD_mIoU','OC_epoch', 'OC_dice', 'OC_mIoU','epochs','lr','warmup_ratio','scheduler','DC_loss','BD_loss','BD_incre','FC_loss','IoU_loss','CEpair_loss','ContrastCrossPixelCorrect_loss','seed','info'])
    for root, dirs, file in os.walk(path):
        if 'ckpt' in root:
            file = [ i for i in file if 'valloss' not in i]
            file = [ i for i in file if 'last' not in i]
            data = [i.replace("val_","").split('-')[0:] for i in file]
            result = {}
            for sublist in data:
                for item in sublist:
                    key, value = item.split('=')
                    if 'OC_dice' in sublist[1] and key == 'epoch':
                        key = 'OC_best_epoch'
                    elif 'OD_dice' in sublist[1] and key == 'epoch':
                        key = 'OD_best_epoch'
                    key = key.strip()  # 去除键的前后空格
                    value = value.replace(".ckpt","")  # 去除文件扩展名
                    result[key] = value
#             print(result)
            # 获得超参配置
            config_path = root.replace("ckpt","hparams.yaml")
            config = get_config_from_file(config_path)
            DC_loss = 0.0
            BD_loss = 0.0
            BD_loss_increase_alpha = 0.0
            FC_loss = 0.0
            IoU_loss = 0.0
            CEpair_loss = 0.0
            ContrastCrossPixelCorrect_loss = 0.0
            scheduler = 'cosine'
            setting = ''
            if hasattr(config.MODEL,'DC_loss'):
                DC_loss = config.MODEL.DC_loss
            if hasattr(config.MODEL,'BD_loss'):
                BD_loss = config.MODEL.BD_loss
            if hasattr(config.MODEL,'BD_loss_increase_alpha'):
                BD_loss_increase_alpha = config.MODEL.BD_loss_increase_alpha

            if hasattr(config.MODEL,'FC_loss'):
                FC_loss = config.MODEL.FC_loss
            if hasattr(config.MODEL,'IoU_loss'):
                IoU_loss = config.MODEL.IoU_loss
            if hasattr(config.MODEL,'CEpair_loss'):
                CEpair_loss = config.MODEL.CEpair_loss
            if hasattr(config.MODEL,'ContrastCrossPixelCorrect_loss'):
                ContrastCrossPixelCorrect_loss = config.MODEL.ContrastCrossPixelCorrect_loss
            if hasattr(config.MODEL,'scheduler'):
                scheduler = config.MODEL.scheduler
            if hasattr(config.info,'setting'):
                scheduler = config.info.setting
            w.writerow([root.replace(path,"").replace("/lightning_logs/","").replace("/ckpt",""),
                        config.MODEL.backbone,
                        result['OD_best_epoch'],
                        round(float(result['OD_dice']) * 100,2),
                        round(float(result['OD_mIoU']) * 100,2),
                        result['OC_best_epoch'],
                        round(float(result['OC_dice']) * 100,2),
                        round(float(result['OC_mIoU']) * 100,2),
                        config.MODEL.epochs,
                        config.MODEL.lr,
                        config.MODEL.lr_warmup_steps_ratio,
                        scheduler,
                        DC_loss,
                        BD_loss,
                        BD_loss_increase_alpha,
                        FC_loss,
                        IoU_loss,
                        CEpair_loss,
                        ContrastCrossPixelCorrect_loss,
                        config.info.seed,
                        config.info.setting,
                       ]
                        )




