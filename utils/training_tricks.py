import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
# TTA(测试时增强)
def TTA(model,image):

        if isinstance(model,pl.LightningModule):
            out_put = model(image)
            # 水平翻转预测，再翻转回来
            out_2 = model(torch.flip(image, [-1]))
            out_2['out'] = torch.flip(out_2['out'], [-1])
            # 垂直翻转预测，再翻转回来
            out_3 = model(torch.flip(image, [-2]))
            out_3['out'] = torch.flip(out_3['out'], [-2])
            # 水平垂直翻转预测，再翻转回来
            out_4 = model(torch.flip(image, [-1, -2]))
            out_4['out'] = torch.flip(out_4['out'], [-1, -2])
            out_put['out'] = out_2['out'] + out_3['out'] + out_4['out']
        else:
            predict_1 = model(image)

            # 水平翻转预测，再翻转回来
            predict_2 = model(torch.flip(image, [-1]))['out']
            predict_2 = torch.flip(predict_2, [-1])
            # 垂直翻转预测，再翻转回来
            predict_3 = model(torch.flip(image, [-2]))['out']
            predict_3 = torch.flip(predict_3, [-2])
            # 水平垂直翻转预测，再翻转回来
            predict_4 = model(torch.flip(image, [-1, -2]))['out']
            predict_4 = torch.flip(predict_4, [-1, -2])

            # 将上述预测结果相加
            predict_list = predict_1 + predict_2 + predict_3 + predict_4
            probs = nn.functional.softmax(predict_list, dim=1)
            threshold = 0.5
            thresholded_preds = (probs >= threshold).float()
            out_put = torch.argmax(thresholded_preds, dim=1)

        return out_put

