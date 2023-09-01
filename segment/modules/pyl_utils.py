import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import normal
from torch.optim import SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from segment.util import meanIOU
from segment.losses.loss import PrototypeContrastiveLoss
from segment.losses.grw_cross_entropy_loss import GRWCrossEntropyLoss,Dice_GRWCrossEntropyLoss
from segment.losses.seg.boundary_loss import SurfaceLoss
from segment.losses.seg.dice_loss import DiceLoss
from segment.losses.seg.focal_loss import FocalLoss
from segment.losses.abl import ABL
from segment.losses.cbl import CBL,ContrastCenterCBL,CEpair_CBL
from segment.losses.cbl import ContrastPixelCBL,ContrastPixelCorrectCBL,ContrastCrossPixelCorrectCBL
# from segment.losses.cbl import ContrastPixelCBLV2 as ContrastPixelCBL
from segment.losses.lovasz_loss import lovasz_softmax
from segment.modules.prototype_dist_estimator import prototype_dist_estimator
from typing import List,Tuple, Dict, Any, Optional
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex,Dice
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.deeplabv2 import DeepLabV2

from segment.modules.semseg.unet import UNet
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.general import initialize_from_config
from utils.my_torchmetrics import BoundaryIoU


def init_from_ckpt(pl_module: pl.LightningModule, path: str, ignore_keys: List[str] = list()):
    sd = torch.load(path, map_location='cpu')
    if 'state_dict' in sd:
        # If 'state_dict' exists, use it directly
        sd = sd['state_dict']
        pl_module.load_state_dict(sd, strict=False)
    else:
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        pl_module.model.load_state_dict(sd, strict=False)
    print(f"Restored from {path}")

def init_loss(pl_module: pl.LightningModule):
    if pl_module.cfg.MODEL.ABL_loss:
        pl_module.ABL_loss = ABL()
    if pl_module.cfg.MODEL.DC_loss:
        pl_module.Dice_loss = DiceLoss(n_classes=pl_module.num_classes)
    if pl_module.cfg.MODEL.BD_loss:
        pl_module.BD_loss = SurfaceLoss(idc=[1, 2])
    if pl_module.cfg.MODEL.FC_loss:
        pl_module.FC_loss = FocalLoss()
    if pl_module.cfg.MODEL.BlvLoss:
        pl_module.sampler = normal.Normal(0, 4)
        cls_num_list = torch.tensor([200482, 42736, 18925])
        frequency_list = torch.log(cls_num_list)
        pl_module.frequency_list = (torch.log(sum(cls_num_list)) - frequency_list)
    if pl_module.cfg.MODEL.CBL_loss is not None:
        # pl_module.CBL_loss = Faster_CBL(pl_module.num_classes)
        pl_module.CBL_loss = CBL(pl_module.num_classes, pl_module.cfg.MODEL.CBL_loss)
    if pl_module.cfg.MODEL.ContrastCenterCBL_loss is not None:
        pl_module.ContrastCenterCBL_loss = ContrastCenterCBL(pl_module.num_classes, pl_module.cfg.MODEL.ContrastCenterCBL_loss)
    if pl_module.cfg.MODEL.ContrastPixelCBL_loss is not None:
        pl_module.ContrastPixelCBL_loss = ContrastPixelCBL(pl_module.num_classes, pl_module.cfg.MODEL.ContrastPixelCBL_loss)
    if pl_module.cfg.MODEL.ContrastPixelCorrectCBL_loss is not None:
        pl_module.ContrastPixelCorrectCBL_loss = ContrastPixelCorrectCBL(pl_module.num_classes,
                                                                    pl_module.cfg.MODEL.ContrastPixelCorrectCBL_loss)
    if pl_module.cfg.MODEL.ContrastCrossPixelCorrectCBL_loss is not None:
        pl_module.ContrastCrossPixelCorrectCBL_loss = ContrastCrossPixelCorrectCBL(pl_module.num_classes,
                                                                              pl_module.cfg.MODEL.ContrastCrossPixelCorrectCBL_loss)

    if pl_module.cfg.MODEL.Pairwise_CBL_loss is not None:
        pl_module.Pairwise_CBL_loss = CEpair_CBL(pl_module.num_classes, pl_module.cfg.MODEL.Pairwise_CBL_loss)

    if pl_module.cfg.MODEL.logitsTransform:
        pl_module.confidence_layer = nn.Sequential(
            nn.Conv2d(pl_module.model.classifier.out_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        pl_module.logit_scale = nn.Parameter(torch.ones(1, pl_module.num_classes, 1, 1))
        pl_module.logit_bias = nn.Parameter(torch.zeros(1, pl_module.num_classes, 1, 1))

def init_metrics(pl_module: pl.LightningModule):
    '''
    - 这里计算dice有几个注意事项，如果计算二分类的dice时，num_classes=2，则返回的是正负样本的dice均值
    如果num_classes=1，则返回的是正样本的dice，但此时需要手动调整multiclass=False

    - 计算iou也有同样的事项需要注意：如果二分类任务的时候，num_classes=2，且task='binary'，那么此时计算的是正样本的iou。
    如果num_classes=2，且task='multiclass'，则计算的是正负样本的iou的总和取均值

    配置文件中的v2是指dice开了multiclass=True
    配置文件中的v3是指dice开了multiclass=False
    '''
    pl_module.val_od_dice_score = Dice(num_classes=1, multiclass=False).to(pl_module.device)
    pl_module.val_od_withB_dice_score = Dice(num_classes=2, average='macro').to(pl_module.device)
    pl_module.val_od_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(pl_module.device)
    pl_module.val_od_binary_jaccard = JaccardIndex(num_classes=2, task='binary').to(pl_module.device)
    pl_module.val_od_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(pl_module.device)
    pl_module.val_od_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(pl_module.device)

    if pl_module.cfg.MODEL.NUM_CLASSES == 3:
        pl_module.val_oc_dice_score = Dice(num_classes=1, multiclass=False).to(pl_module.device)
        pl_module.val_oc_withB_dice_score = Dice(num_classes=2, average='macro').to(pl_module.device)
        pl_module.val_oc_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(pl_module.device)
        pl_module.val_oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary').to(pl_module.device)
        pl_module.val_oc_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(pl_module.device)
        pl_module.val_oc_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(pl_module.device)

        pl_module.val_od_rmOC_dice_score = Dice(num_classes=1, multiclass=False).to(pl_module.device)
        pl_module.val_od_rmOC_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(pl_module.device)

def compute_loss(pl_module: pl.LightningModule,output,batch):
    y = batch['mask']
    backbone_feat, logits = output['backbone_features'], output['out']
    out_soft = nn.functional.softmax(logits, dim=1)
    ce_loss = pl_module.loss(logits, y)
    loss = ce_loss
    if pl_module.cfg.MODEL.DC_loss:
        loss = ce_loss + pl_module.Dice_loss(out_soft, y)
    if pl_module.cfg.MODEL.BD_loss:
        dist = batch['boundary']
        loss = 0.5 * loss + 0.5 * pl_module.BD_loss(out_soft, dist)
    if pl_module.cfg.MODEL.FC_loss:
        loss = loss + pl_module.FC_loss(logits, y)
    if pl_module.cfg.MODEL.ABL_loss:
        if pl_module.ABL_loss(logits, y) is not None:
            loss = loss + pl_module.ABL_loss(logits, y)
        if pl_module.cfg.MODEL.LOVASZ_loss:
            loss = loss + lovasz_softmax(out_soft, y, ignore=255)
    if pl_module.cfg.MODEL.CBL_loss:
        loss = loss + pl_module.CBL_loss(output,y,pl_module.model.classifier.weight,pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.ContrastCenterCBL_loss:
        loss = loss + pl_module.ContrastCenterCBL_loss(output, y, pl_module.model.classifier.weight, pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.ContrastPixelCBL_loss:
        loss = loss + pl_module.ContrastPixelCBL_loss(output, y, pl_module.model.classifier.weight,
                                                  pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.ContrastPixelCorrectCBL_loss:
        loss = loss + pl_module.ContrastPixelCorrectCBL_loss(output, y, pl_module.model.classifier.weight,
                                                  pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.ContrastCrossPixelCorrectCBL_loss:
        loss = loss + pl_module.ContrastCrossPixelCorrectCBL_loss(output, y, pl_module.model.classifier.weight,
                                                        pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.Pairwise_CBL_loss:
        loss = loss + pl_module.Pairwise_CBL_loss(output, y, pl_module.model.classifier.weight, pl_module.model.classifier.bias)
    return loss


def uda_train(pl_module: pl.LightningModule,batch):
        src, tgt = batch
        src_input, src_label, tgt_input, tgt_label = src['img'], src['mask'], tgt['img'], tgt['mask']

        pcl_criterion = PrototypeContrastiveLoss(pl_module.cfg)


        # 源域图片的大小
        src_size = src_input.shape[-2:]
        # 获取高维特征和最终logits
        src_output = pl_module(src_input)
        src_feat, src_out = src_output['backbone_features'], src_output['out']
        tgt_output = pl_module(tgt_input)
        tgt_feat, tgt_out = tgt_output['backbone_features'], tgt_output['out']

        # 监督损失
        src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)
        if pl_module.cfg.SOLVER.LAMBDA_LOV > 0:
            pred_softmax = F.softmax(src_pred, dim=1)
            loss_lov = lovasz_softmax(pred_softmax, src_label, ignore=255)
            loss_sup = pl_module.loss(src_pred, src_label) + pl_module.cfg.SOLVER.LAMBDA_LOV * loss_lov
        else:
            loss_sup = pl_module.loss(src_pred, src_label)

        # 获取源域高维特征尺寸
        B, A, Hs_feat, Ws_feat = src_feat.size()
        src_feat_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs_feat, Ws_feat), mode='nearest').squeeze(
            0).long()
        src_feat_mask = src_feat_mask.contiguous().view(B * Hs_feat * Ws_feat, )
        assert not src_feat_mask.requires_grad

        # 获取目标域的预测mask
        _, _, Ht_feat, Wt_feat = tgt_feat.size()
        tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
        for j in range(pl_module.cfg.MODEL.NUM_CLASSES):
            tgt_mask[(tgt_out_maxvalue < pl_module.cfg.SOLVER.DELTA) * (tgt_mask == j)] = 255

        # 使用真实标签作为监督
        if pl_module.cfg.MODEL.uda_tgt_label:
            tgt_mask = tgt_label

        tgt_feat_mask = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_feat, Wt_feat), mode='nearest').squeeze(
            0).long()
        tgt_feat_mask = tgt_feat_mask.contiguous().view(B * Ht_feat * Wt_feat, )
        assert not tgt_feat_mask.requires_grad

        src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs_feat * Ws_feat, A)
        tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht_feat * Wt_feat, A)
        # update feature-level statistics
        pl_module.feat_estimator.update(features=tgt_feat.detach(), labels=tgt_feat_mask)
        pl_module.feat_estimator.update(features=src_feat.detach(), labels=src_feat_mask)

        # contrastive loss on both domains
        loss_feat = pcl_criterion(Proto=pl_module.feat_estimator.Proto.detach(),
                                  feat=src_feat,
                                  labels=src_feat_mask) \
                    + pcl_criterion(Proto=pl_module.feat_estimator.Proto.detach(),
                                    feat=tgt_feat,
                                    labels=tgt_feat_mask)
        if pl_module.cfg.SOLVER.MULTI_LEVEL:
            _, _, Hs_out, Ws_out = src_out.size()
            _, _, Ht_out, Wt_out = tgt_out.size()
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, pl_module.cfg.MODEL.NUM_CLASSES)
            tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht_out * Wt_out, pl_module.cfg.MODEL.NUM_CLASSES)

            src_out_mask = src_label.unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, )
            tgt_pseudo_label = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_out, Wt_out),
                                             mode='nearest').squeeze(0).long()
            tgt_out_mask = tgt_pseudo_label.contiguous().view(B * Ht_out * Wt_out, )

            # update output-level statistics
            pl_module.out_estimator.update(features=tgt_out.detach(), labels=src_out_mask)
            pl_module.out_estimator.update(features=src_out.detach(), labels=tgt_out_mask)

            # the proposed contrastive loss on prediction map
            loss_out = pcl_criterion(Proto=pl_module.out_estimator.Proto.detach(),
                                     feat=src_out,
                                     labels=src_out_mask) \
                       + pcl_criterion(Proto=pl_module.out_estimator.Proto.detach(),
                                       feat=tgt_out,
                                       labels=tgt_out_mask)

            loss = loss_sup \
                   + pl_module.cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                   + pl_module.cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss_sup + pl_module.cfg.SOLVER.LAMBDA_FEAT * loss_feat

        return loss

def gray2rgb(pl_module: pl.LightningModule,y,predict):
    # Convert labels and predictions to color images.
    y_color = torch.zeros(y.size(0), 3, y.size(1), y.size(2), device=pl_module.device)
    predict_color = torch.zeros(predict.size(0), 3, predict.size(1), predict.size(2), device=pl_module.device)
    for label, color in pl_module.color_map.items():
        mask_y = (y == int(label))
        mask_p = (predict == int(label))
        for i in range(3):  # apply each channel individually
            y_color[mask_y, i] = color[i]
            predict_color[mask_p, i] = color[i]
    return y_color,predict_color
