import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import normal
from torch.optim import SGD,AdamW
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
from segment.losses.lovasz_loss import lovasz_softmax,lovasz_softmaxPlus
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
    if pl_module.cfg.MODEL.ABL_loss > 0.0:
        pl_module.ABL_loss = ABL()
    if pl_module.cfg.MODEL.DC_loss > 0.0:
        pl_module.Dice_loss = DiceLoss(n_classes=pl_module.num_classes)
    if pl_module.cfg.MODEL.BD_loss > 0.0:
        pl_module.BD_loss = SurfaceLoss(idc=[1, 2])
    if pl_module.cfg.MODEL.BD_loss_reblance_alpha > 0.0:
        pl_module.BD_loss_reblance_alpha = pl_module.cfg.MODEL.BD_loss_reblance_alpha
    if pl_module.cfg.MODEL.BD_loss_increase_alpha > 0.0:
        pl_module.BD_loss_increase_alpha = pl_module.cfg.MODEL.BD_loss_increase_alpha
    if pl_module.cfg.MODEL.FC_loss > 0.0:
        pl_module.FC_loss = FocalLoss()
    if pl_module.cfg.MODEL.BlvLoss:
        pl_module.sampler = normal.Normal(0, 4)
        cls_num_list = torch.tensor([200482, 42736, 18925])
        frequency_list = torch.log(cls_num_list)
        pl_module.frequency_list = (torch.log(sum(cls_num_list)) - frequency_list)
    if pl_module.cfg.MODEL.CBL_loss is not None:
        extractor_channel = 256
        if pl_module.cfg.MODEL.model == 'SegFormer':
            extractor_channel = 768
        # pl_module.CBL_loss = Faster_CBL(pl_module.num_classes)
        pl_module.CBL_loss = CBL(pl_module.num_classes, pl_module.cfg.MODEL.CBL_loss,extractor_channel=extractor_channel)
    if pl_module.cfg.MODEL.ContrastCenterCBL_loss is not None:
        pl_module.ContrastCenterCBL_loss = ContrastCenterCBL(pl_module.num_classes, pl_module.cfg.MODEL.ContrastCenterCBL_loss)
    if pl_module.cfg.MODEL.ContrastPixelCBL_loss is not None:
        pl_module.ContrastPixelCBL_loss = ContrastPixelCBL(pl_module.num_classes, pl_module.cfg.MODEL.ContrastPixelCBL_loss)
    if pl_module.cfg.MODEL.ContrastPixelCorrectCBL_loss is not None:
        extractor_channel = 256
        if pl_module.cfg.MODEL.model == 'SegFormer':
            extractor_channel = 768
        pl_module.ContrastPixelCorrectCBL_loss = ContrastPixelCorrectCBL(pl_module.num_classes,
                                                                    pl_module.cfg.MODEL.ContrastPixelCorrectCBL_loss,extractor_channel=extractor_channel)
    if pl_module.cfg.MODEL.ContrastCrossPixelCorrectCBL_loss is not None:
        extractor_channel = 256
        if pl_module.cfg.MODEL.model == 'SegFormer':
            extractor_channel = 768
        pl_module.ContrastCrossPixelCorrectCBL_loss = ContrastCrossPixelCorrectCBL(pl_module.num_classes,
                                                                              pl_module.cfg.MODEL.ContrastCrossPixelCorrectCBL_loss,extractor_channel=extractor_channel)

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
def compute_loss(pl_module: pl.LightningModule,output,batch):
    y = batch['mask']
    backbone_feat, logits = output['backbone_features'], output['out']
    out_soft = nn.functional.softmax(logits, dim=1)
    ce_loss = pl_module.loss(logits, y)

    _CE = ce_loss
    if pl_module.cfg.MODEL.DC_loss > 0.0:
        _DC =  pl_module.cfg.MODEL.DC_loss * pl_module.Dice_loss(out_soft, y)
    if pl_module.cfg.MODEL.BD_loss > 0.0:
        dist = batch['boundary']
        if pl_module.cfg.MODEL.BD_loss_reblance_alpha > 0.0:
            _DC = _DC * (1 - pl_module.BD_loss_reblance_alpha) + pl_module.BD_loss_reblance_alpha *  pl_module.BD_loss(out_soft, dist)
            pl_module.BD_loss_reblance_alpha = pl_module.BD_loss_reblance_alpha * pl_module.current_epoch
        elif pl_module.cfg.MODEL.BD_loss_increase_alpha > 0.0:
            _DC = _DC  + pl_module.BD_loss_increase_alpha * pl_module.BD_loss(
                out_soft, dist)
            pl_module.BD_loss_increase_alpha = pl_module.BD_loss_increase_alpha * pl_module.current_epoch
        else:
            _DC = _DC + pl_module.cfg.MODEL.BD_loss * pl_module.BD_loss(out_soft, dist)

    loss = _CE + _DC
    if pl_module.cfg.MODEL.FC_loss > 0.0:
        if pl_module.current_epoch > pl_module.cfg.MODEL.FC_stop_epoch:
            loss = loss
        else:
            _FC = pl_module.cfg.MODEL.FC_loss * pl_module.FC_loss(logits, y)
            loss = loss + _FC

    if pl_module.cfg.MODEL.LOVASZ_loss > 0.0:
        _IoU = pl_module.cfg.MODEL.LOVASZ_loss * lovasz_softmax(out_soft, y, ignore=255)
        loss = loss + _IoU

    if pl_module.cfg.MODEL.Pairwise_CBL_loss:
        loss = loss + pl_module.Pairwise_CBL_loss(output, y, pl_module.model.classifier.weight, pl_module.model.classifier.bias)
    if pl_module.cfg.MODEL.ABL_loss:
        if pl_module.ABL_loss(logits, y) is not None:
            loss = loss + pl_module.ABL_loss(logits, y)

    if pl_module.cfg.MODEL.LOVASZPlus_loss:
        loss = loss + lovasz_softmaxPlus(out_soft, y, ignore=255)
    if pl_module.current_epoch > pl_module.cfg.MODEL.CBLcontrast_start_epoch:
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

    if pl_module.cfg.MODEL.aux != 0.0:
        classification_label = batch['classification_label']
        classification_loss = pl_module.loss(output['classification_logits'], classification_label)
        loss = loss + pl_module.cfg.MODEL.aux * classification_loss
    return loss

def init_metrics(pl_module: pl.LightningModule):
    '''
    - 这里计算dice有几个注意事项，如果计算二分类的dice时，num_classes=2，则返回的是正负样本的dice均值
    如果num_classes=1，则返回的是正样本的dice，但此时需要手动调整multiclass=False

    - 计算iou也有同样的事项需要注意：如果二分类任务的时候，num_classes=2，且task='binary'，那么此时计算的是正样本的iou。
    如果num_classes=2，且task='multiclass'，则计算的是正负样本的iou的总和取均值

    配置文件中的v2是指dice开了multiclass=True
    配置文件中的v3是指dice开了multiclass=False
    '''
    pl_module.od_dice_score = Dice(num_classes=1, multiclass=False,average='samples').to(pl_module.device)
    pl_module.od_withB_dice_score = Dice(num_classes=2,average='samples').to(pl_module.device)
    pl_module.od_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass',average='micro').to(pl_module.device)
    pl_module.od_binary_jaccard = JaccardIndex(num_classes=2, task='binary',average='micro').to(pl_module.device)
    pl_module.od_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(pl_module.device)
    pl_module.od_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(pl_module.device)

    if pl_module.cfg.MODEL.NUM_CLASSES == 3:
        pl_module.oc_dice_score = Dice(num_classes=1, multiclass=False,average='samples').to(pl_module.device)
        pl_module.oc_withB_dice_score = Dice(num_classes=2,average='samples').to(pl_module.device)
        pl_module.oc_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass',average='micro').to(pl_module.device)
        pl_module.oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary',average='micro').to(pl_module.device)
        pl_module.oc_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(pl_module.device)
        pl_module.oc_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(pl_module.device)

        pl_module.od_rmOC_dice_score = Dice(num_classes=1, multiclass=False,average='samples').to(pl_module.device)
        pl_module.od_rmOC_jaccard = JaccardIndex(num_classes=2, task='multiclass',average='micro').to(pl_module.device)

def gt2boundary(gt,boundary_width=1, ignore_label=-1):  # gt NHW
    gt_ud = gt[:, boundary_width:, :] - gt[:, :-boundary_width, :]  # NHW
    gt_lr = gt[:, :, boundary_width:] - gt[:, :, :-boundary_width]
    gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, boundary_width, 0, 0], mode='constant', value=0) != 0
    gt_lr = torch.nn.functional.pad(gt_lr, [0, boundary_width, 0, 0, 0, 0], mode='constant', value=0) != 0
    gt_combine = gt_lr + gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt == ignore_label)
    return gt_combine > 0


def step_end_compute_update_metrics(pl_module: pl.LightningModule, outputs):

    preds, y = outputs['preds'], outputs['y']
    # 在这里对边缘进行裁剪
    if pl_module.cfg.MODEL.preds_postprocess > 0:
        new_od_preds = copy.deepcopy(preds)
        new_od_preds[new_od_preds > 1] = 1
        new_oc_preds = copy.deepcopy(preds)
        new_oc_preds[new_oc_preds == 1] = 0
        new_oc_preds[new_oc_preds == 2] = 1

        od_preds_boundary = gt2boundary(new_od_preds,boundary_width=pl_module.cfg.MODEL.preds_postprocess) * 1
        oc_preds_boundary = gt2boundary(new_oc_preds,boundary_width=pl_module.cfg.MODEL.preds_postprocess) * 1
        # print("preds的唯一值:{}".format(torch.unique(preds)))
        # print("preds的形状:{}".format(preds.size()))
        # print("preds_boundary的唯一值:{}".format(torch.unique(preds_boundary)))
        # print("preds_boundary的形状:{}".format(preds_boundary.size()))
        # 按道理preds应该要减去边缘,
        preds = preds - od_preds_boundary - oc_preds_boundary
        preds[preds < 0] = 0

    # preds = gt_boundary.unsqueeze(1)


    # 首先是计算各个类别的dice和iou，preds里面的值就代表了对每个像素点的预测
    # 背景的指标不必计算
    # 计算视盘的指标,因为视盘的像素标签值为1，视杯为2，因此，值为1的都是od，其他的都为0
    od_preds = copy.deepcopy(preds)
    od_y = copy.deepcopy(y)
    od_preds[od_preds != 1] = 0
    od_y[od_y != 1] = 0



    if pl_module.cfg.MODEL.NUM_CLASSES == 3:
        oc_preds = copy.deepcopy(preds)
        oc_y = copy.deepcopy(y)
        oc_preds[oc_preds != 2] = 0
        oc_preds[oc_preds != 0] = 1
        oc_y[oc_y != 2] = 0
        oc_y[oc_y != 0] = 1
        pl_module.oc_dice_score.update(oc_preds, oc_y)
        pl_module.oc_withB_dice_score.update(oc_preds, oc_y)
        pl_module.oc_multiclass_jaccard.update(oc_preds, oc_y)
        pl_module.oc_binary_jaccard.update(oc_preds, oc_y)
        pl_module.oc_binary_boundary_jaccard.update(oc_preds, oc_y)
        pl_module.oc_multiclass_boundary_jaccard.update(oc_preds, oc_y)

        # 计算 od_cover_oc
        od_cover_gt = od_y + oc_y
        od_cover_gt[od_cover_gt > 0] = 1
        od_cover_preds = od_preds + oc_preds
        od_cover_preds[od_cover_preds > 0] = 1

        pl_module.od_dice_score.update(od_cover_preds, od_cover_gt)
        pl_module.od_withB_dice_score.update(od_cover_preds, od_cover_gt)
        pl_module.od_multiclass_jaccard.update(od_cover_preds, od_cover_gt)
        pl_module.od_binary_jaccard.update(od_cover_preds, od_cover_gt)
        pl_module.od_binary_boundary_jaccard.update(od_cover_preds, od_cover_gt)
        pl_module.od_multiclass_boundary_jaccard.update(od_cover_preds, od_cover_gt)

    pl_module.od_rmOC_dice_score.update(od_preds, od_y)
    pl_module.od_rmOC_jaccard.update(od_preds, od_y)

def epoch_end_show_metrics(pl_module: pl.LightningModule,tag):
    od_miou = pl_module.od_multiclass_jaccard.compute()
    pl_module.od_multiclass_jaccard.reset()

    od_iou = pl_module.od_binary_jaccard.compute()
    pl_module.od_binary_jaccard.reset()

    od_biou = pl_module.od_binary_boundary_jaccard.compute()
    pl_module.od_binary_boundary_jaccard.reset()

    od_mbiou = pl_module.od_multiclass_boundary_jaccard.compute()
    pl_module.od_multiclass_boundary_jaccard.reset()

    od_dice = pl_module.od_dice_score.compute()
    pl_module.od_dice_score.reset()

    od_withBdice = pl_module.od_withB_dice_score.compute()
    pl_module.od_withB_dice_score.reset()





    pl_module.log("{}_OD_IoU".format(tag), od_iou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OD_BIoU".format(tag), od_biou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OD_mBIoU".format(tag), od_mbiou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OD_mIoU".format(tag), od_miou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)

    pl_module.log("{}_OD_dice".format(tag), od_dice, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OD_withBdice".format(tag), od_withBdice, prog_bar=True, logger=False, on_step=False, on_epoch=True,
             sync_dist=True, rank_zero_only=True)

    pl_module.log("{}/OD_IoU".format(tag), od_iou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OD_BIoU".format(tag), od_biou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OD_mBIoU".format(tag), od_mbiou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OD_mIoU".format(tag), od_miou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OD_dice".format(tag), od_dice, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OD_withBdice".format(tag), od_withBdice, prog_bar=False, logger=True, on_step=False, on_epoch=True,
             sync_dist=True, rank_zero_only=True)

    # 每一次validation后的值都应该是最新的，而不是一直累计之前的值，因此需要一个epoch，reset一次

    oc_miou = pl_module.oc_multiclass_jaccard.compute()
    pl_module.oc_multiclass_jaccard.reset()

    oc_iou = pl_module.oc_binary_jaccard.compute()
    pl_module.oc_binary_jaccard.reset()

    oc_biou = pl_module.oc_binary_boundary_jaccard.compute()
    pl_module.oc_binary_boundary_jaccard.reset()

    oc_mbiou = pl_module.oc_multiclass_boundary_jaccard.compute()
    pl_module.oc_multiclass_boundary_jaccard.reset()

    oc_dice = pl_module.oc_dice_score.compute()
    pl_module.oc_dice_score.reset()

    oc_withBdice = pl_module.oc_withB_dice_score.compute()
    pl_module.oc_withB_dice_score.reset()

    pl_module.log("{}_OC_IoU".format(tag), oc_iou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OC_BIoU".format(tag), oc_biou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OC_mBIoU".format(tag), oc_mbiou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OC_mIoU".format(tag), oc_miou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OC_dice".format(tag), oc_dice, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}_OC_withBdice".format(tag), oc_withBdice, prog_bar=True, logger=False, on_step=False, on_epoch=True,
             sync_dist=True, rank_zero_only=True)

    pl_module.log("{}/OC_IoU".format(tag), oc_iou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OC_BIoU".format(tag), oc_biou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OC_mBIoU".format(tag), oc_mbiou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OC_mIoU".format(tag), oc_miou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OC_dice".format(tag), oc_dice, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,
             rank_zero_only=True)
    pl_module.log("{}/OC_withBdice".format(tag), oc_withBdice, prog_bar=False, logger=True, on_step=False, on_epoch=True,
             sync_dist=True, rank_zero_only=True)

    od_rm_oc_iou = pl_module.od_rmOC_jaccard.compute()
    pl_module.od_rmOC_jaccard.reset()

    od_rm_oc_dice = pl_module.od_rmOC_dice_score.compute()
    pl_module.od_rmOC_dice_score.reset()
    pl_module.log("{}_OD_rm_OC_IoU".format(tag), od_rm_oc_iou, prog_bar=True, logger=False, on_step=False,
             on_epoch=True, sync_dist=True, rank_zero_only=True)
    pl_module.log("{}_OD_rm_OC_dice".format(tag), od_rm_oc_dice, prog_bar=True, logger=False, on_step=False,
             on_epoch=True, sync_dist=True, rank_zero_only=True)
    pl_module.log("{}/OD_rm_OC_IoU".format(tag), od_rm_oc_iou, prog_bar=False, logger=True, on_step=False,
             on_epoch=True, sync_dist=True, rank_zero_only=True)
    pl_module.log("{}/OD_rm_OC_dice".format(tag), od_rm_oc_dice, prog_bar=False, logger=True, on_step=False,
             on_epoch=True, sync_dist=True, rank_zero_only=True)

def uda_train(pl_module: pl.LightningModule, batch):
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
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out,
                                                                    pl_module.cfg.MODEL.NUM_CLASSES)
            tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht_out * Wt_out,
                                                                    pl_module.cfg.MODEL.NUM_CLASSES)

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

def optimizer_config(pl_module: pl.LightningModule):
    lr = pl_module.learning_rate
    total_iters = pl_module.train_steps
    # 获取backbone的参数
    backbone_params = set(pl_module.model.backbone.parameters())
    # 获取非backbone的参数
    non_backbone_params = [p for p in pl_module.model.parameters() if p not in backbone_params]

    if pl_module.cfg.MODEL.optimizer == 'AdamW':
        param_groups = [
            {'params': pl_module.model.backbone.parameters(), 'lr': lr},
            {'params': non_backbone_params, 'lr': lr * 10}
        ]
        optimizers = [AdamW(param_groups,weight_decay=1e-2)]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0],T_max=total_iters)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]
    elif pl_module.cfg.MODEL.optimizer == 'SGD':
        # 创建两个参数组，一个用于backbone，一个用于非backbone部分
        param_groups = [
            {'params': pl_module.model.backbone.parameters(), 'lr': lr},
            {'params': non_backbone_params, 'lr': lr * 10}
        ]

        optimizers = [SGD(param_groups, momentum=0.9, weight_decay=1e-4)]
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizers[0], total_iters=total_iters, power=0.9)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]

        print(">>>>>>>>>>>>>total iters:{}<<<<<<<<<<<<<<<<".format(total_iters))
    return optimizers, schedulers

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
