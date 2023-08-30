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
# from segment.losses.cbl import ContrastPixelCBL
from segment.losses.cbl import ContrastPixelCBLV2 as ContrastPixelCBL
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


def color_code_labels(labels):
    unique_labels = torch.unique(labels)
    num_labels = len(unique_labels)
    colormap = plt.cm.get_cmap('tab10')  # 使用tab10色彩映射，可根据需要选择其他映射
    colors = colormap(np.linspace(0, 1, num_labels))

    # 创建彩色编码的图像
    color_image = torch.zeros((labels.shape[0], labels.shape[1], 3), dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        color = torch.tensor(colors[i][:3], dtype=torch.float32)  # 取RGB通道的颜色值，并指定数据类型
        mask = labels == label
        color_image[mask[:, :, 0]] = color

    return color_image

class Base(pl.LightningModule):
    def __init__(self,
                 model:str,
                 backbone: str,
                 num_classes: int,
                 cfg,
                 loss
                 ):
        super(Base, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        if model == 'deeplabv3plus':
            self.backbone = backbone
            self.model = DeepLabV3Plus(self.backbone,self.num_classes)
        if model == 'deeplabv2':
            self.backbone = backbone
            self.model = DeepLabV2(self.backbone,self.num_classes)
        if model == 'unet':
            self.model = UNet(in_channels=self.num_classes,num_classes=3,base_c=64,bilinear=True)

        self.loss = initialize_from_config(loss)
        if cfg.MODEL.ABL_loss:
            self.ABL_loss = ABL()
        if cfg.MODEL.DC_loss:
            self.Dice_loss = DiceLoss(n_classes=self.num_classes)
        if cfg.MODEL.BD_loss:
            self.BD_loss = SurfaceLoss(idc=[1,2])
        if cfg.MODEL.FC_loss:
            self.FC_loss = FocalLoss()
        if cfg.MODEL.BlvLoss:
            self.sampler = normal.Normal(0, 4)
            cls_num_list = torch.tensor([200482,42736,18925])
            frequency_list = torch.log(cls_num_list)
            self.frequency_list = (torch.log(sum(cls_num_list)) - frequency_list)
        if cfg.MODEL.CBL_loss is not None:
            # self.CBL_loss = Faster_CBL(self.num_classes)
            self.CBL_loss = CBL(self.num_classes,cfg.MODEL.CBL_loss)
        if cfg.MODEL.ContrastCenterCBL_loss is not None:
            self.ContrastCenterCBL_loss = ContrastCenterCBL(self.num_classes,cfg.MODEL.ContrastCenterCBL_loss)
        if cfg.MODEL.ContrastPixelCBL_loss is not None:
            self.ContrastPixelCBL_loss = ContrastPixelCBL(self.num_classes, cfg.MODEL.ContrastPixelCBL_loss)
        if cfg.MODEL.Pairwise_CBL_loss is not None:
            self.Pairwise_CBL_loss = CEpair_CBL(self.num_classes,cfg.MODEL.Pairwise_CBL_loss)

        if cfg.MODEL.logitsTransform:
            self.confidence_layer = nn.Sequential(
                nn.Conv2d(self.model.classifier.out_channels, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU()
            )
            self.logit_scale = nn.Parameter(torch.ones(1, self.num_classes, 1, 1))
            self.logit_bias = nn.Parameter(torch.zeros(1, self.num_classes, 1, 1))
            # self.loss = GRWCrossEntropyLoss(class_weight=cfg.MODEL.class_weight,num_classes=cfg.MODEL.NUM_CLASSES,exp_scale=cfg.MODEL.align_loss)
        # else:
        #     self.loss = CrossEntropyLoss(ignore_index=255)
        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}
        '''
        - 这里计算dice有几个注意事项，如果计算二分类的dice时，num_classes=2，则返回的是正负样本的dice均值
        如果num_classes=1，则返回的是正样本的dice，但此时需要手动调整multiclass=False
        
        - 计算iou也有同样的事项需要注意：如果二分类任务的时候，num_classes=2，且task='binary'，那么此时计算的是正样本的iou。
        如果num_classes=2，且task='multiclass'，则计算的是正负样本的iou的总和取均值
        
        配置文件中的v2是指dice开了multiclass=True
        配置文件中的v3是指dice开了multiclass=False
        '''
        self.val_od_dice_score = Dice(num_classes=1,multiclass=False).to(self.device)
        self.val_od_withB_dice_score = Dice(num_classes=2,average='macro').to(self.device)
        self.val_od_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)
        self.val_od_binary_jaccard = JaccardIndex(num_classes=2, task='binary').to(self.device)
        self.val_od_binary_boundary_jaccard = BoundaryIoU(num_classes=2,task='binary').to(self.device)
        self.val_od_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2,task='multiclass').to(self.device)


        if self.cfg.MODEL.NUM_CLASSES == 3:
            self.val_oc_dice_score = Dice(num_classes=1,multiclass=False).to(self.device)
            self.val_oc_withB_dice_score = Dice(num_classes=2, average='macro').to(self.device)
            self.val_oc_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)
            self.val_oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary').to(self.device)
            self.val_oc_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(self.device)
            self.val_oc_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(self.device)

            self.val_od_rmOC_dice_score = Dice(num_classes=1,multiclass=False).to(self.device)
            self.val_od_rmOC_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)

            # self.test_od_coverOC_dice_score = Dice(num_classes=2, average='macro').to(self.device)
            # self.test_od_coverOC_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)
            # self.test_oc_dice_score = Dice(num_classes=2, average='macro').to(self.device)
            # self.test_oc_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)

        if cfg.MODEL.stage1_ckpt_path is not None and cfg.MODEL.uda_pretrained:
            self.init_from_ckpt(cfg.MODEL.stage1_ckpt_path, ignore_keys='')
        if cfg.MODEL.retraining:
            self.init_from_ckpt(cfg.MODEL.stage2_ckpt_path, ignore_keys='')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        backbone_feat, logits = out['backbone_features'], out['out']
        if self.cfg.MODEL.logitsTransform:
            confidence = self.confidence_layer(logits).sigmoid()
            scores_out_tmp = confidence * (logits * self.logit_scale + self.logit_bias)
            output_out = scores_out_tmp + (1 - confidence) * logits
            out['out'] = output_out
        if self.cfg.MODEL.BlvLoss:
            viariation = self.sampler.sample(logits.shape).clamp(-1, 1)
            viariation = viariation.to(self.device)
            self.frequency_list = self.frequency_list.to(self.device)
            logits = logits + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)
            out['out'] = logits
        return out

    def compute_loss(self,output,batch):
        y = batch['mask']
        backbone_feat, logits = output['backbone_features'], output['out']
        out_soft = nn.functional.softmax(logits, dim=1)
        ce_loss = self.loss(logits, y)
        loss = ce_loss
        if self.cfg.MODEL.DC_loss:
            loss = ce_loss + self.Dice_loss(out_soft, y)
        if self.cfg.MODEL.BD_loss:
            dist = batch['boundary']
            loss = 0.5 * loss + 0.5 * self.BD_loss(out_soft, dist)
        if self.cfg.MODEL.FC_loss:
            loss = loss + self.FC_loss(logits, y)
        if self.cfg.MODEL.ABL_loss:
            if self.ABL_loss(logits, y) is not None:
                loss = loss + self.ABL_loss(logits, y)
            if self.cfg.MODEL.LOVASZ_loss:
                loss = loss + lovasz_softmax(out_soft, y, ignore=255)
        if self.cfg.MODEL.CBL_loss:
            loss = loss + self.CBL_loss(output,y,self.model.classifier.weight,self.model.classifier.bias)
        if self.cfg.MODEL.ContrastCenterCBL_loss:
            loss = loss + self.ContrastCenterCBL_loss(output, y, self.model.classifier.weight, self.model.classifier.bias)
        if self.cfg.MODEL.ContrastPixelCBL_loss:
            loss = loss + self.ContrastPixelCBL_loss(output, y, self.model.classifier.weight,
                                                      self.model.classifier.bias)
        if self.cfg.MODEL.Pairwise_CBL_loss:
            loss = loss + self.Pairwise_CBL_loss(output, y, self.model.classifier.weight, self.model.classifier.bias)
        return loss


    def gray2rgb(self,y,predict):
        # Convert labels and predictions to color images.
        y_color = torch.zeros(y.size(0), 3, y.size(1), y.size(2), device=self.device)
        predict_color = torch.zeros(predict.size(0), 3, predict.size(1), predict.size(2), device=self.device)
        for label, color in self.color_map.items():
            mask_y = (y == int(label))
            mask_p = (predict == int(label))
            for i in range(3):  # apply each channel individually
                y_color[mask_y, i] = color[i]
                predict_color[mask_p, i] = color[i]
        return y_color,predict_color

    def init_from_ckpt(self,path: str,ignore_keys: List[str] = list()):
        sd = torch.load(path,map_location='cpu')
        if 'state_dict' in sd:
            # If 'state_dict' exists, use it directly
            sd = sd['state_dict']
            self.load_state_dict(sd, strict=False)
        else :
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]

            self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def uda_train(self,batch):
        src, tgt = batch
        src_input, src_label, tgt_input, tgt_label = src['img'], src['mask'], tgt['img'], tgt['mask']

        pcl_criterion = PrototypeContrastiveLoss(self.cfg)


        # 源域图片的大小
        src_size = src_input.shape[-2:]
        # 获取高维特征和最终logits
        src_output = self(src_input)
        src_feat, src_out = src_output['backbone_features'], src_output['out']
        tgt_output = self(tgt_input)
        tgt_feat, tgt_out = tgt_output['backbone_features'], tgt_output['out']

        # 监督损失
        src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)
        if self.cfg.SOLVER.LAMBDA_LOV > 0:
            pred_softmax = F.softmax(src_pred, dim=1)
            loss_lov = lovasz_softmax(pred_softmax, src_label, ignore=255)
            loss_sup = self.loss(src_pred, src_label) + self.cfg.SOLVER.LAMBDA_LOV * loss_lov
        else:
            loss_sup = self.loss(src_pred, src_label)

        # 获取源域高维特征尺寸
        B, A, Hs_feat, Ws_feat = src_feat.size()
        src_feat_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs_feat, Ws_feat), mode='nearest').squeeze(
            0).long()
        src_feat_mask = src_feat_mask.contiguous().view(B * Hs_feat * Ws_feat, )
        assert not src_feat_mask.requires_grad

        # 获取目标域的预测mask
        _, _, Ht_feat, Wt_feat = tgt_feat.size()
        tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
        for j in range(self.cfg.MODEL.NUM_CLASSES):
            tgt_mask[(tgt_out_maxvalue < self.cfg.SOLVER.DELTA) * (tgt_mask == j)] = 255

        # 使用真实标签作为监督
        if self.cfg.MODEL.uda_tgt_label:
            tgt_mask = tgt_label

        tgt_feat_mask = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_feat, Wt_feat), mode='nearest').squeeze(
            0).long()
        tgt_feat_mask = tgt_feat_mask.contiguous().view(B * Ht_feat * Wt_feat, )
        assert not tgt_feat_mask.requires_grad

        src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs_feat * Ws_feat, A)
        tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht_feat * Wt_feat, A)
        # update feature-level statistics
        self.feat_estimator.update(features=tgt_feat.detach(), labels=tgt_feat_mask)
        self.feat_estimator.update(features=src_feat.detach(), labels=src_feat_mask)

        # contrastive loss on both domains
        loss_feat = pcl_criterion(Proto=self.feat_estimator.Proto.detach(),
                                  feat=src_feat,
                                  labels=src_feat_mask) \
                    + pcl_criterion(Proto=self.feat_estimator.Proto.detach(),
                                    feat=tgt_feat,
                                    labels=tgt_feat_mask)
        if self.cfg.SOLVER.MULTI_LEVEL:
            _, _, Hs_out, Ws_out = src_out.size()
            _, _, Ht_out, Wt_out = tgt_out.size()
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, self.cfg.MODEL.NUM_CLASSES)
            tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht_out * Wt_out, self.cfg.MODEL.NUM_CLASSES)

            src_out_mask = src_label.unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, )
            tgt_pseudo_label = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_out, Wt_out),
                                             mode='nearest').squeeze(0).long()
            tgt_out_mask = tgt_pseudo_label.contiguous().view(B * Ht_out * Wt_out, )

            # update output-level statistics
            self.out_estimator.update(features=tgt_out.detach(), labels=src_out_mask)
            self.out_estimator.update(features=src_out.detach(), labels=tgt_out_mask)

            # the proposed contrastive loss on prediction map
            loss_out = pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                     feat=src_out,
                                     labels=src_out_mask) \
                       + pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                       feat=tgt_out,
                                       labels=tgt_out_mask)

            loss = loss_sup \
                   + self.cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                   + self.cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss_sup + self.cfg.SOLVER.LAMBDA_FEAT * loss_feat

        return loss

    def on_train_start(self) -> None:
        if self.cfg.MODEL.uda:
            self.feat_estimator = prototype_dist_estimator(feature_num=2048, cfg=self.cfg)
            if self.cfg.SOLVER.MULTI_LEVEL:
                self.out_estimator = prototype_dist_estimator(feature_num=self.cfg.MODEL.NUM_CLASSES, cfg=self.cfg)

        self.print(len(self.trainer.train_dataloader))
        self.training_dice_score = torchmetrics.Dice(num_classes=self.cfg.MODEL.NUM_CLASSES,average='macro').to(self.device)
        self.training_jaccard = torchmetrics.JaccardIndex(num_classes=self.cfg.MODEL.NUM_CLASSES,task='binary' if self.cfg.MODEL.NUM_CLASSES ==  2 else 'multiclass').to(self.device)


    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        if self.cfg.MODEL.uda:
            loss = self.uda_train(batch)
        else:
            x = batch['img']
            y = batch['mask']
            output = self(x)

            loss = self.compute_loss(output,batch)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,rank_zero_only=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,rank_zero_only=True)
        return loss


    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        output = self(x)
        backbone_feat,logits = output['backbone_features'],output['out']
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        loss = self.compute_loss(output,batch)
        return {'val_loss':loss,'preds':preds,'y':y}

    def validation_step_end(self, outputs):
        loss,preds,y = outputs['val_loss'],outputs['preds'],outputs['y']
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        # 首先是计算各个类别的dice和iou，preds里面的值就代表了对每个像素点的预测
        # 背景的指标不必计算
        # 计算视盘的指标,因为视盘的像素标签值为1，视杯为2，因此，值为1的都是od，其他的都为0
        od_preds = copy.deepcopy(preds)
        od_y = copy.deepcopy(y)
        od_preds[od_preds != 1] = 0
        od_y[od_y != 1] = 0

        if self.cfg.MODEL.NUM_CLASSES == 3:
            oc_preds = copy.deepcopy(preds)
            oc_y = copy.deepcopy(y)
            oc_preds[oc_preds != 2] = 0
            oc_preds[oc_preds != 0] = 1
            oc_y[oc_y != 2] = 0
            oc_y[oc_y != 0] = 1
            self.val_oc_dice_score.update(oc_preds, oc_y)
            self.val_oc_withB_dice_score.update(oc_preds, oc_y)
            self.val_oc_multiclass_jaccard.update(oc_preds, oc_y)
            self.val_oc_binary_jaccard.update(oc_preds, oc_y)
            self.val_oc_binary_boundary_jaccard.update(oc_preds, oc_y)
            self.val_oc_multiclass_boundary_jaccard.update(oc_preds, oc_y)

            #计算 od_cover_oc
            od_cover_gt = od_y + oc_y
            od_cover_gt[od_cover_gt > 0] = 1
            od_cover_preds = od_preds + oc_preds
            od_cover_preds[od_cover_preds > 0] = 1

            self.val_od_dice_score.update(od_cover_preds,od_cover_gt)
            self.val_od_withB_dice_score.update(od_cover_preds,od_cover_gt)
            self.val_od_multiclass_jaccard.update(od_cover_preds,od_cover_gt)
            self.val_od_binary_jaccard.update(od_cover_preds,od_cover_gt)
            self.val_od_binary_boundary_jaccard.update(od_cover_preds, od_cover_gt)
            self.val_od_multiclass_boundary_jaccard.update(od_cover_preds, od_cover_gt)


        self.val_od_rmOC_dice_score.update(od_preds, od_y)
        self.val_od_rmOC_jaccard.update(od_preds, od_y)

    def on_validation_epoch_end(self) -> None:
        od_miou = self.val_od_multiclass_jaccard.compute()
        od_iou = self.val_od_binary_jaccard.compute()
        od_biou = self.val_od_binary_boundary_jaccard.compute()
        od_mbiou = self.val_od_multiclass_boundary_jaccard.compute()

        od_dice = self.val_od_dice_score.compute()
        od_withBdice = self.val_od_withB_dice_score.compute()

        self.val_od_multiclass_jaccard.reset()
        self.val_od_binary_jaccard.reset()
        self.val_od_dice_score.reset()
        self.val_od_withB_dice_score.reset()

        self.log("val_OD_IoU", od_iou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val_OD_BIoU", od_biou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val_OD_mBIoU", od_mbiou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val_OD_mIoU", od_miou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)

        self.log("val_OD_dice",od_dice, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val_OD_withBdice",od_withBdice, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)

        self.log("val/OD_IoU", od_iou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/OD_BIoU", od_biou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/OD_mBIoU", od_mbiou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/OD_mIoU", od_miou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/OD_dice", od_dice, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/OD_withBdice", od_withBdice, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)


        # 每一次validation后的值都应该是最新的，而不是一直累计之前的值，因此需要一个epoch，reset一次

        oc_miou = self.val_oc_multiclass_jaccard.compute()
        oc_iou = self.val_oc_binary_jaccard.compute()
        oc_biou = self.val_oc_binary_boundary_jaccard.compute()
        oc_mbiou = self.val_oc_multiclass_boundary_jaccard.compute()
        oc_dice = self.val_oc_dice_score.compute()
        oc_withBdice = self.val_oc_withB_dice_score.compute()
        self.val_oc_multiclass_jaccard.reset()
        self.val_oc_binary_jaccard.reset()
        self.val_oc_dice_score.reset()
        self.val_oc_withB_dice_score.reset()

        self.log("val_OC_IoU", oc_iou, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OC_BIoU", oc_biou, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OC_mBIoU", oc_mbiou, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OC_mIoU", oc_miou, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OC_dice", oc_dice, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OC_withBdice", oc_withBdice, prog_bar=True, logger=False, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)

        self.log("val/OC_IoU", oc_iou, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OC_BIoU", oc_biou, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OC_mBIoU", oc_mbiou, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OC_mIoU", oc_miou, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OC_dice", oc_dice, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OC_withBdice", oc_withBdice, prog_bar=False, logger=True, on_step=False,on_epoch=True, sync_dist=True, rank_zero_only=True)


        od_rm_oc_iou = self.val_od_rmOC_jaccard.compute()
        od_rm_oc_dice = self.val_od_rmOC_dice_score.compute()
        self.val_od_rmOC_jaccard.reset()
        self.val_od_rmOC_dice_score.reset()
        self.log("val_OD_rm_OC_IoU", od_rm_oc_iou, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val_OD_rm_OC_dice", od_rm_oc_dice, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OD_rm_OC_IoU", od_rm_oc_iou, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/OD_rm_OC_dice", od_rm_oc_dice, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)



    def test_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        output = self(x)
        backbone_feat,logits = output['backbone_features'],output['out']
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        return {'preds':preds,'y':y}

    def test_step_end(self, outputs):
        preds,y = outputs['preds'],outputs['y']
        # 首先是计算各个类别的dice和iou，preds里面的值就代表了对每个像素点的预测
        # 背景的指标不必计算
        # 计算视盘的指标,因为视盘的像素标签值为1，视杯为2，因此，值为1的都是od，其他的都为0
        od_preds = copy.deepcopy(preds)
        od_y = copy.deepcopy(y)
        od_preds[od_preds != 1] = 0
        od_y[od_y != 1] = 0

        if self.cfg.MODEL.NUM_CLASSES == 3:
            oc_preds = copy.deepcopy(preds)
            oc_y = copy.deepcopy(y)
            oc_preds[oc_preds != 2] = 0
            oc_preds[oc_preds != 0] = 1
            oc_y[oc_y != 2] = 0
            oc_y[oc_y != 0] = 1
            self.test_oc_dice_score.update(oc_preds, oc_y)
            self.test_oc_jaccard.update(oc_preds, oc_y)

            #计算 od_cover_oc
            od_cover_gt = od_y + oc_y
            od_cover_gt[od_cover_gt > 0] = 1

            od_cover_preds = od_preds + oc_preds
            od_cover_preds[od_cover_preds > 0] = 1
            self.test_od_coverOC_dice_score.update(od_cover_preds,od_cover_gt)
            self.test_oc_jaccard.update(od_cover_preds,od_cover_gt)


        self.test_od_dice_score.update(od_preds, od_y)
        self.test_od_jaccard.update(od_preds, od_y)

        self.test_mean_dice_score.update(preds, y)
        self.test_mean_jaccard.update(preds, y)

    def on_test_epoch_end(self) -> None:
        od_iou = self.test_od_jaccard.compute()
        od_dice = self.test_od_dice_score.compute()
        self.log("test_OD_IoU", od_iou, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test_OD_dice_score",od_dice, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test/OD_IoU", od_iou, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test/OD_dice_score", od_dice, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)

        self.log("test_Mean_bg_IoU", self.test_mean_jaccard.compute(), prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test_Mean_bg_dice_score", self.test_mean_dice_score.compute(), prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test/Mean_bg_IoU", self.test_mean_jaccard.compute(), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("test/Mean_bg_dice_score", self.test_mean_dice_score.compute(), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True,rank_zero_only=True)

        m_iou = od_iou
        m_dice = od_dice
        # 每一次testidation后的值都应该是最新的，而不是一直累计之前的值，因此需要一个epoch，reset一次
        self.test_mean_dice_score.reset()
        self.test_mean_jaccard.reset()
        self.test_od_dice_score.reset()
        self.test_od_jaccard.reset()
        if self.cfg.MODEL.NUM_CLASSES == 3:
            oc_iou = self.test_oc_jaccard.compute()
            oc_dice = self.test_oc_dice_score.compute()
            self.log("test_OC_IoU", oc_iou, prog_bar=True, logger=False, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test_OC_dice_score", oc_dice, prog_bar=True, logger=False, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test/OC_IoU", oc_iou, prog_bar=False, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test/OC_dice_score", oc_dice, prog_bar=False, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.test_oc_dice_score.reset()
            self.test_oc_jaccard.reset()
            m_iou = (od_iou + oc_iou)/2
            m_dice = (od_dice + oc_dice)/2

            od_cover_oc_iou = self.test_od_coverOC_jaccard.compute()
            od_cover_oc_dice = self.test_od_coverOC_dice_score.compute()
            self.log("test_OD_cover_OC_IoU", od_cover_oc_iou, prog_bar=True, logger=False, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test_OD_cover_OC_dice_score", od_cover_oc_dice, prog_bar=True, logger=False, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test/OD_cover_OC_IoU", od_cover_oc_iou, prog_bar=False, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log("test/OD_cover_OC_dice_score", od_cover_oc_dice, prog_bar=False, logger=True, on_step=False,
                     on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.test_od_coverOC_dice_score.reset()
            self.test_od_coverOC_jaccard.reset()

        self.log("test_mIoU", m_iou, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("test/mIoU", m_iou, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("test_mDice", m_dice, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("test/mDice", m_dice, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True, sync_dist=True, rank_zero_only=True)

    def setup(self, stage: str):
        if stage == 'fit':
            limit_batches = self.trainer.limit_train_batches
            batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
            batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
            num_devices = max(1, self.trainer.num_devices)
            effective_accum = self.trainer.accumulate_grad_batches * num_devices
            self.train_steps = (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Tuple[List, List]:

        lr = self.learning_rate
        # total_iters = self.trainer.max_steps
        total_iters = self.train_steps
        optimizers = [SGD(self.model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-4)]
        # lambda_lr = lambda iters: lr * (1 - iters / total_iters) ** 0.9
        # scheduler = LambdaLR(optimizers[0],lr_lambda=lambda_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizers[0], total_iters=total_iters,power=0.9)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]

        print(">>>>>>>>>>>>>total iters:{}<<<<<<<<<<<<<<<<".format(total_iters))
        return optimizers, schedulers
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()

        if isinstance(batch,tuple):
            src, tgt = batch
            src_input, src_label, tgt_input, tgt_label = src['img'], src['mask'], tgt['img'], tgt['mask']
            src_out = self(src_input)['out']
            src_out = torch.nn.functional.softmax(src_out,dim=1)
            src_predict = src_out.argmax(1)

            tgt_out = self(tgt_input)['out']
            tgt_out = torch.nn.functional.softmax(tgt_out,dim=1)
            tgt_predict = tgt_out.argmax(1)

            src_predict_color, tgt_predict_color = self.gray2rgb(src_predict, tgt_predict)
            src_y_color, tgt_y_color = self.gray2rgb(src_label, tgt_label)

            log["src_image"] = src_input
            log["src_label"] = src_y_color
            log["src_predict"] = src_predict_color
            log["tgt_image"] = tgt_input
            log["tgt_label"] = tgt_y_color
            log["tgt_predict"] = tgt_predict_color
        elif isinstance(batch,dict):
            x = batch['img'].to(self.device)
            y = batch['mask']
            out = self(x)['out']
            out = torch.nn.functional.softmax(out,dim=1)
            predict = out.argmax(1)
            y_color,predict_color = self.gray2rgb(y,predict)
            log["image"] = x
            log["label"] = y_color
            log["predict"] = predict_color
        return log
