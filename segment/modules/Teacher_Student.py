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
from segment.losses.blv_loss import BlvLoss
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

class TSBase(pl.LightningModule):
    def __init__(self,
                 model:str,
                 backbone: str,
                 num_classes: int,
                 cfg,
                 loss
                 ):
        super(TSBase, self).__init__()
        # self.save_hyperparameters()
        # self.automatic_optimization = False
        self.cfg = cfg
        self.num_classes = num_classes
        if model == 'deeplabv3plus':
            self.backbone = backbone
            self.model = DeepLabV3Plus(self.backbone,self.num_classes)
            self.ema_model = DeepLabV3Plus(self.backbone,self.num_classes)
        if model == 'deeplabv2':
            self.backbone = backbone
            self.model = DeepLabV2(self.backbone,self.num_classes)
        if model == 'unet':
            self.model = UNet(in_channels=self.num_classes,num_classes=3,base_c=64,bilinear=True)

        self.loss = initialize_from_config(loss)
        self.ema_loss = initialize_from_config(loss)
        if cfg.MODEL.BlvLoss:
            self.sampler = normal.Normal(0, 4)
            cls_num_list = torch.tensor([200482,42736,18925])
            frequency_list = torch.log(cls_num_list)
            self.frequency_list = (torch.log(sum(cls_num_list)) - frequency_list)

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

        # self.test_mean_dice_score = Dice(num_classes=self.cfg.MODEL.NUM_CLASSES,average='macro',multiclass=True).to(self.device)
        # self.test_mean_jaccard = JaccardIndex(num_classes=self.cfg.MODEL.NUM_CLASSES,task='multiclass').to(self.device)
        # self.test_od_dice_score = Dice(num_classes=2,average='macro',multiclass=True).to(self.device)
        # self.test_od_jaccard = JaccardIndex(num_classes=2,task='multiclass').to(self.device)

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

    def forward(self, HQ_input,LQ_input) -> Dict[str, torch.Tensor]:
        # train student
        HQ_input = torch.cat([HQ_input, LQ_input], dim=0)
        HQ_output = self.model(HQ_input)

        LQ_output = self.ema_model(LQ_input)

        return {'HQ_output':HQ_output,
                'LQ_output':LQ_output
        }

    def update_ema_variables(self,alpha=0.99):
        alpha = min(1 - 1 / (self.trainer.global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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



    def on_train_start(self) -> None:
        if self.cfg.MODEL.uda:
            self.feat_estimator = prototype_dist_estimator(feature_num=2048, cfg=self.cfg)
            if self.cfg.SOLVER.MULTI_LEVEL:
                self.out_estimator = prototype_dist_estimator(feature_num=self.cfg.MODEL.NUM_CLASSES, cfg=self.cfg)

        self.print(len(self.trainer.train_dataloader))
        self.training_dice_score = torchmetrics.Dice(num_classes=self.cfg.MODEL.NUM_CLASSES,average='macro').to(self.device)
        self.training_jaccard = torchmetrics.JaccardIndex(num_classes=self.cfg.MODEL.NUM_CLASSES,task='binary' if self.cfg.MODEL.NUM_CLASSES ==  2 else 'multiclass').to(self.device)

    def training_step(self, batch):
        HQ, LQ = batch
        HQ_input, LQ_input,HQ_label, LQ_label = HQ['img'], LQ['img'], HQ['mask'], LQ['mask']
        # HQ_input = torch.cat([HQ_input,LQ_input],dim=0)
        HQ_label = torch.cat([HQ_label,LQ_label],dim=0)
        out = self(HQ_input,LQ_input)
        HQ_output, LQ_output = out['HQ_output'],out['LQ_output']
        HQ_logits,LQ_logits = HQ_output['out'],LQ_output['out']


        loss = self.loss(HQ_logits,HQ_label)
        # train teacher
        ema_loss = self.ema_loss(LQ_logits,LQ_label)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,rank_zero_only=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,rank_zero_only=True)
        self.log("train/ema_total_loss", ema_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,rank_zero_only=True)
        # self.update_ema_variables()
        return loss+ema_loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        out = self(x,x)
        HQ_output = out['HQ_output']
        HQ_logits = HQ_output['out']
        HQ_preds = nn.functional.softmax(HQ_logits, dim=1).argmax(1)
        # print(HQ_logits.shape)
        # print(y.shape)
        loss = self.loss(HQ_logits, torch.cat([y,y],dim=0))

        LQ_output = out['LQ_output']
        LQ_logits = LQ_output['out']
        LQ_preds = nn.functional.softmax(LQ_logits, dim=1).argmax(1)
        ema_loss = self.loss(LQ_logits,y)


        return {'val_loss':loss,
                'val_ema_loss': ema_loss,
                'y': torch.cat([y,y],dim=0),
                'preds':HQ_preds,
                'ema_preds':LQ_preds
                }

    def validation_step_end(self, outputs):
        loss,ema_loss,preds,y,ema_preds= outputs['val_loss'],outputs['val_ema_loss'],outputs['preds'],outputs['y'],outputs['ema_preds']
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


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.cfg.MODEL.lr
        total_iters = self.trainer.max_steps
        optimizers = [SGD(list(self.model.parameters()) + list(self.ema_model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)]
        lambda_lr = lambda iters: lr * (1 - iters / total_iters) ** 0.9
        scheduler = LambdaLR(optimizers[0],lr_lambda=lambda_lr)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            },
        ]
        return optimizers, schedulers

    #
    # def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
    #     log = dict()
    #
    #     if isinstance(batch,tuple):
    #         src, tgt = batch
    #         src_input, src_label, tgt_input, tgt_label = src['img'], src['mask'], tgt['img'], tgt['mask']
    #         src_out = self(src_input)['out']
    #         src_out = torch.nn.functional.softmax(src_out,dim=1)
    #         src_predict = src_out.argmax(1)
    #
    #         tgt_out = self(tgt_input)['out']
    #         tgt_out = torch.nn.functional.softmax(tgt_out,dim=1)
    #         tgt_predict = tgt_out.argmax(1)
    #
    #         src_predict_color, tgt_predict_color = self.gray2rgb(src_predict, tgt_predict)
    #         src_y_color, tgt_y_color = self.gray2rgb(src_label, tgt_label)
    #
    #         log["src_image"] = src_input
    #         log["src_label"] = src_y_color
    #         log["src_predict"] = src_predict_color
    #         log["tgt_image"] = tgt_input
    #         log["tgt_label"] = tgt_y_color
    #         log["tgt_predict"] = tgt_predict_color
    #     elif isinstance(batch,dict):
    #         x = batch['img'].to(self.device)
    #         y = batch['mask']
    #         out = self(x)['out']
    #         out = torch.nn.functional.softmax(out,dim=1)
    #         predict = out.argmax(1)
    #         y_color,predict_color = self.gray2rgb(y,predict)
    #         log["image"] = x
    #         log["label"] = y_color
    #         log["predict"] = predict_color
    #     return log
