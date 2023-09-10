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
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus,My_DeepLabV3PlusPlus
from segment.modules.semseg.deeplabv2 import DeepLabV2

# from segment.modules.semseg.unet import UNet,ResUNet
from segment.modules.semseg.resnet_Unet import Unet
from segment.modules.semseg.segformer import SegFormer
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.general import initialize_from_config
from utils.my_torchmetrics import BoundaryIoU

from segment.modules.pyl_utils import *


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
        if model == 'mydeeplabv3plusplus':
            self.backbone = backbone
            self.model = My_DeepLabV3PlusPlus(self.backbone, self.num_classes,
                                       inplace_seven=cfg.MODEL.backbone_inplace_seven,
                                       bb_pretrained=cfg.MODEL.backbone_pretrained, attention=cfg.MODEL.Attention)
        if model == 'deeplabv3plus':
            self.backbone = backbone
            self.model = DeepLabV3Plus(self.backbone,self.num_classes,Isdysample=cfg.MODEL.Isdysample,inplace_seven=cfg.MODEL.backbone_inplace_seven,bb_pretrained=cfg.MODEL.backbone_pretrained,attention=cfg.MODEL.Attention,seghead_last=cfg.MODEL.seghead_last)
        if model == 'deeplabv2':
            self.backbone = backbone
            self.model = DeepLabV2(self.backbone,self.num_classes)
        if model == 'unet':
            if backbone == 'resnet50':
                self.model = Unet(num_classes = self.num_classes,pretrained=cfg.MODEL.backbone_pretrained,backbone = backbone)
            else:
                self.model = Unet(self.num_classes)
        if model == 'SegFormer':
            self.model = SegFormer(num_classes=self.num_classes, phi=backbone, pretrained=cfg.MODEL.backbone_pretrained,seghead_last=cfg.MODEL.seghead_last,attention=cfg.MODEL.Attention)
        self.init_from_ckpt = init_from_ckpt

        if cfg.MODEL.weightCE_loss is not None:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(cfg.MODEL.weightCE_loss))
        else:
            self.loss = initialize_from_config(loss)
        
        init_loss(self)
        init_metrics(self)

        self.compute_loss = compute_loss
        self.uda_train = uda_train
        self.gray2rgb = gray2rgb

        self.step_end_compute_update_metrics = step_end_compute_update_metrics
        self.epoch_end_show_metrics = epoch_end_show_metrics
        if cfg.MODEL.aux != 0.0:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            self.binary_classifier = nn.Linear(320, 2)



        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}

        if cfg.MODEL.stage1_ckpt_path is not None and cfg.MODEL.uda_pretrained:
            self.init_from_ckpt(self,cfg.MODEL.stage1_ckpt_path, ignore_keys='')
        if cfg.MODEL.retraining or cfg.MODEL.resume_whole_path is not None:
            self.init_from_ckpt(self,cfg.MODEL.resume_whole_path, ignore_keys='linear_pred')

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
        if self.cfg.MODEL.aux != 0.0:
            c3_pooled = self.global_avg_pool(out['c3'])
            c3_flattened = c3_pooled.view(c3_pooled.size(0), -1)
            out['classification_logits'] = self.binary_classifier(c3_flattened)
        return out


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
            loss = self.uda_train(self,batch)
        else:
            x = batch['img']
            y = batch['mask']
            output = self(x)
            loss = self.compute_loss(self,output,batch)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,)
        return loss


    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        output = self(x)
        backbone_feat,logits = output['backbone_features'],output['out']
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        loss = self.compute_loss(self,output,batch)
        return {'val_loss':loss,
                'preds':preds,
                'y':y,
                "classification_logits":output['classification_logits'],
                "classification_label":batch['classification_label'],
            }

    def validation_step_end(self, outputs):
        loss = outputs['val_loss']
        self.log("val/loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True,
                      )
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=False, on_epoch=True,
                      sync_dist=True,
                      )
        self.step_end_compute_update_metrics(self, outputs)

    def on_validation_epoch_end(self) -> None:
        self.epoch_end_show_metrics(self,'val')

    def test_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        output = self(x)
        backbone_feat,logits = output['backbone_features'],output['out']
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        return {'preds':preds,'y':y}

    def test_step_end(self, outputs):
        self.step_end_compute_update_metrics(self, outputs, 'test')

    def on_test_epoch_end(self) -> None:
        self.epoch_end_show_metrics(self, 'test')

    def setup(self, stage: str):
        if stage == 'fit':
            limit_batches = self.trainer.limit_train_batches
            batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
            batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
            num_devices = max(1, self.trainer.num_devices)
            effective_accum = self.trainer.accumulate_grad_batches * num_devices
            self.train_steps = (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Tuple[List, List]:

        return optimizer_config(self)


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

            src_predict_color, tgt_predict_color = self.gray2rgb(self,src_predict, tgt_predict)
            src_y_color, tgt_y_color = self.gray2rgb(self,src_label, tgt_label)

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
            y_color,predict_color = self.gray2rgb(self,y,predict)
            log["image"] = x
            log["label"] = y_color
            log["predict"] = predict_color
        return log
