import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from segment.util import meanIOU
from segment.losses.loss import PrototypeContrastiveLoss
from segment.losses.lovasz_loss import lovasz_softmax
from segment.modules.prototype_dist_estimator import prototype_dist_estimator
from typing import List,Tuple, Dict, Any, Optional
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex,Dice

from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus

import numpy as np
import matplotlib.pyplot as plt

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
                 backbone: str,
                 num_classes: int,
                 cfg
                 ):
        super(Base, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.num_classes = num_classes
        self.model = DeepLabV3Plus(self.backbone,self.num_classes)
        self.loss = CrossEntropyLoss(ignore_index=255)
        self.color_map = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128]}


        if cfg.MODEL.stage1_ckpt_path is not None:
            self.init_from_ckpt(cfg.MODEL.stage1_ckpt_path, ignore_keys='')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        return out

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
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()
        return x.contiguous()

    def uda_train(self,batch):
        src, tgt = batch
        src_input, src_label, tgt_input, tgt_label = src['img'], src['mask'], tgt['img'], tgt['mask']

        ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
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
            loss_sup = ce_criterion(src_pred, src_label) + self.cfg.SOLVER.LAMBDA_LOV * loss_lov
        else:
            loss_sup = ce_criterion(src_pred, src_label)

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

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        if self.cfg.MODEL.uda:
            loss = self.uda_train(batch)
        else:
            x = batch['img']
            y = batch['mask']
            output = self(x)
            backbone_feat,logits = output['backbone_features'],output['out']
            loss = self.loss(logits, y)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,rank_zero_only=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,rank_zero_only=True)
        return loss



    def on_validation_start(self) -> None:
        self.dice_score = torchmetrics.Dice(num_classes=self.cfg.MODEL.NUM_CLASSES,average='macro')
        self.jaccard = torchmetrics.JaccardIndex(num_classes=self.cfg.MODEL.NUM_CLASSES,task='binary' if self.cfg.MODEL.NUM_CLASSES ==  2 else 'multiclass')

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch['img']
        y = batch['mask']
        output = self(x)
        backbone_feat,logits = output['backbone_features'],output['out']
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        loss = self.loss(logits, y)
        self.dice_score(preds, y)
        self.jaccard(preds, y)

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True,rank_zero_only=True)
        self.log("val/mIoU", self.jaccard, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True,rank_zero_only=True)
        self.log("val/dice_score", self.dice_score, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True,rank_zero_only=True)
        return loss


    def on_validation_epoch_end(self) -> None:
        self.log("val/mIoU", self.jaccard.compute(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,rank_zero_only=True)
        self.log(f"val/dice_score", self.dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,rank_zero_only=True)


    def configure_optimizers(self) -> Tuple[List, List]:
        # print(self.train_dataloader)


        lr = self.cfg.MODEL.lr
        total_iters = self.trainer.max_steps
        optimizers = [SGD(self.model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-4)]
        lambda_lr = lambda iters: lr * (1 - iters / total_iters) ** 0.9
        scheduler = LambdaLR(optimizers[0],lr_lambda=lambda_lr)
        schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]
        return optimizers, schedulers
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        if self.cfg.MODEL.uda:
            return
        log = dict()
        x = batch['img'].to(self.device)
        y = batch['mask']
        # log["originals"] = x
        out = self(x)['out']
        # cam,overcam = self.get_cam(x,out)

        out = torch.nn.functional.softmax(out,dim=1)
        predict = out.argmax(1)

        y_color,predict_color = self.gray2rgb(y,predict)
        log["image"] = x
        log["label"] = y_color
        log["predict"] = predict_color
        return log
