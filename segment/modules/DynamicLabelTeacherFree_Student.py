import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import normal
from torch.optim import SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from segment.util import meanIOU
from segment.losses.loss import PrototypeContrastiveLoss
from segment.losses.seg.boundary_loss import DC_and_BD_loss

from segment.losses.grw_cross_entropy_loss import GRWCrossEntropyLoss,Dice_GRWCrossEntropyLoss
from segment.losses.blv_loss import BlvLoss
from segment.losses.lovasz_loss import lovasz_softmax
from segment.modules.prototype_dist_estimator import prototype_dist_estimator
from typing import List,Tuple, Dict, Any, Optional
import pytorch_lightning as pl
import cleanlab
import torchmetrics
from torchmetrics import JaccardIndex,Dice
from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
from segment.modules.semseg.deeplabv2 import DeepLabV2

from segment.modules.semseg.unet import UNet
from segment.ramps import sigmoid_rampup
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
        if cfg.MODEL.DC_BD_loss:
            self.DC_BD_loss = DC_and_BD_loss()
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


        if self.cfg.MODEL.NUM_CLASSES == 3:
            self.val_oc_dice_score = Dice(num_classes=1,multiclass=False).to(self.device)
            self.val_oc_withB_dice_score = Dice(num_classes=2, average='macro').to(self.device)
            self.val_oc_multiclass_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)
            self.val_oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary').to(self.device)
            self.val_oc_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(self.device)
            self.val_oc_multiclass_boundary_jaccard = BoundaryIoU(num_classes=2, task='multiclass').to(self.device)

            self.val_od_rmOC_dice_score = Dice(num_classes=1,multiclass=False).to(self.device)
            self.val_od_rmOC_jaccard = JaccardIndex(num_classes=2, task='multiclass').to(self.device)


        if cfg.MODEL.Teacher_pretrined:
            self.init_from_ckpt(cfg.MODEL.stage1_ckpt_path, ignore_keys='',model='teacher')
        if cfg.MODEL.Student_pretrined:
            self.init_from_ckpt(cfg.MODEL.stage1_ckpt_path, ignore_keys='',model='student')

    def forward(self, HQ_input,LQ_input) -> Dict[str, torch.Tensor]:
        # train student
        HQ_input = torch.cat([HQ_input, LQ_input], dim=0)
        HQ_output = self.model(HQ_input)
        return {'HQ_output':HQ_output}

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

    def init_from_ckpt(self,path: str,ignore_keys: List[str] = list(),model='student'):
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
            if model=='student':
                self.model.load_state_dict(sd, strict=False)
            elif model=='teacher':
                self.ema_model.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")



    def get_current_consistency_weight(self):
        consistency = 0.1
        consistency_rampup = 200.0
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return consistency * sigmoid_rampup(self.trainer.current_epoch, consistency_rampup)

    def get_confident_maps(self,masks,preds):
        b,_,w,h = preds.shape
        preds_np = preds.cpu().detach().numpy()
        masks_np = masks.cpu().detach().numpy()

        preds_softmax_np_accumulated = np.swapaxes(preds_np, 1, 2)
        preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
        preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, self.num_classes)
        preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
        masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                   prune_method='both', n_jobs=1)
        confident_maps_np = noise.reshape(-1, w, h).astype(np.uint8)
        confident_maps = torch.from_numpy(confident_maps_np).to(self.device)
        confident_maps = confident_maps.repeat(b,1,1,1)
        return confident_maps

    def training_step(self, batch):
        HQ, LQ = batch
        HQ_input, LQ_input,HQ_label, LQ_label = HQ['img'], LQ['img'], HQ['mask'], LQ['mask']
        # HQ_input = torch.cat([HQ_input,LQ_input],dim=0)
        out = self(HQ_input,LQ_input)
        HQ_output = out['HQ_output']

        self.ema_model.eval()

        LQ_output = self.ema_model(LQ_input)
        HQ_logits,LQ_logits = HQ_output['out'],LQ_output['out']

        HQ_outputs_soft = torch.softmax(HQ_logits, dim=1)
        LQ_outputs_soft = torch.softmax(LQ_logits, dim=1)
        LQ_pseudo = LQ_outputs_soft.argmax(1)

        HQ_label = torch.cat([HQ_label,LQ_pseudo],dim=0)

        loss = self.loss(HQ_logits,HQ_label)
        # train teacher
        consistency_loss = torch.mean((HQ_outputs_soft[LQ_outputs_soft.shape[0]:].float()  - LQ_outputs_soft.float() ) ** 2)

        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True,rank_zero_only=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,rank_zero_only=True)
        self.update_ema_variables()

        consistency_weight = self.get_current_consistency_weight()
        if self.cfg.MODEL.weighted_loss:
            all_loss = loss + consistency_weight*(consistency_loss)
        else:
            all_loss = loss
        return all_loss

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

        LQ_output = self.ema_model(x)
        LQ_logits = LQ_output['out']
        LQ_preds = nn.functional.softmax(LQ_logits, dim=1).argmax(1)

        consistency_loss = torch.mean((HQ_preds[LQ_preds.shape[0]:].float()  - LQ_preds.float() ) ** 2)
        consistency_weight = self.get_current_consistency_weight()
        all_loss = loss + consistency_weight*( consistency_loss)

        return {'val_loss':all_loss,
                'y': torch.cat([y,y],dim=0),
                'preds':HQ_preds,
                'ema_preds':LQ_preds
                }

    def validation_step_end(self, outputs):
        loss,preds,y,ema_preds= outputs['val_loss'],outputs['preds'],outputs['y'],outputs['ema_preds']
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


    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()

        if isinstance(batch, tuple):
            HQ, LQ = batch
            x, LQ_input, y, LQ_label = HQ['img'], LQ['img'], HQ['mask'], LQ['mask']
            out = self(x, LQ_input)
        elif isinstance(batch, dict):
            x = batch['img']
            y = batch['mask']
            out = self(x,x)
        HQ_output = out['HQ_output']
        HQ_logits = HQ_output['out']
        HQ_preds = nn.functional.softmax(HQ_logits, dim=1).argmax(1)

        LQ_output = self.ema_model(LQ_input)
        LQ_logits = LQ_output['out']
        LQ_preds = nn.functional.softmax(LQ_logits, dim=1).argmax(1)


        HQ_preds_color, LQ_preds_color = self.gray2rgb(HQ_preds, LQ_preds)
        y_color, _ = self.gray2rgb(y, HQ_preds)
        log["image"] = x
        log["label"] = y_color
        log["student_pred"] = HQ_preds_color
        log["teacher_pred"] = LQ_preds_color
        return log
