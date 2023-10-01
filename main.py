# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

##
import os
import sys
from yacs.config import CfgNode as CN
import yaml
from torch.utils.data import DataLoader
from segment.configs import cfg
from segment.modules.semibase import Base
from segment.modules.Teacher_Student import TSBase
from segment.modules.dualsemibase import DualBase
from segment.prototype_dist_init import prototype_dist_init
from segment.dataloader.od_oc_dataset import SemiDataset, SemiUabledTrain
from segment.label import label

import argparse, os, sys, datetime, glob, importlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from utils.general import get_config_from_file, initialize_from_config, setup_callbacks, merge_cfg, get_random_seed
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, StochasticWeightAveraging


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=-0.1)
    parser.add_argument('--warmup', type=float, default=-0.1)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--DC_loss', type=float, default=-0.1)
    parser.add_argument('--BD_loss', type=float, default=-0.1)
    parser.add_argument('--BD_loss_increase_alpha', type=float, default=-0.1)
    parser.add_argument('--BD_Contrast_rebalance_loss', type=bool, default=False)
    parser.add_argument('--FC_loss', type=float, default=-0.1)
    parser.add_argument('--IoU_loss', type=float, default=-0.1)
    parser.add_argument('--CEpair_loss', type=float, default=-0.1)
    parser.add_argument('--ContrastCrossPixelCorrect_loss', type=float, default=-0.1)
    parser.add_argument('--ohem_thres', type=float, default=0.5)
    parser.add_argument('--ohem_weight', type=float,nargs="+", default=[1.0,2.8,3.0])
    parser.add_argument('--Attention', type=str, default=None)
    parser.add_argument('--optimizer_decoupling', type=int, default=10)

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='b2')

    parser.add_argument('-c', '--config', type=str, default='domain_shift_semi/1_7/strong1/G1R7R4_B_CJ_semi')
    parser.add_argument('-s', '--seed', type=int, default=42)

    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ng', '--num_gpus', type=int, default=1)

    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-gc', '--grad_clip', default=0.0, type=float)

    parser.add_argument('-b', '--batch_frequency', type=int, default=10000)
    parser.add_argument('-m', '--max_images', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=int, default=8)
    parser.add_argument('-tune', default=False, action='store_true')
    parser.add_argument('-saw', default=False, action='store_true')
    parser.add_argument('-abs', '--auto_scale_batch_size', default=False, action='store_true')
    parser.add_argument('-alf', '--auto_lr_find', default=False, action='store_true')
    parser.add_argument('-v', '--check_val_every_n_epoch', type=int, default=1)
    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)
    get_random_seed(args.seed)
    # cfg.merge_from_file(Path("configs")/(args.config+".yaml"))
    # Load configuration
    config = get_config_from_file(Path("configs") / (args.config + ".yaml"))
    # 修改新参数
    if args.lr >= 0:
        config.MODEL.lr = args.lr
    if args.warmup >= 0:
        config.MODEL.lr_warmup_steps_ratio = args.warmup
    if args.DC_loss >= 0:
        config.MODEL.DC_loss = args.DC_loss
    if args.BD_loss >= 0:
        config.MODEL.BD_loss = args.BD_loss
    if args.BD_loss_increase_alpha >= 0:
        config.MODEL.BD_loss_increase_alpha = args.BD_loss_increase_alpha
    if args.FC_loss >= 0:
        config.MODEL.FC_loss = args.FC_loss
    if args.IoU_loss >= 0:
        config.MODEL.IoU_loss = args.IoU_loss
    if args.CEpair_loss >= 0:
        config.MODEL.CEpair_loss = args.CEpair_loss
    if args.ContrastCrossPixelCorrect_loss >= 0:
        config.MODEL.ContrastCrossPixelCorrect_loss = args.ContrastCrossPixelCorrect_loss
    if args.Attention is not None:
        config.MODEL.Attention = args.Attention
    if args.model is not None:
        config.MODEL.model = args.model

    config.MODEL.backbone = args.backbone
    config.MODEL.optimizer_decoupling = args.optimizer_decoupling
    config.MODEL.scheduler = args.scheduler
    config.MODEL.epochs = args.epochs
    config.MODEL.BD_Contrast_rebalance_loss = args.BD_Contrast_rebalance_loss
    config.MODEL.loss.params.thres = args.ohem_thres
    config.MODEL.loss.params.weight = args.ohem_weight
    config.info.seed = args.seed
    config.info.setting = args.d

    config_dict = OmegaConf.to_container(config, resolve=True)
    # 将新的配置字典中的键添加到之前的CfgNode对象中
    merge_cfg(cfg, config_dict)
    loss_config = config['MODEL']['loss']

    now_experiment_path = Path("experiments") / (args.config)
    now_ex_pseudo_masks_path = os.path.join(now_experiment_path, 'pseudo_masks')
    now_ex_prototypes_path = os.path.join(now_experiment_path, 'prototypes')
    now_ex_logs_path = os.path.join(now_experiment_path, 'logs')

    if not os.path.exists(now_experiment_path):
        os.makedirs(now_experiment_path)
    if not os.path.exists(now_ex_prototypes_path):
        os.makedirs(now_ex_prototypes_path)
    if not os.path.exists(now_ex_logs_path):
        os.makedirs(now_ex_logs_path)

    cfg.prototype_path = now_ex_prototypes_path
    cfg.MODEL.pseudo_mask_path = now_ex_pseudo_masks_path
    cfg.MODEL.logs_path = now_ex_logs_path

    exp_config = OmegaConf.create({"name": args.config, "epochs": cfg.MODEL.epochs, "update_every": args.update_every,
                                   "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    if cfg.MODEL.Dual:
        model = DualBase('resnet50', cfg.MODEL.NUM_CLASSES, cfg)
    elif cfg.MODEL.Teacher_Student:
        model = TSBase(cfg.MODEL.model, cfg.MODEL.backbone, cfg.MODEL.NUM_CLASSES, cfg, loss_config)
    else:
        model = Base(cfg.MODEL.model, cfg.MODEL.backbone, cfg.MODEL.NUM_CLASSES, cfg, loss_config)
    model.learning_rate = cfg.MODEL.lr * args.num_gpus
    # Setup callbacks
    callbacks, logger, simple_Profiler = setup_callbacks(exp_config, config)

    if len(list(filter(lambda x: x.endswith('.pth'), os.listdir(cfg.prototype_path)))) < 2 and cfg.MODEL.uda:
        src_dataset = initialize_from_config(config.dataset.params['train'])
        src_dataloader = DataLoader(src_dataset, batch_size=1,
                                    num_workers=8, shuffle=True, drop_last=True)
        print('>>>>>>>>>>>>>>>>正在计算 prototypes >>>>>>>>>>>>>>>>')
        prototype_dist_init(cfg, src_train_loader=src_dataloader)

    if cfg.MODEL.label and len(os.listdir(cfg.MODEL.pseudo_mask_path)) == 0:
        unlabeled_dataset = SemiUabledTrain(task=cfg.dataset.params.train2.params.task,
                                            name=cfg.dataset.params.train2.params.name,
                                            root=cfg.dataset.params.train2.params.root,
                                            mode='label',
                                            size=cfg.dataset.params.train2.params.size,
                                            labeled_id_path=None,
                                            unlabeled_id_path=cfg.dataset.params.train2.params.unlabeled_id_path,
                                            add_unlabeled_id_path=cfg.dataset.params.train2.params.add_unlabeled_id_path,
                                            pseudo_mask_path=None)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False,
                                          pin_memory=True, num_workers=8, drop_last=False)

        ckpt_path = cfg.MODEL.stage1_ckpt_path

        label(unlabeled_dataloader, ckpt_path, cfg)

    # 二次训练
    if cfg.MODEL.retraining and len(os.listdir(cfg.MODEL.pseudo_mask_path)) == 0:
        print('>>>>>>>>>>>>>>>>二次训练 >>>>>>>>>>>>>>>>')
        unlabeled_dataset = SemiUabledTrain(task=cfg.dataset.params.train.params.task,
                                            name=cfg.dataset.params.train.params.name,
                                            root=cfg.dataset.params.train.params.root,
                                            mode='label',
                                            size=cfg.dataset.params.train.params.size,
                                            labeled_id_path=None,
                                            unlabeled_id_path=cfg.dataset.params.train.params.unlabeled_id_path,
                                            pseudo_mask_path=None)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False,
                                          pin_memory=True, num_workers=8, drop_last=False)
        ckpt_path = cfg.MODEL.stage2_ckpt_path
        label(unlabeled_dataloader, ckpt_path, cfg)

    # 如果是semi训练的话，是需要修改配置文件中的pseudo_masks_path的
    if cfg.dataset.params.train2.target != '' and (cfg.MODEL.uda or cfg.MODEL.Teacher_Student):
        config.dataset.params.train2.params.pseudo_mask_path = now_ex_pseudo_masks_path
        config.dataset.params.train2.params.labeled_id_path = config.dataset.params.train.params.labeled_id_path
    else:
        config.dataset.params.train.params.pseudo_mask_path = now_ex_pseudo_masks_path

    # 设置此时不进行uda
    if not cfg.MODEL.uda and not cfg.MODEL.Teacher_Student:
        config.dataset.params.train2 = None
    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()

    if args.saw:
        SWA_callback = StochasticWeightAveraging(swa_lrs=cfg.MODEL.lr * 0.1, swa_epoch_start=50, annealing_epochs=50)
        callbacks.append(SWA_callback)

    # Build trainer
    trainer = pl.Trainer(max_epochs=exp_config.epochs,
                         precision=16 if exp_config.use_amp else 32,
                         callbacks=callbacks,
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy="ddp" if args.num_nodes > 1 or args.num_gpus > 1 else None,
                         accumulate_grad_batches=exp_config.update_every,
                         logger=logger,
                         profiler=simple_Profiler,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         auto_lr_find=args.auto_lr_find,
                         gradient_clip_val=args.update_every,
                         )

    if args.auto_lr_find and args.tune:
        trainer.tune(model, data)
    # Train
    trainer.fit(model, data, ckpt_path=cfg.MODEL.resume_path)

