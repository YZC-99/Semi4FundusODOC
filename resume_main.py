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
from segment.modules.dualsemibase import DualBase
from segment.prototype_dist_init import prototype_dist_init
from segment.dataloader.od_oc_dataset import SemiDataset,SemiUabledTrain
from segment.label import label

import argparse, os, sys, datetime, glob, importlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from utils.general import get_config_from_file, initialize_from_config, setup_callbacks,merge_cfg

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='domain_shift_semi/1_7/strong1/G1R7R4_B_CJ_semi')
    parser.add_argument( '--resume_path', type=str, default='xx')
    parser.add_argument('-s', '--seed', type=int, default=0)

    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ng', '--num_gpus', type=int, default=2)

    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=10000)
    parser.add_argument('-m', '--max_images', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=int, default=8)
    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # cfg.merge_from_file(Path("configs")/(args.config+".yaml"))
    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    config_dict = OmegaConf.to_container(config, resolve=True)
    # 将新的配置字典中的键添加到之前的CfgNode对象中
    merge_cfg(cfg, config_dict)
    loss_config = config['MODEL']['loss']

    now_experiment_path = Path("experiments")/(args.config)
    now_ex_pseudo_masks_path = os.path.join(now_experiment_path,'pseudo_masks')
    now_ex_prototypes_path = os.path.join(now_experiment_path,'prototypes')
    now_ex_logs_path = os.path.join(now_experiment_path,'logs')
    now_ex_models_path = os.path.join(now_experiment_path,'models')

    if not os.path.exists(now_experiment_path):
        os.makedirs(now_experiment_path)
    if not os.path.exists(now_ex_prototypes_path):
        os.makedirs(now_ex_prototypes_path)
    if not os.path.exists(now_ex_pseudo_masks_path):
        os.makedirs(now_ex_pseudo_masks_path)

    cfg.MODEL.logs_path = now_ex_logs_path
    cfg.MODEL.save_path = now_ex_models_path
    cfg.prototype_path = now_ex_prototypes_path
    cfg.MODEL.pseudo_mask_path = now_ex_pseudo_masks_path

    exp_config = OmegaConf.create({"name": args.config, "epochs": cfg.MODEL.epochs, "update_every": args.update_every,
                                    "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    if cfg.MODEL.Dual:
        model = DualBase('resnet50', cfg.MODEL.NUM_CLASSES, cfg)
    else:
        model = Base(cfg.MODEL.model,cfg.MODEL.backbone,cfg.MODEL.NUM_CLASSES,cfg,loss_config)

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)



    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()




    # Build trainer
    trainer = pl.Trainer(max_epochs=200,
                         precision=16 if exp_config.use_amp else 32,
                         resume_from_checkpoint=args.resume_path,
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy="ddp" if args.num_nodes > 1 or args.num_gpus > 1 else None,
                         accumulate_grad_batches=exp_config.update_every,
                         )

    # Train
    trainer.fit(model, data)

