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
from segment.prototype_dist_init import prototype_dist_init
from segment.dataloader.od_oc_dataset import SemiDataset,SemiUabledTrain
from segment.label import label

import argparse, os, sys, datetime, glob, importlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from utils.general import get_config_from_file, initialize_from_config, setup_callbacks

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='domain_shift_sup/G1R7R4_sup_uda')
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

    cfg.merge_from_file(Path("configs")/(args.config+".yaml"))
    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    now_experiment_path = Path("experiments")/(args.config)
    now_ex_pseudo_masks_path = os.path.join(now_experiment_path,'pseudo_masks')
    now_ex_prototypes_path = os.path.join(now_experiment_path,'prototypes')
    now_ex_logs_path = os.path.join(now_experiment_path,'logs')
    now_ex_models_path = os.path.join(now_experiment_path,'models')

    if not os.path.exists(now_experiment_path):
        os.makedirs(now_experiment_path)
    if not os.path.exists(now_ex_prototypes_path):
        os.makedirs(now_ex_prototypes_path)

    cfg.MODEL.logs_path = now_ex_logs_path
    cfg.MODEL.save_path = now_ex_models_path
    cfg.prototype_path = now_ex_prototypes_path
    cfg.pseudo_mask_path = now_ex_pseudo_masks_path

    exp_config = OmegaConf.create({"name": args.config, "epochs": cfg.MODEL.epochs, "update_every": args.update_every,
                                    "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    model = Base('resnet50',cfg.MODEL.NUM_CLASSES,cfg)
    model.learning_rate = cfg.MODEL.lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    if len(os.listdir(cfg.prototype_path)) < 2 and cfg.MODEL.uda:
        src_dataset = initialize_from_config(config.dataset.params['train'])
        src_dataloader = DataLoader(src_dataset, batch_size=cfg.MODEL.batch_size,
                                    num_workers=8, shuffle=True, drop_last=True)
        print('>>>>>>>>>>>>>>>>正在计算 prototypes >>>>>>>>>>>>>>>>')
        prototype_dist_init(cfg, src_train_loader= src_dataloader)
    if cfg.MODEL.label:
        unlabeled_dataset = SemiUabledTrain(task=cfg.dataset.params.train.params.task,
                                            name=cfg.dataset.params.train.params.name,
                                            root=cfg.dataset.params.train.params.root,
                                            mode='label',
                                            size=cfg.dataset.params.train.params.size,
                                            labeled_id_path=None,
                                            unlabeled_id_path=cfg.dataset.params.train.params.unlabeled_id_path,
                                            pseudo_mask_path=None,cfg=cfg)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False,
                                     pin_memory=True, num_workers=8, drop_last=False)
        # 这里model应当是重新初始化的
        label(unlabeled_dataloader,cfg)

    # 如果是semi训练的话，是需要修改配置文件中的pseudo_masks_path的
    if cfg.MODEL.dataset == 'semi':
        cfg.dataset.params.train.params.unlabeled_id_path = now_ex_pseudo_masks_path

    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()




    # Build trainer
    trainer = pl.Trainer(max_epochs=exp_config.epochs,
                         precision=16 if exp_config.use_amp else 32,
                         callbacks=callbacks,
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy="ddp" if args.num_nodes > 1 or args.num_gpus > 1 else None,
                         accumulate_grad_batches=exp_config.update_every,
                         logger=logger,
                         )

    # Train
    trainer.fit(model, data)

