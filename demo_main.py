# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

##
import yaml
from torch.utils.data import DataLoader
from segment.configs import cfg

import argparse, os, sys, datetime, glob, importlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from utils.general import get_config_from_file, initialize_from_config, setup_callbacks,merge_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='domain_shift_semi/1_7/strong1/G1R7R4_B_CJ_semi')
    parser.add_argument('-s', '--seed', type=int, default=42)

    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ng', '--num_gpus', type=int, default=2)

    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=10000)
    parser.add_argument('-m', '--max_images', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=int, default=8)
    parser.add_argument('-tune', default=False, action='store_true')
    parser.add_argument('-saw', default=False, action='store_true')
    parser.add_argument('-abs','--auto_scale_batch_size', default=False, action='store_true')
    parser.add_argument('-alf','--auto_lr_find', default=False, action='store_true')
    parser.add_argument('-v','--check_val_every_n_epoch', type=int, default=1)
    args = parser.parse_args()


    # cfg.merge_from_file(Path("configs")/(args.config+".yaml"))
    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    config_dict = OmegaConf.to_container(config, resolve=True)
    # 将新的配置字典中的键添加到之前的CfgNode对象中
    merge_cfg(cfg, config_dict)




    exp_config = OmegaConf.create({"name": args.config, "epochs": cfg.MODEL.epochs, "update_every": args.update_every,
                                    "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})


    print(config)






