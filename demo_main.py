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
from yacs.config import CfgNode
from utils.general import get_config_from_file,merge_cfg
#
# def merge_cfg(cfg_node, config_dict):
#     for key, value in config_dict.items():
#         if isinstance(value, dict):
#             # 如果键不存在于cfg_node中，或者cfg_node中的值不是CfgNode对象，创建一个新的CfgNode
#             if key not in cfg_node or not isinstance(cfg_node[key], CfgNode):
#                 cfg_node[key] = CfgNode()
#             # 递归融合
#             merge_cfg(cfg_node[key], value)
#         else:
#             # 否则，直接使用config_dict中的值更新cfg_node中的值
#             cfg_node[key] = value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='test')
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
    base = get_config_from_file(Path("configs")/("base.yaml"))
    config_dict = OmegaConf.to_container(config, resolve=True)
    # 将新的配置字典中的键添加到之前的CfgNode对象中
    merge_cfg(cfg, config_dict)




    exp_config = OmegaConf.create({"name": args.config, "epochs": cfg.MODEL.epochs, "update_every": args.update_every,
                                    "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    from omegaconf import DictConfig
    cfg_node_dict = cfg.__dict__
    DictConfig(cfg_node_dict)
    print(config)
    print(cfg)






