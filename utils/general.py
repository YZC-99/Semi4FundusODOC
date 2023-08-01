# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import random
import importlib
import pathlib
from typing import Tuple, List, Dict, ClassVar

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
import torch.distributed as dist
from .callback import *

def get_obj_from_str(name: str, reload: bool = False) -> ClassVar:
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False) -> ClassVar:
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
        
    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def setup_callbacks(exp_config: OmegaConf, config: OmegaConf) -> Tuple[List[Callback], TensorBoardLogger]:
    # now = datetime.now().strftime('%d%m%Y_%H%M%S')
    basedir = pathlib.Path("experiments", exp_config.name)
    if dist.is_initialized() and dist.get_rank() == 0:
        os.makedirs(basedir, exist_ok=True)

    setup_callback = SetupCallback(config, exp_config, basedir)
    on_best_mIoU = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_mIoU:.6f}-{val_mDice:.6f}-{val_OD_dice_score:.6f}-{val_OD_IoU:.6f}-{val_OC_dice_score:.6f}-{val_OC_IoU:.6f}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )
    on_best_mDice = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_mDice:.6f}-{val_mIoU:.6f}-{val_OD_dice_score:.6f}-{val_OD_IoU:.6f}-{val_OC_dice_score:.6f}-{val_OC_IoU:.6f}",
        monitor="val_mDice",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    on_best_OD_Dice = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_OD_dice_score:.6f}-{val_mIoU:.6f}-{val_mDice:.6f}-{val_OD_IoU:.6f}-{val_OC_dice_score:.6f}-{val_OC_IoU:.6f}",
        monitor="val_OD_dice_score",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    on_best_OD_IoU = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_OD_IoU:.6f}-{val_mIoU:.6f}-{val_OD_dice_score:.6f}-{val_mDice:.6f}-{val_OC_dice_score:.6f}-{val_OC_IoU:.6f}",
        monitor="val_OD_IoU",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    on_best_OC_Dice = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_OC_dice_score:.6f}-{val_mIoU:.6f}-{val_OD_dice_score:.6f}-{val_OD_IoU:.6f}-{val_mDice:.6f}-{val_OC_IoU:.6f}",
        monitor="val_OC_dice_score",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    on_best_OC_IoU = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="{val_OC_IoU:.6f}-{val_mIoU:.6f}-{val_OD_dice_score:.6f}-{val_OD_IoU:.6f}-{val_mDice:.6f}-{val_OC_dice_score:.6f}",
        monitor="val_OC_IoU",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        os.makedirs(setup_callback.logdir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=str(setup_callback.logdir))
    # csv_logger = CSVLogger(str(setup_callback.logdir), 'results.csv')
    logger_img_callback = ImageLogger(exp_config.batch_frequency, exp_config.max_images)
    model_architecture_callback = ModelArchitectureCallback(path=str(setup_callback.logdir))
    # return [setup_callback, checkpoint_callback, logger_img_callback,model_architecture_callback], logger
    if config.MODEL.NUM_CLASSES == 3:
        return [setup_callback, on_best_mIoU,on_best_mDice,on_best_OD_Dice,on_best_OD_IoU,on_best_OC_Dice,on_best_OC_IoU,logger_img_callback], logger
    return [setup_callback, on_best_mIoU,on_best_mDice,on_best_OD_Dice,on_best_OD_IoU,logger_img_callback], logger


def get_config_from_file(config_file: str) -> Dict:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)
    
    return config_file
